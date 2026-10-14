"""CPU suffix fine-tuning with lossless FP32 prefix caching and safe checkpoints.

Only the final two BERT blocks are trainable by default. Every forward uses eval
mode (including during gradient updates), so dropout is disabled. Prefix entries
are uncompressed, variable-length NPZ arrays, one text/ID per file; padding is
reintroduced when a training batch is assembled. Batched cache construction can
change floating-point reduction details across batch shapes, so the explicit
equivalence check reports bit equality and requires an absolute tolerance of
1e-6 by default. No approximate/quantized hidden-state cache is used.
Suffix forwards sort by true length and use small separately padded microbatches,
then restore the original order before the single, complete in-batch loss.

This module performs no downloads, mining, validation-based model selection or
test evaluation. Callers own the immutable plan, epoch order, validation and
training budget. Teacher embeddings may have a different dimension: only their
query-document cosine distributions are distilled. KL is NOT multiplied by T².
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Sequence

import numpy as np

from .adaptation import AdaptationConfig, AdaptedEncoder, _dependencies

LOGGER = logging.getLogger(__name__)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _tensor_digest(tensors: dict) -> str:
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
    return result.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    """Hash exactly the existing C-order bytes without allocating another copy."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


@dataclass(frozen=True)
class DistillConfig:
    suffix_layers: int = 2
    max_length: int = 256
    threads: int = 2
    forward_microbatch_size: int = 8
    seed: int = 42
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ce_temperature: float = 0.05
    kd_temperature: float = 0.05
    kd_weight: float = 1.0
    anchor_weight: float = 1.0

    def validate(self) -> None:
        for name in ("suffix_layers", "max_length", "threads", "forward_microbatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "max_grad_norm", "ce_temperature", "kd_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "kd_weight", "anchor_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class DistillStudent:
    """Official floating-point BERT; train a suffix while caching its frozen prefix."""

    def __init__(self, base_model_dir: Path, config: DistillConfig | None = None,
                 prefix_cache_dir: Path | None = None):
        self.config = config or DistillConfig()
        self.config.validate()
        torch, *_ = _dependencies()
        self.encoder = AdaptedEncoder(Path(base_model_dir), AdaptationConfig(
            seed=self.config.seed, max_length=self.config.max_length, threads=self.config.threads))
        self.model, self.tokenizer = self.encoder.model, self.encoder.tokenizer
        layer_count = len(self.model.encoder.layer)
        if self.config.suffix_layers >= layer_count:
            raise ValueError("suffix_layers must leave at least one frozen Transformer block")
        self.frozen_layers = layer_count - self.config.suffix_layers
        self.suffix = list(self.model.encoder.layer[self.frozen_layers:])
        self.base_suffix = copy.deepcopy(self.suffix)
        for layer in self.base_suffix:
            layer.eval()
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        for layer in self.suffix:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        self.model.eval()
        self.dimension = self.model.config.hidden_size
        self.parameters = {name: parameter for name, parameter in self.model.named_parameters()
                           if parameter.requires_grad}
        self.trainable_parameter_count = sum(parameter.numel() for parameter in self.parameters.values())
        self.initial_suffix_sha256 = _tensor_digest(self.parameters)
        self.identity = {
            "schema": 1, "architecture": "BertModel full-parameter Transformer suffix",
            "base_model_sha256": self.encoder.metadata["base_model_sha256"],
            "config_sha256": self.encoder.metadata["config_sha256"],
            "tokenizer_sha256": self.encoder.metadata["tokenizer_sha256"],
            "max_length": self.config.max_length, "dimension": self.dimension,
            "layer_count": layer_count, "frozen_layers": self.frozen_layers,
            "trainable_layers": list(range(self.frozen_layers, layer_count)),
            "dtype": "float32", "pooling": "attention-mask-mean", "normalize": True,
            "training_mode": "eval with gradients; dropout disabled in all blocks",
            "torch_version": torch.__version__, "threads": self.config.threads,
            "suffix_forward_strategy": "stable length-descending microbatches; inverse permutation before shared loss",
            "forward_microbatch_size": self.config.forward_microbatch_size,
        }
        # Suffix execution shape cannot change the frozen prefix content; retain
        # the exact pre-existing prefix identity when this strategy changes.
        self.cache_identity = {key: value for key, value in self.identity.items()
                               if key not in {"suffix_forward_strategy", "forward_microbatch_size"}}
        self.cache_identity["schema"] = "prefix-fp32-npz-v1"
        self.metadata = dict(self.identity, trainable_parameter_count=self.trainable_parameter_count,
                             initial_suffix_sha256=self.initial_suffix_sha256)
        self.prefix_cache_dir = Path(prefix_cache_dir) if prefix_cache_dir is not None else None
        self.cache_root = (self.prefix_cache_dir / _digest(self.cache_identity)[:20]
                           if self.prefix_cache_dir is not None else None)
        self.last_forward_stats = None

    def _inputs(self, texts: Sequence[str]):
        torch, *_ = _dependencies()
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be a nonempty sequence of strings")
        encoded = self.tokenizer.encode_batch(list(texts))
        return {"input_ids": torch.tensor([row.ids for row in encoded], dtype=torch.long),
                "attention_mask": torch.tensor([row.attention_mask for row in encoded], dtype=torch.long),
                "token_type_ids": torch.tensor([row.type_ids for row in encoded], dtype=torch.long)}

    def _uncached_prefixes(self, texts: Sequence[str], *, with_base: bool = False):
        torch, *_ = _dependencies()
        self.model.eval()
        with torch.no_grad():
            inputs = self._inputs(texts)
            hidden = self.model.embeddings(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
            mask = self.model.get_extended_attention_mask(inputs["attention_mask"], inputs["input_ids"].shape)
            for layer in self.model.encoder.layer[:self.frozen_layers]:
                hidden = layer(hidden, attention_mask=mask)[0]
            prefixes = [hidden[index, :int(length)].detach().cpu().numpy().copy()
                        for index, length in enumerate(inputs["attention_mask"].sum(dim=1))]
            if not with_base:
                return prefixes
            for layer in self.base_suffix:
                hidden = layer(hidden, attention_mask=mask)[0]
            pooling_mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * pooling_mask).sum(dim=1) / pooling_mask.sum(dim=1)
            base = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().copy()
            return prefixes, base

    def _entry(self, text: str, prefix_id: str) -> tuple[Path, dict]:
        if self.cache_root is None:
            raise ValueError("prefix_cache_dir is required for disk caching")
        identity = {"prefix_identity_sha256": _digest(self.cache_identity), "prefix_id": str(prefix_id),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        key = _digest(identity)
        return self.cache_root / f"{key[:24]}.npz", dict(identity, key_sha256=key)

    def _read_entry(self, path: Path, identity: dict, *, include_base: bool = False):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"hidden", "base_vector", "metadata"}:
                    raise ValueError("Unexpected prefix cache fields")
                metadata = json.loads(str(archive["metadata"].item()))
                hidden = archive["hidden"]
                base = archive["base_vector"]
                if metadata["identity"] != identity:
                    raise ValueError("Prefix cache identity mismatch")
                if (hidden.dtype != np.float32 or hidden.ndim != 2 or hidden.shape[1] != self.dimension
                        or not 1 <= hidden.shape[0] <= self.config.max_length or not np.isfinite(hidden).all()
                        or _array_sha(hidden) != metadata["hidden_sha256"]):
                    raise ValueError("Prefix cache shape, dtype, values or checksum mismatch")
                if (base.dtype != np.float32 or base.shape != (self.dimension,) or not np.isfinite(base).all()
                        or abs(float(np.linalg.norm(base)) - 1) > 1e-5
                        or _array_sha(base) != metadata["base_sha256"]):
                    raise ValueError("Frozen base vector shape, norm or checksum mismatch")
                # NPZ members are already independently allocated arrays, not
                # views into the closed ZipFile. Do not copy every full prefix.
                return (hidden, base) if include_base else hidden
        except FileNotFoundError:
            raise
        except (KeyError, OSError, EOFError) as exc:
            raise ValueError(f"Invalid prefix cache: {path}") from exc

    def _write_entry(self, path: Path, identity: dict, hidden: np.ndarray, base: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"identity": identity, "hidden_sha256": _array_sha(hidden), "base_sha256": _array_sha(base)}
        descriptor, temporary = tempfile.mkstemp(prefix=".prefix-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(handle, hidden=hidden, base_vector=base, metadata=np.array(json.dumps(metadata, sort_keys=True)))
                handle.flush()
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def cache_prefixes(self, texts: Sequence[str], prefix_ids: Sequence[str], *, batch_size: int = 8) -> dict:
        """Build durable variable-length FP32 entries and return measured cache costs."""
        if len(texts) != len(prefix_ids) or batch_size < 1:
            raise ValueError("Prefix IDs/texts must align and batch_size must be positive")
        started, inference_seconds, hits, written_bytes = time.perf_counter(), 0.0, 0, 0
        pending, seen = {}, {}
        for text, prefix_id in zip(texts, prefix_ids, strict=True):
            path, identity = self._entry(text, prefix_id)
            if path in seen:
                if seen[path] != identity:
                    raise ValueError("Prefix cache path collision")
                hits += 1
            else:
                seen[path] = identity
                try:
                    self._read_entry(path, identity)
                    hits += 1
                except FileNotFoundError:
                    pending[path] = (identity, text)
        rows = list(pending.items())
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            then = time.perf_counter()
            values, base_vectors = self._uncached_prefixes([row[1][1] for row in batch], with_base=True)
            inference_seconds += time.perf_counter() - then
            for (path, (identity, _)), hidden, base in zip(batch, values, base_vectors, strict=True):
                self._write_entry(path, identity, hidden, base)
                written_bytes += path.stat().st_size
        return {"items": len(texts), "cache_hits": hits, "new_entries": len(rows),
                "bytes_written": written_bytes, "prefix_and_base_inference_seconds": inference_seconds,
                "wall_seconds": time.perf_counter() - started, "batch_size": batch_size,
                "format": "one uncompressed variable-length FP32 NPZ per text/ID, plus immutable base vector; no padding stored"}

    def _cached_batch(self, texts: Sequence[str], prefix_ids: Sequence[str], *, include_base: bool = False):
        """Read/verify each unique file once per call; never retain stale in-memory entries."""
        entries = [self._entry(text, prefix_id) for text, prefix_id in zip(texts, prefix_ids, strict=True)]
        loaded, identities, missing = {}, {}, {}
        for (path, identity), text, prefix_id in zip(entries, texts, prefix_ids, strict=True):
            if path in identities:
                if identities[path] != identity:
                    raise ValueError("Prefix cache path collision")
                continue
            identities[path] = identity
            try:
                loaded[path] = self._read_entry(path, identity, include_base=include_base)
            except FileNotFoundError:
                missing[path] = (text, prefix_id)
        if missing:
            rows = list(missing.values())
            self.cache_prefixes([row[0] for row in rows], [row[1] for row in rows], batch_size=min(8, len(rows)))
            for path in missing:
                loaded[path] = self._read_entry(path, identities[path], include_base=include_base)
        result, returned = [], set()
        for path, _ in entries:
            value = loaded[path]
            # Preserve independent mutable outputs for duplicate callers while
            # avoiding duplicate disk reads and copying every unique prefix.
            if path in returned:
                value = tuple(array.copy() for array in value) if include_base else value.copy()
            result.append(value)
            returned.add(path)
        return result

    def base_vectors(self, texts: Sequence[str], prefix_ids: Sequence[str] | None = None) -> np.ndarray:
        """Read original frozen-base anchor vectors; never use the current trained suffix."""
        if prefix_ids is None:
            prefix_ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        if len(texts) != len(prefix_ids):
            raise ValueError("Prefix IDs/texts must align")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.cache_root is None:
            return self._uncached_prefixes(texts, with_base=True)[1]
        return np.stack([value[1] for value in self._cached_batch(texts, prefix_ids, include_base=True)])

    def get_prefixes(self, texts: Sequence[str], prefix_ids: Sequence[str] | None = None,
                     *, use_cache: bool = True) -> list[np.ndarray]:
        if prefix_ids is None:
            prefix_ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        if len(texts) != len(prefix_ids):
            raise ValueError("Prefix IDs/texts must align")
        if not use_cache or self.cache_root is None:
            return self._uncached_prefixes(texts)
        return self._cached_batch(texts, prefix_ids)

    def forward_prefixes(self, prefixes: Sequence[np.ndarray]):
        """Length-bucket suffix computation only; restore original order before any loss."""
        torch, *_ = _dependencies()
        if not prefixes:
            raise ValueError("Empty prefix batch")
        for hidden in prefixes:
            if (not isinstance(hidden, np.ndarray) or hidden.dtype != np.float32 or hidden.ndim != 2
                    or hidden.shape[1] != self.dimension or not 1 <= len(hidden) <= self.config.max_length
                    or not np.isfinite(hidden).all()):
                raise ValueError("Invalid FP32 prefix array")
        order = sorted(range(len(prefixes)), key=lambda index: (-len(prefixes[index]), index))
        inverse = [0] * len(order)
        for position, original_index in enumerate(order):
            inverse[original_index] = position
        vectors, padded_tokens = [], 0
        size = self.config.forward_microbatch_size
        for start in range(0, len(order), size):
            group = [prefixes[index] for index in order[start:start + size]]
            vectors.append(self._forward_prefix_group(group))
            padded_tokens += len(group) * max(len(value) for value in group)
        self.last_forward_stats = {
            "items": len(prefixes), "microbatches": len(vectors), "microbatch_size": size,
            "actual_tokens": sum(len(value) for value in prefixes), "padded_tokens": padded_tokens,
            "full_batch_padded_tokens": len(prefixes) * max(len(value) for value in prefixes),
        }
        return torch.cat(vectors, dim=0)[torch.tensor(inverse, dtype=torch.long)]

    def _forward_prefix_group(self, prefixes: Sequence[np.ndarray]):
        torch, *_ = _dependencies()
        length = max(len(value) for value in prefixes)
        hidden = torch.zeros((len(prefixes), length, self.dimension), dtype=torch.float32)
        mask = torch.zeros((len(prefixes), length), dtype=torch.long)
        for index, values in enumerate(prefixes):
            hidden[index, :len(values)] = torch.from_numpy(values)
            mask[index, :len(values)] = 1
        return self.forward_cached(hidden, mask)

    def forward_cached(self, hidden_states, attention_mask):
        """Train from a padded FP32 prefix tensor/array plus a binary padding mask."""
        torch, *_ = _dependencies()
        hidden = torch.as_tensor(hidden_states, dtype=torch.float32).detach()
        mask = torch.as_tensor(attention_mask)
        if (hidden.ndim != 3 or hidden.shape[-1] != self.dimension or mask.shape != hidden.shape[:2]
                or not 1 <= hidden.shape[1] <= self.config.max_length or not torch.isfinite(hidden).all()
                or not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum(dim=1) > 0)):
            raise ValueError("Invalid cached prefix tensor/mask")
        self.model.eval()
        extended = self.model.get_extended_attention_mask(mask, mask.shape)
        for layer in self.suffix:
            hidden = layer(hidden, attention_mask=extended)[0]
        pooling_mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * pooling_mask).sum(dim=1) / pooling_mask.sum(dim=1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def forward(self, texts: Sequence[str], prefix_ids: Sequence[str] | None = None, *, use_cache: bool = True):
        return self.forward_prefixes(self.get_prefixes(texts, prefix_ids, use_cache=use_cache))

    def encode(self, texts: Sequence[str], *, prefix_ids: Sequence[str] | None = None,
               batch_size: int = 8, use_cache: bool = True) -> np.ndarray:
        torch, *_ = _dependencies()
        if batch_size < 1 or (prefix_ids is not None and len(prefix_ids) != len(texts)):
            raise ValueError("Invalid encode batch size or IDs")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with torch.no_grad():
            batches = [self.forward(texts[start:start + batch_size],
                                   None if prefix_ids is None else prefix_ids[start:start + batch_size],
                                   use_cache=use_cache).cpu().numpy()
                       for start in range(0, len(texts), batch_size)]
        return np.concatenate(batches)

    def verify_cached_equivalence(self, texts: Sequence[str], prefix_ids: Sequence[str],
                                  *, atol: float = 1e-6) -> dict:
        torch, *_ = _dependencies()
        if self.cache_root is None or not math.isfinite(atol) or atol < 0:
            raise ValueError("Equivalence check requires a cache and finite nonnegative tolerance")
        with torch.no_grad():
            cached = self.forward(texts, prefix_ids, use_cache=True)
            uncached = self.forward(texts, prefix_ids, use_cache=False)
        difference = float((cached - uncached).abs().max())
        if not math.isfinite(difference) or difference > atol:
            raise ValueError(f"Cached/uncached suffix forward mismatch: {difference} > {atol}")
        return {"ok": True, "strict_equal": torch.equal(cached, uncached), "max_abs_difference": difference,
                "absolute_tolerance": atol, "relative_tolerance": 0.0, "items": len(texts)}

    def save_suffix(self, directory: Path) -> dict:
        """Publish inference weights; generation filenames keep the previous manifest valid on interruption."""
        _, load_file, save_file, *_ = _dependencies()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        content_digest = _tensor_digest(self.parameters)
        filename = f"suffix-{content_digest[:20]}.safetensors"
        path = directory / filename
        if not path.exists():
            temporary = directory / f".{filename}.tmp"
            save_file({name: value.detach().cpu().contiguous() for name, value in self.parameters.items()}, str(temporary))
            os.replace(temporary, path)
        elif _tensor_digest(load_file(str(path), device="cpu")) != content_digest:
            raise ValueError("Existing suffix generation is corrupt")
        metadata = {"identity": self.identity, "config": asdict(self.config), "filename": filename,
                    "suffix_sha256": _sha(path), "parameter_sha256": content_digest,
                    "initial_parameter_sha256": self.initial_suffix_sha256,
                    "trainable_parameter_count": self.trainable_parameter_count,
                    "trainable_parameter_names": list(self.parameters)}
        _json(directory / "metadata.json", metadata)
        return metadata

    def load_suffix(self, directory: Path) -> dict:
        torch, load_file, *_ = _dependencies()
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["identity"] != self.identity:
            raise ValueError("Suffix/base identity mismatch")
        path = _child_file(directory, metadata["filename"])
        if _sha(path) != metadata["suffix_sha256"]:
            raise ValueError("Suffix checksum mismatch")
        weights = load_file(str(path), device="cpu")
        _validate_weights(self.parameters, weights)
        if _tensor_digest(weights) != metadata["parameter_sha256"]:
            raise ValueError("Suffix parameter content hash mismatch")
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(weights[name])
        return metadata

    # A suffix adapter contains full block weights, not LoRA low-rank matrices.
    save_adapter = save_suffix
    load_adapter = load_suffix


def _child_file(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("Unsafe checkpoint filename")
    return directory / filename


def _validate_weights(parameters: dict, weights: dict) -> None:
    torch, *_ = _dependencies()
    if set(parameters) != set(weights):
        raise ValueError("Suffix parameter names mismatch")
    for name, parameter in parameters.items():
        if weights[name].shape != parameter.shape or weights[name].dtype != parameter.dtype or not torch.isfinite(weights[name]).all():
            raise ValueError(f"Invalid suffix parameter: {name}")


def _candidate_masks(positive_mask, primary_positive_indices, candidate_ids, candidate_texts, batch, count):
    """Collapse ID/text aliases; keep one target and exclude other known positives."""
    torch, *_ = _dependencies()
    positive = torch.as_tensor(positive_mask)
    if positive.dtype != torch.bool or positive.shape != (batch, count):
        raise ValueError("positive_mask must be a boolean [queries, candidates] array")
    primary = torch.arange(batch) if primary_positive_indices is None else torch.as_tensor(primary_positive_indices)
    if primary.dtype not in (torch.int32, torch.int64) or primary.shape != (batch,):
        raise ValueError("primary_positive_indices must contain one integer per query")
    if torch.any(primary < 0) or torch.any(primary >= count) or not positive[torch.arange(batch), primary].all():
        raise ValueError("Every primary target must be marked as a known positive")
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for aliases in (candidate_ids, candidate_texts):
        if aliases is None:
            continue
        if len(aliases) != count:
            raise ValueError("Candidate IDs/texts must align with candidate vectors")
        seen = {}
        for index, value in enumerate(aliases):
            key = str(value)
            if key in seen:
                left, right = find(index), find(seen[key])
                parent[max(left, right)] = min(left, right)
            else:
                seen[key] = index
    canonical = [find(index) for index in range(count)]
    representatives = sorted(set(canonical))
    valid = torch.zeros_like(positive)
    targets = torch.tensor([canonical[index] for index in primary.tolist()], dtype=torch.long)
    for representative in representatives:
        aliases = [index for index, root in enumerate(canonical) if root == representative]
        valid[:, representative] = ~positive[:, aliases].any(dim=1)
    valid[torch.arange(batch), targets] = True
    if torch.any(valid.sum(dim=1) < 2):
        raise ValueError("Every query must retain at least one distinct, not-known-positive negative")
    return valid, targets, representatives


def distillation_loss(query_vectors, candidate_vectors, positive_mask, *, config: DistillConfig | None = None,
                      primary_positive_indices=None, candidate_ids=None, candidate_texts=None,
                      teacher_query_vectors=None, teacher_candidate_vectors=None,
                      base_query_vectors=None, base_candidate_vectors=None) -> dict:
    """CE + independently weighted cosine-listwise KL and frozen-base anchoring.

    Candidates normally comprise B designated positives followed by B hard
    negatives. positive_mask marks ALL known positives in this candidate set.
    Only the designated primary target remains in each CE/KD denominator; aliases
    and additional known positives never become negatives. KL is teacher||student
    at kd_temperature, without T². Anchor is half the mean query cosine distance
    plus half the mean unique-candidate cosine distance, using detached targets.
    When supplied, teacher vectors also produce entropy (nats) and target-mass
    diagnostics at kd_temperature even if kd_weight is zero. No teacher means
    these diagnostic fields are None; CE-only training never requires a teacher.
    """
    torch, *_ = _dependencies()
    config = config or DistillConfig()
    config.validate()
    query, documents = torch.as_tensor(query_vectors), torch.as_tensor(candidate_vectors)
    if (query.ndim != 2 or documents.ndim != 2 or query.shape[1] != documents.shape[1]
            or query.shape[0] < 1 or documents.shape[0] < 2 or not query.is_floating_point()
            or not documents.is_floating_point() or not torch.isfinite(query).all() or not torch.isfinite(documents).all()):
        raise ValueError("Student vectors must be finite 2D floating-point tensors with matching dimensions")
    batch, count = len(query), len(documents)
    valid, targets, unique = _candidate_masks(positive_mask, primary_positive_indices, candidate_ids,
                                              candidate_texts, batch, count)
    query = torch.nn.functional.normalize(query, p=2, dim=1)
    documents = torch.nn.functional.normalize(documents, p=2, dim=1)
    scores = query @ documents.T
    negative_limit = torch.finfo(scores.dtype).min
    ce = torch.nn.functional.cross_entropy((scores / config.ce_temperature).masked_fill(~valid, negative_limit), targets)
    kd, anchor = scores.new_zeros(()), scores.new_zeros(())
    diagnostics = {"teacher_distribution_entropy": None, "student_distribution_entropy": None,
                   "teacher_primary_positive_probability": None}
    has_teacher = teacher_query_vectors is not None or teacher_candidate_vectors is not None
    if config.kd_weight or has_teacher:
        tq = torch.as_tensor(teacher_query_vectors, dtype=query.dtype).detach() if teacher_query_vectors is not None else None
        td = torch.as_tensor(teacher_candidate_vectors, dtype=query.dtype).detach() if teacher_candidate_vectors is not None else None
        if (tq is None or td is None or tq.ndim != 2 or td.ndim != 2 or len(tq) != batch or len(td) != count
                or tq.shape[1] != td.shape[1] or not torch.isfinite(tq).all() or not torch.isfinite(td).all()):
            raise ValueError("KD requires aligned finite teacher query/document vectors")
        teacher_scores = torch.nn.functional.normalize(tq, p=2, dim=1) @ torch.nn.functional.normalize(td, p=2, dim=1).T
        teacher_log = torch.nn.functional.log_softmax(
            (teacher_scores / config.kd_temperature).masked_fill(~valid, negative_limit), dim=1)
        student_log = torch.nn.functional.log_softmax(
            (scores / config.kd_temperature).masked_fill(~valid, negative_limit), dim=1)
        teacher_prob = teacher_log.exp()
        if config.kd_weight:
            kd = (teacher_prob * (teacher_log - student_log)).sum(dim=1).mean()
        diagnostics = {
            "teacher_distribution_entropy": float(-(teacher_prob * teacher_log).sum(dim=1).mean().detach()),
            "student_distribution_entropy": float(-(student_log.exp() * student_log).sum(dim=1).mean().detach()),
            "teacher_primary_positive_probability": float(teacher_prob[torch.arange(batch), targets].mean()),
        }
    if config.anchor_weight:
        bq = torch.as_tensor(base_query_vectors, dtype=query.dtype).detach() if base_query_vectors is not None else None
        bd = torch.as_tensor(base_candidate_vectors, dtype=query.dtype).detach() if base_candidate_vectors is not None else None
        if (bq is None or bd is None or bq.shape != query.shape or bd.shape != documents.shape
                or not torch.isfinite(bq).all() or not torch.isfinite(bd).all()):
            raise ValueError("Anchoring requires aligned finite frozen-base vectors in the student dimension")
        bq, bd = torch.nn.functional.normalize(bq, p=2, dim=1), torch.nn.functional.normalize(bd, p=2, dim=1)
        anchor = 0.5 * ((1 - (query * bq).sum(dim=1)).mean()
                        + (1 - (documents[unique] * bd[unique]).sum(dim=1)).mean())
    loss = ce + config.kd_weight * kd + config.anchor_weight * anchor
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite distillation loss")
    return {"loss": loss, "ce": ce, "kd": kd, "anchor": anchor, **diagnostics,
            "valid_candidate_mask": valid, "target_indices": targets,
            "unique_candidate_count": len(unique), "mean_valid_candidates": float(valid.sum(dim=1).float().mean())}


class DistillTrainer:
    """One optimizer and explicit row order; validation/epoch orchestration is external."""

    def __init__(self, student: DistillStudent, plan_identity: dict, row_order: Sequence[str]):
        torch, *_ = _dependencies()
        self.student, self.config = student, student.config
        self.plan_identity = json.loads(json.dumps(plan_identity, allow_nan=False))
        self.row_order = list(row_order)
        if not self.row_order or len(self.row_order) != len(set(self.row_order)):
            raise ValueError("row_order must be a nonempty sequence of unique row IDs")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.optimizer = torch.optim.AdamW(list(student.parameters.values()), lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        self.progress = {"completed_steps": 0, "nonzero_gradient_steps": 0, "epoch": 0, "row_cursor": 0}
        self.identity = {"student": student.identity, "config": asdict(self.config),
                         "plan_sha256": _digest(self.plan_identity),
                         "initial_suffix_sha256": student.initial_suffix_sha256,
                         "objective": "CE + kd_weight * KL(teacher||student) + anchor_weight * cosine_anchor; no T^2"}

    def set_epoch_order(self, row_order: Sequence[str], epoch: int) -> None:
        if set(row_order) != set(self.row_order) or len(row_order) != len(self.row_order) or epoch < 0:
            raise ValueError("Epoch order must be a permutation of the same rows and epoch must be nonnegative")
        self.row_order = list(row_order)
        self.progress.update(epoch=epoch, row_cursor=0)

    def step(self, query_texts: Sequence[str], candidate_texts: Sequence[str], positive_mask, *,
             query_ids: Sequence[str] | None = None, candidate_ids: Sequence[str] | None = None,
             next_row_cursor: int | None = None, **loss_targets) -> dict:
        torch, *_ = _dependencies()
        if len(candidate_texts) != 2 * len(query_texts):
            raise ValueError("A training batch must contain B positives followed by B hard negatives")
        if next_row_cursor is not None and not 0 <= next_row_cursor <= len(self.row_order):
            raise ValueError("next_row_cursor is outside the epoch order")
        self.student.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        query = self.student.forward(query_texts, query_ids)
        query_forward_stats = dict(self.student.last_forward_stats)
        documents = self.student.forward(candidate_texts, candidate_ids)
        candidate_forward_stats = dict(self.student.last_forward_stats)
        losses = distillation_loss(query, documents, positive_mask, config=self.config,
                                   candidate_ids=candidate_ids, candidate_texts=candidate_texts, **loss_targets)
        losses["loss"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(list(self.student.parameters.values()), self.config.max_grad_norm))
        if not math.isfinite(grad_norm):
            raise ValueError("Nonfinite suffix gradient")
        self.optimizer.step()
        self.progress["completed_steps"] += 1
        self.progress["nonzero_gradient_steps"] += int(grad_norm > 0)
        if next_row_cursor is not None:
            self.progress["row_cursor"] = next_row_cursor
        record = {"step": self.progress["completed_steps"], "epoch": self.progress["epoch"],
                  **{name: float(losses[name].detach()) for name in ("loss", "ce", "kd", "anchor")},
                  "gradient_l2_before_clip": grad_norm, "trainable_parameter_count": self.student.trainable_parameter_count,
                  "unique_candidate_count": losses["unique_candidate_count"],
                  "mean_valid_candidates": losses["mean_valid_candidates"], "dropout_enabled": False,
                  "query_suffix_forward": query_forward_stats, "candidate_suffix_forward": candidate_forward_stats,
                  **{name: losses[name] for name in ("teacher_distribution_entropy", "student_distribution_entropy",
                                                    "teacher_primary_positive_probability")}}
        LOGGER.info("Suffix step %s: loss %.6f CE %.6f KD %.6f anchor %.6f grad %.6f",
                    record["step"], record["loss"], record["ce"], record["kd"], record["anchor"], grad_norm)
        return record

    def save_checkpoint(self, directory: Path) -> dict:
        torch, load_file, save_file, *_ = _dependencies()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {f"suffix.{name}": value.detach().cpu().contiguous() for name, value in self.student.parameters.items()}
        for name, parameter in self.student.parameters.items():
            for key, value in self.optimizer.state[parameter].items():
                tensors[f"optimizer.{name}.{key}"] = value.detach().cpu().contiguous()
        tensors["rng.torch"] = torch.get_rng_state()
        np_state = np.random.get_state()
        filename = f"state-{self.progress['completed_steps']:09d}-{_tensor_digest(tensors)[:16]}.safetensors"
        path = directory / filename
        if not path.exists():
            temporary = directory / f".{filename}.tmp"
            save_file(tensors, str(temporary))
            os.replace(temporary, path)
        elif _tensor_digest(load_file(str(path), device="cpu")) != _tensor_digest(tensors):
            raise ValueError("Existing checkpoint generation is corrupt")
        metadata = {"schema": 1, "identity": self.identity, "plan_identity": self.plan_identity,
                    "progress": self.progress, "row_order": self.row_order, "row_order_sha256": _digest(self.row_order),
                    "filename": filename, "state_sha256": _sha(path),
                    "optimizer_group": {key: value for key, value in self.optimizer.param_groups[0].items() if key != "params"},
                    "rng_python": random.getstate(),
                    "rng_numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
                    "trainable_parameter_count": self.student.trainable_parameter_count,
                    "training_mode": "eval with gradients; dropout disabled"}
        _json(directory / "checkpoint.json", metadata)
        return metadata

    def load_checkpoint(self, directory: Path) -> dict:
        torch, load_file, *_ = _dependencies()
        directory = Path(directory)
        metadata = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        if metadata["identity"] != self.identity or _digest(metadata["plan_identity"]) != self.identity["plan_sha256"]:
            raise ValueError("Checkpoint plan/config/base identity mismatch")
        order = metadata["row_order"]
        if (len(order) != len(self.row_order) or set(order) != set(self.row_order)
                or metadata["row_order_sha256"] != _digest(order)):
            raise ValueError("Checkpoint row order identity mismatch")
        progress = metadata["progress"]
        if (any(not isinstance(progress[key], int) or progress[key] < 0 for key in
                ("completed_steps", "nonzero_gradient_steps", "epoch", "row_cursor"))
                or progress["row_cursor"] > len(order) or progress["nonzero_gradient_steps"] > progress["completed_steps"]):
            raise ValueError("Invalid checkpoint progress")
        path = _child_file(directory, metadata["filename"])
        if _sha(path) != metadata["state_sha256"]:
            raise ValueError("Checkpoint state checksum mismatch")
        tensors = load_file(str(path), device="cpu")
        weights = {name.removeprefix("suffix."): value for name, value in tensors.items() if name.startswith("suffix.")}
        _validate_weights(self.student.parameters, weights)
        optimizer = {}
        for name, parameter in self.student.parameters.items():
            prefix = f"optimizer.{name}."
            state = {key.removeprefix(prefix): value for key, value in tensors.items() if key.startswith(prefix)}
            if progress["completed_steps"] and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Missing Adam state for {name}")
            for key, value in state.items():
                expected = torch.Size([]) if key == "step" else parameter.shape
                if (value.shape != expected or value.dtype != torch.float32 or not torch.isfinite(value).all()
                        or (key == "step" and float(value) != progress["completed_steps"])):
                    raise ValueError(f"Invalid Adam state for {name}/{key}")
            optimizer[name] = state
        group = dict(metadata["optimizer_group"])
        group["betas"] = tuple(group["betas"])
        expected_group = {key: value for key, value in self.optimizer.param_groups[0].items() if key != "params"}
        if group != expected_group:
            raise ValueError("Checkpoint AdamW settings differ from this fixed optimizer configuration")
        # Validate all RNG records before mutating the student or its optimizer.
        torch.Generator().set_state(tensors["rng.torch"])
        py_state = metadata["rng_python"]
        py_state = (py_state[0], tuple(py_state[1]), py_state[2])
        random.Random().setstate(py_state)
        np_state = metadata["rng_numpy"]
        np_state = (np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:])
        np.random.RandomState().set_state(np_state)
        with torch.no_grad():
            for name, parameter in self.student.parameters.items():
                parameter.copy_(weights[name])
                self.optimizer.state[parameter] = optimizer[name]
        self.optimizer.param_groups[0].update(group)
        self.progress, self.row_order = dict(progress), list(order)
        torch.set_rng_state(tensors["rng.torch"])
        random.setstate(py_state)
        np.random.set_state(np_state)
        self.student.model.eval()
        return metadata
