"""Real CPU MiniLM LoRA adaptation with matched negative-sampling experiments.

Optional dependencies: CPU PyTorch, transformers and safetensors. No hosted
model calls and no pickle loading. This module never accesses test judgments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .ranker import grouped_train_validation_split, normalized_claim

LOGGER = logging.getLogger(__name__)
ARMS = ("random", "retrieved", "filtered")


@dataclass(frozen=True)
class AdaptationConfig:
    seed: int = 42
    max_train_queries: int = 400
    epochs: int = 1
    batch_size: int = 2
    negatives_per_query: int = 2
    max_length: int = 128
    lora_rank: int = 8
    lora_alpha: float = 16.0
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    temperature: float = 0.05
    max_grad_norm: float = 1.0
    candidate_pool_size: int = 50
    filter_margin: float = 0.05
    threads: int = 2
    checkpoint_every: int = 50
    arms: tuple[str, ...] = ARMS

    def validate(self) -> None:
        for name in ("max_train_queries", "epochs", "batch_size", "negatives_per_query", "lora_rank", "threads", "checkpoint_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 8 <= self.max_length <= 512 or self.candidate_pool_size < self.negatives_per_query:
            raise ValueError("Invalid token length or negative candidate depth")
        if any(not math.isfinite(getattr(self, name)) for name in ("lora_alpha", "learning_rate", "weight_decay", "temperature", "max_grad_norm", "filter_margin")):
            raise ValueError("Training hyperparameters must be finite")
        if min(self.lora_alpha, self.learning_rate, self.temperature, self.max_grad_norm) <= 0 or min(self.weight_decay, self.filter_margin) < 0:
            raise ValueError("Invalid optimization/filter hyperparameters")
        if not self.arms or len(set(self.arms)) != len(self.arms) or set(self.arms) - set(ARMS):
            raise ValueError("arms must be a nonempty unique subset of random/retrieved/filtered")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _query_rng(seed: int, qid: str, salt: str) -> np.random.Generator:
    value = hashlib.sha256(f"{seed}:{qid}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8], "little"))


def _family_positives(qids: list[str], queries: dict, qrels: dict) -> dict[str, set[str]]:
    """Training-subset labels only; validation evidence cannot affect filtering."""
    parent = {qid: qid for qid in qids}

    def find(qid):
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    seen = {}
    for qid in qids:
        keys = [("claim", normalized_claim(queries[qid]))]
        keys.extend(("doc", doc) for doc, relevance in qrels[qid].items() if relevance > 0)
        for key in keys:
            if key in seen:
                a, b = find(qid), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = qid
    grouped: dict[str, set[str]] = {}
    for qid in qids:
        grouped.setdefault(find(qid), set()).update(doc for doc, relevance in qrels[qid].items() if relevance > 0)
    return {qid: grouped[find(qid)] for qid in qids}


def build_adaptation_plan(dataset, index, config: AdaptationConfig | None = None, *, teacher_scores: dict | None = None,
                          teacher_metadata: dict | None = None) -> dict:
    """Return reusable, inspectable train/validation triples before any gradient step.

    Optional teacher scores are comparable higher-is-more-relevant scalar scores.
    For every selected training query its chosen positive must be scored. Missing
    negative scores are retained and explicitly counted; no stronger-teacher
    claim is made for unscored negatives or random fallback examples.
    """
    config = config or AdaptationConfig()
    config.validate()
    qrels = dataset.qrels["train"]
    train_partition, validation_qids, split = grouped_train_validation_split(dataset.queries, qrels, seed=config.seed)
    selected = np.random.default_rng(config.seed).permutation(train_partition)[:config.max_train_queries].tolist()
    train_qids = sorted(selected)
    families = _family_positives(train_qids, dataset.queries, qrels)
    document_ids = sorted(str(doc["_id"]) for doc in dataset.docs)
    if len(document_ids) <= config.negatives_per_query:
        raise ValueError("Not enough corpus documents for the requested negatives")
    triples = {arm: [] for arm in ARMS}
    validation, mining = [], []
    stats = {"candidate_count": 0, "family_excluded": 0, "teacher_excluded": 0,
             "missing_candidate_teacher_score": 0, "random_fallback_negatives": 0,
             "queries_with_filtered_negatives": 0}
    all_qids = train_qids + validation_qids
    for number, qid in enumerate(all_qids, 1):
        positives = sorted(doc for doc, rel in qrels[qid].items() if rel > 0)
        positive = positives[int(_query_rng(config.seed, qid, "positive").integers(len(positives)))]
        hits = index.search(dataset.queries[qid], method="rrf", k=config.candidate_pool_size)
        rrf_ranks = {str(hit["doc_id"]): position for position, hit in enumerate(hits, 1)}
        hard_pool = [hit for hit in hits if str(hit["doc_id"]) not in positives]
        excluded = set(positives)
        random_pool = [doc for doc in document_ids if doc not in excluded]
        if len(random_pool) < config.negatives_per_query:
            raise ValueError(f"Query {qid} has insufficient nonpositive documents")

        def fill(chosen: list[str], *, family: set[str] | None = None, salt="fill") -> tuple[list[str], int]:
            needed = config.negatives_per_query - len(chosen)
            if needed <= 0:
                return chosen[:config.negatives_per_query], 0
            remaining = [doc for doc in random_pool if doc not in chosen and doc not in (family or set())]
            if len(remaining) < needed:
                raise ValueError(f"Query {qid}: filtering leaves too few negatives")
            fallback = _query_rng(config.seed, qid, salt).choice(remaining, size=needed, replace=False).tolist()
            return chosen + fallback, needed

        retrieved, _ = fill([str(hit["doc_id"]) for hit in hard_pool[:config.negatives_per_query]])
        entry = {"query_id": qid, "positive_doc_id": positive}
        if qid not in families:
            validation.append(dict(entry, negative_doc_ids=retrieved))
            continue
        random_negatives = _query_rng(config.seed, qid, "random").choice(random_pool, size=config.negatives_per_query, replace=False).tolist()
        score_map = teacher_scores.get(qid, {}) if teacher_scores is not None else {
            str(hit["doc_id"]): hit.get("dense_score") for hit in hits
        }
        if teacher_scores is None and (positive not in score_map or score_map[positive] is None):
            for hit in index.search(dataset.queries[qid], method="rrf", k=len(document_ids)):
                if str(hit["doc_id"]) == positive:
                    score_map[positive] = hit.get("dense_score")
                    break
        positive_score = score_map.get(positive)
        if positive_score is None or not math.isfinite(float(positive_score)):
            raise ValueError(f"Teacher score missing/nonfinite for chosen positive {qid}/{positive}")
        cutoff = float(positive_score) - config.filter_margin
        accepted, decisions = [], []
        for hit in hard_pool:
            doc_id = str(hit["doc_id"])
            score = score_map.get(doc_id)
            reason = "retained"
            stats["candidate_count"] += 1
            if doc_id in families[qid]:
                stats["family_excluded"] += 1
                reason = "related_training_claim_positive"
            elif score is not None:
                if not math.isfinite(float(score)):
                    raise ValueError(f"Nonfinite teacher score: {qid}/{doc_id}")
                if float(score) >= cutoff:
                    stats["teacher_excluded"] += 1
                    reason = "teacher_close_to_positive"
            else:
                stats["missing_candidate_teacher_score"] += 1
                reason = "retained_unscored"
            if reason in {"retained", "retained_unscored"}:
                accepted.append(doc_id)
            decisions.append({"doc_id": doc_id, "rrf_rank": rrf_ranks[doc_id],
                              "teacher_score": None if score is None else float(score), "decision": reason})
        filtered, fallback_count = fill(accepted[:config.negatives_per_query], family=families[qid], salt="filtered-fallback")
        stats["random_fallback_negatives"] += fallback_count
        stats["queries_with_filtered_negatives"] += int(filtered != retrieved)
        triples["random"].append(dict(entry, negative_doc_ids=random_negatives))
        triples["retrieved"].append(dict(entry, negative_doc_ids=retrieved))
        triples["filtered"].append(dict(entry, negative_doc_ids=filtered))
        mining.append({"query_id": qid, "positive_doc_id": positive, "positive_teacher_score": float(positive_score),
                       "filter_cutoff": cutoff, "candidates": decisions, "fallback_count": fallback_count})
        if number % 100 == 0 or number == len(train_qids):
            LOGGER.info("Adaptation negatives: %s/%s training queries", number, len(train_qids))
    teacher = teacher_metadata or ({"name": "caller-provided teacher scores", "score_scale": "caller-defined higher-is-more-relevant"}
                                   if teacher_scores is not None else {
                                       "name": "frozen retrieval encoder cosine", "model_id": index.metadata.get("model_id"),
                                       "model_revision": index.metadata.get("model_revision"), "score_scale": "cosine similarity",
                                   })
    plan = {
        "schema_version": 1, "config": asdict(config), "train_query_ids": train_qids, "validation_query_ids": validation_qids,
        "split": split, "arms": triples, "validation": validation, "mining": mining, "mining_statistics": stats,
        "teacher": teacher, "external_teacher_scores": teacher_scores is not None,
        "filter_rule": "Exclude related selected-training-claim positives and candidates with teacher score >= positive score - margin; unscored candidates are retained; shortage uses documented random fallback.",
        "validation_policy": "Fixed retrieved-negative InfoNCE monitoring only; final epoch is selected in advance. No test access or validation-driven early stopping.",
        "budget_note": "Only the deterministically selected training queries produce gradient updates; held-out official-train claims monitor loss. No full-train refit.",
        "corpus_sha256": getattr(dataset, "corpus_hash", None) or _digest(dataset.docs),
        "selected_train_qrels_sha256": _digest({qid: qrels[qid] for qid in train_qids}),
        "selected_train_queries_sha256": _digest({qid: dataset.queries[qid] for qid in train_qids}),
        "retrieval_metadata": index.metadata,
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def _dependencies():
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from tokenizers import Tokenizer
        from transformers import BertConfig, BertModel
    except ImportError as exc:
        raise RuntimeError("Neural adaptation requires CPU torch, transformers and safetensors; install the neural extra") from exc
    return torch, load_file, save_file, Tokenizer, BertConfig, BertModel


def _attach_lora(model, rank: int, alpha: float):
    torch, *_ = _dependencies()

    class LoRALinear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_A = torch.nn.Parameter(torch.empty(rank, base.in_features))
            self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scaling = alpha / rank

        def forward(self, inputs):
            return self.base(inputs) + self.scaling * torch.nn.functional.linear(
                torch.nn.functional.linear(inputs, self.lora_A), self.lora_B)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attention = model.encoder.layer[-1].attention.self
    attention.query = LoRALinear(attention.query)
    attention.value = LoRALinear(attention.value)


class AdaptedEncoder:
    """Official transformers.BertModel, optionally with a validated LoRA adapter."""
    def __init__(self, base_model_dir: Path, config: AdaptationConfig | None = None, *, trainable: bool = False):
        torch, load_file, _, Tokenizer, BertConfig, BertModel = _dependencies()
        self.config = config or AdaptationConfig()
        self.config.validate()
        self.base_model_dir = Path(base_model_dir)
        torch.set_num_threads(self.config.threads)
        torch.manual_seed(self.config.seed)
        torch.use_deterministic_algorithms(True)
        model_config = BertConfig.from_json_file(str(self.base_model_dir / "config.json"))
        model_config._attn_implementation = "eager"
        if self.config.max_length > model_config.max_position_embeddings:
            raise ValueError("Requested max_length exceeds the pretrained positional embeddings")
        self.model = BertModel(model_config, add_pooling_layer=False)
        state = load_file(str(self.base_model_dir / "model.safetensors"), device="cpu")
        state = {name: value for name, value in state.items() if not name.startswith("pooler.")}
        # Older official checkpoints persisted buffers that modern BertModel
        # deterministically regenerates. Verify their values before dropping them.
        regenerated_buffers = []
        for name in ("embeddings.position_ids", "embeddings.token_type_ids"):
            if name in state:
                expected = torch.arange(model_config.max_position_embeddings).unsqueeze(0)
                if name.endswith("token_type_ids"):
                    expected = torch.zeros_like(expected)
                if not torch.equal(state[name], expected):
                    raise ValueError(f"Unexpected persisted BERT buffer values: {name}")
                del state[name]
                regenerated_buffers.append(name)
        self.model.load_state_dict(state, strict=True)
        del state
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if trainable:
            _attach_lora(self.model, self.config.lora_rank, self.config.lora_alpha)
        self.tokenizer = Tokenizer.from_file(str(self.base_model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.config.max_length)
        self.tokenizer.enable_padding(pad_id=model_config.pad_token_id or 0, pad_token="[PAD]")
        base_sha256 = _file_digest(self.base_model_dir / "model.safetensors")
        self.metadata = {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2", "model_revision": "local-float-weights",
            "base_model_sha256": base_sha256, "model_sha256": base_sha256,
            "config_sha256": _file_digest(self.base_model_dir / "config.json"),
            "tokenizer_sha256": _file_digest(self.base_model_dir / "tokenizer.json"),
            "dimension": model_config.hidden_size, "max_length": self.config.max_length,
            "pooling": "attention-mask-mean", "normalize": True, "backend": "pytorch-cpu",
            "threads": self.config.threads, "torch_version": torch.__version__, "adapter": trainable,
            "verified_regenerated_buffers": regenerated_buffers,
        }
        self.model.eval()

    def _forward(self, texts: list[str]):
        torch, *_ = _dependencies()
        encoded = self.tokenizer.encode_batch(texts)
        inputs = {"input_ids": torch.tensor([item.ids for item in encoded], dtype=torch.long),
                  "attention_mask": torch.tensor([item.attention_mask for item in encoded], dtype=torch.long),
                  "token_type_ids": torch.tensor([item.type_ids for item in encoded], dtype=torch.long)}
        tokens = self.model(**inputs, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        torch, *_ = _dependencies()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, self.metadata["dimension"]), dtype=np.float32)
        self.model.eval()
        output = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                vectors = self._forward(texts[start:start + batch_size])
                if not torch.isfinite(vectors).all():
                    raise RuntimeError("Nonfinite neural embeddings")
                output.append(vectors.cpu().numpy())
                if len(texts) > 100 and (start // batch_size) % 100 == 0:
                    LOGGER.info("Float encoder documents: %s/%s", min(start + batch_size, len(texts)), len(texts))
        return np.concatenate(output).astype(np.float32, copy=False)

    @classmethod
    def load(cls, base_model_dir: Path, adapter_dir: Path, *, threads: int = 2) -> "AdaptedEncoder":
        _, load_file, *_ = _dependencies()
        adapter_dir = Path(adapter_dir)
        metadata = json.loads((adapter_dir / "metadata.json").read_text(encoding="utf-8"))
        config_values = dict(metadata["config"], threads=threads)
        config_values["arms"] = tuple(config_values["arms"])
        encoder = cls(base_model_dir, AdaptationConfig(**config_values), trainable=True)
        expected = metadata["base_encoder"]
        for field in ("base_model_sha256", "config_sha256", "tokenizer_sha256", "dimension", "max_length"):
            if encoder.metadata[field] != expected[field]:
                raise ValueError(f"Adapter/base model mismatch: {field}")
        adapter_path = adapter_dir / "adapter.safetensors"
        if _file_digest(adapter_path) != metadata["adapter_sha256"]:
            raise ValueError("Adapter checksum mismatch")
        _load_parameters(encoder, load_file(str(adapter_path), device="cpu"))
        encoder.metadata.update({"adapter_sha256": metadata["adapter_sha256"], "adaptation_arm": metadata["arm"],
                                 "model_sha256": _digest({"base": encoder.metadata["base_model_sha256"], "adapter": metadata["adapter_sha256"]})})
        encoder.model.eval()
        return encoder


class MultiAdapterEncoder:
    """Stream shared frozen-prefix inference across baseline and LoRA variants.

    Keep one instance across corpus chunks to avoid reloading the base weights.
    Each batch runs the complete baseline once, then reuses its penultimate
    hidden states for each adapter's final layer. No corpus-sized prefix cache.
    """
    def __init__(self, base_model_dir: Path, adapter_dirs: dict[str, Path], config: AdaptationConfig | None = None,
                 *, max_length: int = 128, threads: int = 2):
        _, load_file, *_ = _dependencies()
        if "base" in adapter_dirs:
            raise ValueError("'base' is reserved for the unadapted float model")
        records = {name: json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8")) for name, path in adapter_dirs.items()}
        if config is not None:
            config.validate()
        elif records:
            values = dict(next(iter(records.values()))["config"], threads=threads, max_length=max_length)
            values["arms"] = tuple(values["arms"])
            config = AdaptationConfig(**values)
        else:
            config = AdaptationConfig(threads=threads, max_length=max_length)
        self.encoder = AdaptedEncoder(base_model_dir, config, trainable=bool(records))
        self.baseline_parameters = {name: value.detach().clone() for name, value in _parameters(self.encoder).items()}
        self.adapters, self.metadata = {}, {"base": dict(self.encoder.metadata, adapter=False)}
        for name, record in records.items():
            for field in ("lora_rank", "lora_alpha", "max_length"):
                if record["config"][field] != getattr(config, field):
                    raise ValueError(f"Shared-prefix adapters use incompatible {field}")
            for field in ("base_model_sha256", "config_sha256", "tokenizer_sha256", "dimension", "max_length"):
                if self.encoder.metadata[field] != record["base_encoder"][field]:
                    raise ValueError(f"Adapter/base model mismatch: {field}")
            path = Path(adapter_dirs[name]) / "adapter.safetensors"
            if _file_digest(path) != record["adapter_sha256"]:
                raise ValueError("Adapter checksum mismatch")
            self.adapters[name] = load_file(str(path), device="cpu")
            _load_parameters(self.encoder, self.adapters[name])
            self.metadata[name] = dict(self.encoder.metadata, adapter_sha256=record["adapter_sha256"], adaptation_arm=record["arm"],
                                       model_sha256=_digest({"base": self.encoder.metadata["base_model_sha256"], "adapter": record["adapter_sha256"]}))
        if self.baseline_parameters:
            _load_parameters(self.encoder, self.baseline_parameters)
        self.encoder.model.eval()

    def encode(self, texts: list[str], batch_size: int = 8) -> dict[str, np.ndarray]:
        torch, *_ = _dependencies()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        names = ["base", *self.adapters]
        if not texts:
            return {name: np.empty((0, self.encoder.metadata["dimension"]), dtype=np.float32) for name in names}
        collected = {name: [] for name in names}
        model = self.encoder.model
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                if self.baseline_parameters:
                    _load_parameters(self.encoder, self.baseline_parameters)
                encoded = self.encoder.tokenizer.encode_batch(texts[start:start + batch_size])
                inputs = {"input_ids": torch.tensor([item.ids for item in encoded], dtype=torch.long),
                          "attention_mask": torch.tensor([item.attention_mask for item in encoded], dtype=torch.long),
                          "token_type_ids": torch.tensor([item.type_ids for item in encoded], dtype=torch.long)}
                output = model(**inputs, output_hidden_states=True, return_dict=True)
                prefix = output.hidden_states[-2]
                pooling_mask = inputs["attention_mask"].unsqueeze(-1).to(prefix.dtype)

                def pool(tokens):
                    vectors = (tokens * pooling_mask).sum(1) / pooling_mask.sum(1).clamp_min(1)
                    normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)
                    if not torch.isfinite(normalized).all():
                        raise RuntimeError("Nonfinite shared-prefix embeddings")
                    return normalized.cpu().numpy()

                collected["base"].append(pool(output.last_hidden_state))
                extended_mask = model.get_extended_attention_mask(inputs["attention_mask"], inputs["input_ids"].shape)
                for name, adapter in self.adapters.items():
                    _load_parameters(self.encoder, adapter)
                    last_hidden = model.encoder.layer[-1](prefix, attention_mask=extended_mask)[0]
                    collected[name].append(pool(last_hidden))
        return {name: np.concatenate(batches).astype(np.float32, copy=False) for name, batches in collected.items()}


def encode_variants(base_model_dir: Path, adapter_dirs: dict[str, Path], texts: list[str], *, batch_size: int = 8,
                    max_length: int = 128, threads: int = 2) -> dict[str, np.ndarray]:
    """One-shot shared-prefix encoding; use MultiAdapterEncoder for repeated chunks."""
    return MultiAdapterEncoder(base_model_dir, adapter_dirs, max_length=max_length, threads=threads).encode(texts, batch_size=batch_size)


def _parameters(encoder: AdaptedEncoder) -> dict:
    return {name: parameter for name, parameter in encoder.model.named_parameters() if parameter.requires_grad}


def _parameter_digest(parameters: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_parameters(encoder: AdaptedEncoder, tensors: dict) -> None:
    torch, *_ = _dependencies()
    parameters = _parameters(encoder)
    if set(tensors) != set(parameters):
        raise ValueError("Adapter trainable-parameter schema mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != tensors[name].shape or not torch.isfinite(tensors[name]).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
            parameter.copy_(tensors[name])


def _contrastive_loss(encoder, examples, queries, documents, temperature):
    torch, *_ = _dependencies()
    query_vectors = encoder._forward([queries[row["query_id"]] for row in examples])
    ids = [[row["positive_doc_id"], *row["negative_doc_ids"]] for row in examples]
    document_vectors = encoder._forward([documents[doc_id] for group in ids for doc_id in group])
    document_vectors = document_vectors.reshape(len(examples), len(ids[0]), -1)
    logits = torch.einsum("bd,bkd->bk", query_vectors, document_vectors) / temperature
    return torch.nn.functional.cross_entropy(logits, torch.zeros(len(examples), dtype=torch.long))


def _validation_loss(encoder, examples, queries, documents, config):
    torch, *_ = _dependencies()
    encoder.model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start:start + config.batch_size]
            loss = _contrastive_loss(encoder, batch, queries, documents, config.temperature)
            total += float(loss) * len(batch)
            count += len(batch)
    return total / max(1, count)


def _save_checkpoint(arm_dir, encoder, optimizer, progress, identity):
    _, _, save_file, *_ = _dependencies()
    checkpoint_dir = arm_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step-{progress['completed_steps']:06d}"
    tensor_path, json_path = checkpoint_dir / f"{stem}.safetensors", checkpoint_dir / f"{stem}.json"
    payload = {}
    for name, parameter in _parameters(encoder).items():
        payload[f"adapter.{name}"] = parameter.detach().cpu().contiguous()
        for field, value in optimizer.state[parameter].items():
            if hasattr(value, "detach"):
                payload[f"optimizer.{name}.{field}"] = value.detach().cpu().contiguous()
            else:
                raise ValueError(f"Unexpected nontensor Adam state: {field}")
    temp = tensor_path.with_suffix(".tmp")
    save_file(payload, str(temp))
    os.replace(temp, tensor_path)
    _write_json(json_path, {"identity": identity, "progress": progress, "tensor_sha256": _file_digest(tensor_path)})
    _write_json(arm_dir / "latest.json", {"checkpoint": stem})


def _resume_checkpoint(arm_dir, encoder, optimizer, identity):
    _, load_file, *_ = _dependencies()
    latest_path = arm_dir / "latest.json"
    if not latest_path.exists():
        return None
    stem = json.loads(latest_path.read_text())["checkpoint"]
    if not stem.startswith("step-") or not stem[5:].isdigit():
        raise ValueError("Invalid checkpoint pointer")
    tensor_path = arm_dir / "checkpoints" / f"{stem}.safetensors"
    metadata = json.loads((arm_dir / "checkpoints" / f"{stem}.json").read_text())
    if metadata["identity"] != identity or metadata["tensor_sha256"] != _file_digest(tensor_path):
        raise ValueError("Checkpoint config/data/base identity or checksum mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    _load_parameters(encoder, {name.removeprefix("adapter."): tensor for name, tensor in tensors.items() if name.startswith("adapter.")})
    for name, parameter in _parameters(encoder).items():
        prefix = f"optimizer.{name}."
        optimizer.state[parameter] = {key.removeprefix(prefix): value for key, value in tensors.items() if key.startswith(prefix)}
    return metadata["progress"]


def train_adaptation(dataset, index, base_model_dir: Path, output_dir: Path, config: AdaptationConfig | None = None,
                     *, teacher_scores: dict | None = None, teacher_metadata: dict | None = None, resume: bool = True,
                     progress_callback: Callable[[str, int], None] | None = None) -> dict:
    """Train actual final-layer Q/V LoRA on CPU with equal budgets across arms.

    Every step restores a deterministic dropout seed. Safetensors checkpoints
    contain adapters and Adam tensor states, so interrupted runs resume exactly
    at batch boundaries. Changed configuration/data/teacher/base weights reject
    resume; extending epochs requires a new explicitly planned run directory.
    """
    torch, _, save_file, *_ = _dependencies()
    config = config or AdaptationConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan = build_adaptation_plan(dataset, index, config, teacher_scores=teacher_scores, teacher_metadata=teacher_metadata)
    plan_path = output_dir / "training_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8"))["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Output directory contains a different training plan; use a new directory")
    _write_json(plan_path, plan)
    documents = {str(doc["_id"]): f"{doc.get('title', '')}\n{doc['text']}" for doc in dataset.docs}
    results, baseline_validation_loss = {}, None
    for arm in config.arms:
        arm_started = time.perf_counter()
        arm_dir = output_dir / arm
        arm_dir.mkdir(exist_ok=True)
        encoder = AdaptedEncoder(base_model_dir, config, trainable=True)
        parameters = _parameters(encoder)
        initial_parameters = {name: value.detach().clone() for name, value in parameters.items()}
        initial_sha256 = _parameter_digest(parameters)
        identity = {"plan_sha256": plan["plan_sha256"], "config_sha256": _digest(asdict(config)), "arm": arm,
                    "base_encoder": encoder.metadata, "initial_adapter_sha256": initial_sha256}
        if (arm_dir / "metadata.json").exists() and resume:
            saved = json.loads((arm_dir / "metadata.json").read_text(encoding="utf-8"))
            if saved["identity"] != identity or saved["adapter_sha256"] != _file_digest(arm_dir / "adapter.safetensors"):
                raise ValueError("Completed adapter identity/checksum mismatch")
            results[arm] = saved
            LOGGER.info("Adaptation arm %s already complete; verified and reused", arm)
            del encoder
            continue
        if not resume and (arm_dir / "latest.json").exists():
            raise FileExistsError("Refusing to overwrite a training checkpoint; use a new output directory")
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=config.learning_rate, weight_decay=config.weight_decay)
        progress = {"completed_steps": 0, "trace": [], "nonzero_gradient_steps": 0}
        if resume:
            progress = _resume_checkpoint(arm_dir, encoder, optimizer, identity) or progress
        if baseline_validation_loss is None:
            # The baseline is measured from the original initialization, even on resume.
            restored = {name: value.detach().clone() for name, value in parameters.items()}
            _load_parameters(encoder, initial_parameters)
            baseline_validation_loss = _validation_loss(encoder, plan["validation"], dataset.queries, documents, config)
            _load_parameters(encoder, restored)
        examples = plan["arms"][arm]
        batches_per_epoch = math.ceil(len(examples) / config.batch_size)
        total_steps = batches_per_epoch * config.epochs
        LOGGER.info("LoRA arm %s: %s train queries, %s steps, %s trainable parameters", arm, len(examples), total_steps,
                    sum(parameter.numel() for parameter in parameters.values()))
        for step in range(progress["completed_steps"], total_steps):
            epoch, batch_index = divmod(step, batches_per_epoch)
            order = np.random.default_rng(config.seed + epoch).permutation(len(examples))
            selected = order[batch_index * config.batch_size:(batch_index + 1) * config.batch_size]
            batch = [examples[int(position)] for position in selected]
            torch.manual_seed(config.seed + epoch * 1_000_000 + batch_index)
            # Frozen prefix stays deterministic; only the final block uses training dropout.
            encoder.model.eval()
            encoder.model.encoder.layer[-1].train()
            optimizer.zero_grad(set_to_none=True)
            loss = _contrastive_loss(encoder, batch, dataset.queries, documents, config.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {arm}, step {step}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(parameters.values()), config.max_grad_norm))
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient in {arm}, step {step}")
            optimizer.step()
            progress["completed_steps"] = step + 1
            progress["nonzero_gradient_steps"] += int(grad_norm > 0)
            progress["trace"].append({"step": step + 1, "epoch": epoch + 1, "loss": float(loss.detach()),
                                      "gradient_l2_before_clip": grad_norm, "examples": len(batch)})
            if (step + 1) % 25 == 0 or step + 1 == total_steps:
                LOGGER.info("LoRA %s step %s/%s: loss %.6f, grad norm %.6f", arm, step + 1, total_steps, float(loss.detach()), grad_norm)
            if (step + 1) % config.checkpoint_every == 0 or step + 1 == total_steps:
                _save_checkpoint(arm_dir, encoder, optimizer, progress, identity)
            if progress_callback is not None:
                progress_callback(arm, step + 1)
        final_validation_loss = _validation_loss(encoder, plan["validation"], dataset.queries, documents, config)
        deltas = [parameters[name].detach() - initial_parameters[name] for name in parameters]
        delta_l2 = math.sqrt(sum(float((delta.double() ** 2).sum()) for delta in deltas))
        if delta_l2 <= 0 or progress["nonzero_gradient_steps"] == 0:
            raise RuntimeError("Training produced no parameter updates; refusing to report a trained adapter")
        adapter_path = arm_dir / "adapter.safetensors"
        save_file({name: parameter.detach().cpu().contiguous() for name, parameter in parameters.items()}, str(adapter_path))
        metadata = {
            "schema_version": 1, "arm": arm, "config": asdict(config), "identity": identity, "base_encoder": encoder.metadata,
            "architecture": "Official transformers.BertModel with rank-r LoRA on final Transformer block attention query/value only",
            "objective": "Per-query InfoNCE: one labeled positive and K explicit negatives; no in-batch negatives",
            "trainable_parameter_names": list(parameters), "trainable_parameter_count": sum(parameter.numel() for parameter in parameters.values()),
            "total_parameter_count": sum(parameter.numel() for parameter in encoder.model.parameters()),
            "initial_trainable_sha256": initial_sha256, "final_trainable_sha256": _parameter_digest(parameters),
            "parameter_delta_l2": delta_l2, "parameter_delta_max_abs": max(float(delta.abs().max()) for delta in deltas),
            "adapter_sha256": _file_digest(adapter_path), "completed_steps": progress["completed_steps"],
            "nonzero_gradient_steps": progress["nonzero_gradient_steps"], "trace": progress["trace"],
            "initial_validation_infonce": baseline_validation_loss, "final_validation_infonce": final_validation_loss,
            "mean_training_loss": float(np.mean([row["loss"] for row in progress["trace"]])),
            "training_query_count": len(examples), "validation_query_count": len(plan["validation"]),
            "selection_rule": "Use the predeclared final epoch; validation loss is monitoring only",
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
        _write_json(arm_dir / "metadata.json", metadata)
        results[arm] = metadata
        LOGGER.info("LoRA %s saved: delta L2 %.6g, validation loss %.6f -> %.6f", arm, delta_l2,
                    baseline_validation_loss, final_validation_loss)
        del optimizer, parameters, initial_parameters, encoder
    report = {"schema_version": 1, "config": asdict(config), "plan_sha256": plan["plan_sha256"],
              "train_query_count": len(plan["train_query_ids"]), "validation_query_count": len(plan["validation_query_ids"]),
              "split": plan["split"], "mining_statistics": plan["mining_statistics"], "teacher": plan["teacher"], "arms": results,
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["This is a finite-budget parameter-efficient encoder adaptation, not full LLM fine-tuning.",
                              "Teacher similarity filtering is a heuristic for potential false negatives, not verified truth labels.",
                              "Validation loss uses fixed retrieved negatives and is not a full-corpus retrieval metric; evaluate all frozen arms separately.",
                              "No test judgments are accessed by this training module."]}
    _write_json(output_dir / "training_report.json", report)
    return report
