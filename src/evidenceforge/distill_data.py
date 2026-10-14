"""Audited, query-disjoint samples of the pinned MS MARCO BM25 triplets.

This is training data and a sampled internal validation set, not the official
MS MARCO passage-ranking benchmark. Reading Parquet requires optional pyarrow;
JSONL input is supported by the generic sampler for small offline fixtures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Iterator
import unicodedata

from .modern import _download_verified, _sha256

LOGGER = logging.getLogger(__name__)
DATASET_ID = "sentence-transformers/msmarco-bm25"
DATASET_REVISION = "ce8a493a65af5e872c3c92f72a89e2e99e175f02"
SOURCE_FILE = "triplet/train-00000-of-00001.parquet"
SOURCE_BYTES = 231_262_240
SOURCE_SHA256 = "b624e7bff0654eb1b538e8d7d729b14004e97ef2e5cb1d2656e30a34eb00c5b6"
SOURCE_ROWS = 502_931
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{SOURCE_FILE}"
LICENSE_URL = "https://microsoft.github.io/msmarco/#terms-and-conditions"
ALGORITHM_VERSION = "msmarco-query-hash-sample-v1"
NORMALIZATION = {
    "document": "Unicode NFKC; collapse all whitespace; preserve case, punctuation and digits",
    "query_group": "Document normalization then Unicode casefold; preserve punctuation and digits",
    "document_id": "doc_ + SHA256 of UTF-8 canonical document text",
    "query_id": "query_ + SHA256 of UTF-8 normalized query group text",
}


def canonical_text(text: str) -> str:
    """The actual text emitted and encoded, so a document ID has one text."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalize_query(text: str) -> str:
    return canonical_text(text).casefold()


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def document_id(text: str) -> str:
    return "doc_" + _digest(canonical_text(text)).hex()


def query_id(text: str) -> str:
    return "query_" + _digest(normalize_query(text)).hex()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _iter_triplets(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at source row {index}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Source row {index} is not an object")
                yield row
        return
    if path.suffix.lower() != ".parquet":
        raise ValueError("Source must be a .parquet or .jsonl file")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet preparation requires the optional pyarrow dependency") from exc
    with pq.ParquetFile(path) as source:
        if not {"query", "positive", "negative"} <= set(source.schema_arrow.names):
            raise ValueError("Parquet source is missing query/positive/negative columns")
        for batch in source.iter_batches(batch_size=4096, columns=["query", "positive", "negative"]):
            yield from batch.to_pylist()


def _texts(row: dict) -> tuple[str, str, str]:
    return tuple(canonical_text(row[key]) if isinstance(row.get(key), str) else ""
                 for key in ("query", "positive", "negative"))


@dataclass(slots=True)
class _QueryGroup:
    # Most source queries have one positive: avoid a Python set per singleton.
    positives: bytes | set[bytes] | None = None
    eligible: bool = False

    def add_positive(self, positive: bytes) -> None:
        if self.positives is None:
            self.positives = positive
        elif isinstance(self.positives, set):
            self.positives.add(positive)
        elif self.positives != positive:
            self.positives = {self.positives, positive}

    def is_positive(self, document: bytes) -> bool:
        return document in self.positives if isinstance(self.positives, set) else document == self.positives

    def positive_ids(self) -> list[str]:
        values = self.positives if isinstance(self.positives, set) else [self.positives]
        return sorted("doc_" + value.hex() for value in values if value is not None)


def _sampling_rank(seed: int, group_id: bytes) -> int:
    # Seed is part of a cryptographic hash, not a platform-specific PRNG stream.
    score = _digest(f"{ALGORITHM_VERSION}\n{seed}\n{group_id.hex()}")
    return int.from_bytes(score + group_id, "big")


def _row_priority(texts: tuple[str, str, str], row_index: int) -> tuple[bytes, int]:
    # Content determines the representative, not input order; row index only
    # breaks ties between repeated identical canonical triplets.
    return _digest(_json(texts)), row_index


def _write_atomic(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_manifest(output_dir: Path, protocol: dict) -> dict | None:
    path = output_dir / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol:
        raise ValueError("Existing distillation manifest has a different protocol; choose another output directory")
    if manifest.get("protocol_sha256") != _digest(_json(protocol)).hex():
        raise ValueError("Prepared distillation protocol integrity failure")
    for name in ("train.jsonl", "validation.jsonl"):
        artifact = output_dir / name
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"Prepared distillation artifact integrity failure: {artifact}")
    LOGGER.info("Using verified prepared data in %s", output_dir)
    return manifest


def prepare_triplets(
    source_path: Path, output_dir: Path, *, n_train: int = 12_000,
    n_validation: int = 1_000, seed: int = 42, source_metadata: dict | None = None,
    expected_sha256: str | None = None, expected_rows: int | None = None,
) -> dict:
    """Scan the entire source twice, audit labels, and choose one row per query.

    Query groups are sampled by the smallest seeded SHA256 ranks among *all*
    groups with a valid negative. The first n_train groups form train, the next
    n_validation validation. Both partitions are disjoint by normalized query;
    documents may overlap and that overlap is explicitly counted.
    """
    if any(type(value) is not int for value in (n_train, n_validation, seed)):
        raise ValueError("Sample counts and seed must be integers")
    if n_train <= 0 or n_validation <= 0:
        raise ValueError("Train and validation counts must both be positive")
    source_path, output_dir = Path(source_path), Path(output_dir)
    if source_path.resolve() in {(output_dir / name).resolve()
                                 for name in ("train.jsonl", "validation.jsonl", "manifest.json")}:
        raise ValueError("Output must not overwrite its source")
    source_sha256 = _sha256(source_path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError("Source SHA256 does not match the pinned source")
    metadata = dict(source_metadata or {"kind": "local_fixture", "license": "caller-supplied"})
    protocol = {
        "algorithm_version": ALGORITHM_VERSION, "source_sha256": source_sha256,
        "source_bytes": source_path.stat().st_size, "source_metadata": metadata,
        "expected_source_rows": expected_rows, "n_train": n_train, "n_validation": n_validation,
        "sampling_seed": seed, "normalization": NORMALIZATION,
        "sampling": "Globally smallest SHA256(version + newline + seed + newline + normalized-query SHA256)",
        "representative": "Smallest canonical triplet SHA256, then zero-based source row index",
        "selection_uses_heldout_data": False,
    }
    cached = _cached_manifest(output_dir, protocol)
    if cached is not None:
        return cached
    groups: dict[bytes, _QueryGroup] = {}
    source_document_ids: set[bytes] = set()
    audit = dict.fromkeys([
        "source_rows", "invalid_triplet_rows", "valid_query_occurrences", "valid_document_text_occurrences",
        "document_normalization_changed_occurrences", "identical_positive_negative_rows",
        "negative_is_other_known_positive_rows", "eligible_triplet_rows", "eligible_unique_queries",
    ], 0)
    # All positives are recorded, including a row with a malformed negative.
    # Thus a positive later in the file can invalidate an earlier negative.
    for row_index, row in enumerate(_iter_triplets(source_path)):
        query, positive, negative = _texts(row)
        audit["source_rows"] += 1
        if not all((query, positive, negative)):
            audit["invalid_triplet_rows"] += 1
        if query:
            audit["valid_query_occurrences"] += 1
            group = groups.setdefault(_digest(normalize_query(query)), _QueryGroup())
            if positive:
                group.add_positive(_digest(positive))
        for key, text in (("positive", positive), ("negative", negative)):
            if text:
                source_document_ids.add(_digest(text))
                audit["valid_document_text_occurrences"] += 1
                audit["document_normalization_changed_occurrences"] += text != row[key]
        if (row_index + 1) % 100_000 == 0:
            LOGGER.info("Positive-label audit: %s rows, %s query groups", row_index + 1, len(groups))
    if expected_rows is not None and audit["source_rows"] != expected_rows:
        raise ValueError(f"Source row count {audit['source_rows']} != expected {expected_rows}")
    audit["unique_normalized_queries"] = len(groups)
    audit["repeated_normalized_query_occurrences"] = audit["valid_query_occurrences"] - len(groups)
    audit["unique_document_texts"] = len(source_document_ids)
    audit["deduplicated_document_occurrences"] = audit["valid_document_text_occurrences"] - len(source_document_ids)
    audit["queries_with_multiple_known_positives"] = sum(isinstance(g.positives, set) for g in groups.values())
    del source_document_ids
    # A bounded max heap retains only the K smallest query ranks. Once a group
    # is evicted, the decreasing rank threshold means it can never re-enter.
    capacity = n_train + n_validation
    heap: list[tuple[int, bytes]] = []
    selected: dict[bytes, tuple[tuple[bytes, int], dict]] = {}
    second_pass_rows = 0
    for row_index, row in enumerate(_iter_triplets(source_path)):
        second_pass_rows += 1
        texts = _texts(row)
        if not all(texts):
            continue
        query, positive, negative = texts
        group_id = _digest(normalize_query(query))
        group = groups.get(group_id)
        if group is None:
            raise ValueError("Source changed between audit passes")
        positive_id, negative_id = _digest(positive), _digest(negative)
        if positive_id == negative_id:
            audit["identical_positive_negative_rows"] += 1
            continue
        if group.is_positive(negative_id):
            audit["negative_is_other_known_positive_rows"] += 1
            continue
        audit["eligible_triplet_rows"] += 1
        if not group.eligible:
            group.eligible = True
            audit["eligible_unique_queries"] += 1
        priority = _row_priority(texts, row_index)
        if group_id in selected:
            if priority >= selected[group_id][0]:
                continue
        else:
            rank = _sampling_rank(seed, group_id)
            if len(heap) == capacity:
                if rank >= -heap[0][0]:
                    continue
                _, evicted = heapq.heapreplace(heap, (-rank, group_id))
                del selected[evicted]
            else:
                heapq.heappush(heap, (-rank, group_id))
        selected[group_id] = (priority, {
            "query_id": "query_" + group_id.hex(), "query": query,
            "positive_id": "doc_" + positive_id.hex(), "positive": positive,
            "negative_id": "doc_" + negative_id.hex(), "negative": negative,
            "known_positive_ids": group.positive_ids(), "source_row_index": row_index,
        })
        if (row_index + 1) % 100_000 == 0:
            LOGGER.info("Sampling audit: %s rows, %s eligible queries", row_index + 1,
                        audit["eligible_unique_queries"])
    if second_pass_rows != audit["source_rows"] or _sha256(source_path) != source_sha256:
        raise ValueError("Source changed between audit passes")
    if len(selected) != capacity:
        raise ValueError(f"Only {len(selected)} eligible unique queries; requested {capacity}")
    ranked_ids = sorted(selected, key=lambda value: _sampling_rank(seed, value))
    rows = [selected[value][1] for value in ranked_ids]
    split_rows = {"train": rows[:n_train], "validation": rows[n_train:]}
    split_docs = {
        split: {row[field] for row in values for field in ("positive_id", "negative_id")}
        for split, values in split_rows.items()
    }
    audit["selected_document_occurrences"] = capacity * 2
    audit["selected_unique_document_texts"] = len(split_docs["train"] | split_docs["validation"])
    audit["selected_deduplicated_document_occurrences"] = capacity * 2 - audit["selected_unique_document_texts"]
    audit["train_validation_shared_document_texts"] = len(split_docs["train"] & split_docs["validation"])
    manifest = {
        "protocol": protocol, "protocol_sha256": _digest(_json(protocol)).hex(), "audit": audit,
        "selected_query_ids_sha256": _digest("\n".join(row["query_id"] for row in rows) + "\n").hex(),
        "split_query_ids_sha256": {
            split: _digest("\n".join(row["query_id"] for row in values) + "\n").hex()
            for split, values in split_rows.items()
        },
        "artifacts": {},
        "limitations": [
            "Validation is a query-disjoint sample of training triplets, not the official MS MARCO benchmark.",
            "Known positives are limited to this pinned triplet source, not the complete MS MARCO qrels.",
            "Static upstream BM25 negatives are used; this pipeline does not mine the full 8.8M-passage corpus.",
            "Query grouping retains digits and punctuation; paraphrases may still cross the split.",
            "Document normalization preserves case; case variants and semantic duplicates are not merged.",
            "Documents may appear in both partitions; this is query-disjoint, not document-disjoint validation.",
            "The data terms differ from the repository code license; do not assume permission to redistribute text.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in split_rows.items():
        filename = f"{split}.jsonl"
        target = output_dir / filename
        _write_atomic(target, "".join(_json(row) + "\n" for row in values))
        manifest["artifacts"][filename] = {
            "rows": len(values), "bytes": target.stat().st_size, "sha256": _sha256(target),
        }
    # A manifest is published last, so interrupted partial output cannot be
    # mistaken for a completed, verified preparation.
    _write_atomic(output_dir / "manifest.json", _json(manifest) + "\n")
    LOGGER.info("Prepared %s train + %s validation queries from %s source rows", n_train, n_validation,
                audit["source_rows"])
    return manifest


def prepare_msmarco_distill(
    data_dir: Path, output_dir: Path, *, n_train: int = 12_000, n_validation: int = 1_000,
    seed: int = 42, source_path: Path | None = None,
) -> dict:
    """Download or verify the pinned 231 MB source, then sample its full scan."""
    source_path = Path(source_path) if source_path is not None else Path(data_dir) / "source" / "msmarco-bm25-triplet.parquet"
    source_path = _download_verified(SOURCE_URL, source_path, SOURCE_BYTES, SOURCE_SHA256)
    metadata = {
        "dataset_id": DATASET_ID, "revision": DATASET_REVISION, "config": "triplet", "split": "train",
        "source_file": SOURCE_FILE, "url": SOURCE_URL,
        "dataset_card": f"https://huggingface.co/datasets/{DATASET_ID}/blob/{DATASET_REVISION}/README.md",
        "license": {
            "name": "MS MARCO non-commercial research terms", "url": LICENSE_URL,
            "note": "Free for non-commercial research; no extension of underlying document IP rights. "
                    "The dataset card does not declare an SPDX license. Repository code licensing does not apply.",
        },
    }
    return prepare_triplets(source_path, output_dir, n_train=n_train, n_validation=n_validation, seed=seed,
                            source_metadata=metadata, expected_sha256=SOURCE_SHA256, expected_rows=SOURCE_ROWS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/distillation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/distillation/prepared"))
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--n-train", type=int, default=12_000)
    parser.add_argument("--n-validation", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = prepare_msmarco_distill(**vars(args))
    print(json.dumps({"protocol_sha256": manifest["protocol_sha256"], "audit": manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()
