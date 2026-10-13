"""Frozen, resumable full-corpus SciFact evaluation of quantized Qwen3.

This module never imports torch and never runs an experiment merely on import.
The CPU runtime must be scheduled separately from student training on this host.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

from .data import load_dataset
from .modern import QwenEmbeddingEncoder, _memory_snapshot
from .neural_experiment import digest, evaluate_vectors, write_json

LOGGER = logging.getLogger(__name__)
EXPECTED_DOCUMENTS = 5183
EXPECTED_TEST_QUERIES = 300
EXPECTED_TRAIN_QUERIES = 809
CHUNK_SIZE = 16
METHOD = "qwen3_06b_q8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_all(encoder, texts: list[str], *, is_query: bool, phase: str,
                output_dir: Path, protocol_hash: str) -> tuple[np.ndarray, dict]:
    chunks = []
    summary = {"items": 0, "cache_hits": 0, "truncated": 0, "tokens": 0,
               "uncached_inference_seconds": 0.0, "recorded_original_inference_seconds": 0.0,
               "recorded_threads_item_counts": {}, "uncached_item_seconds": []}
    started = time.perf_counter()
    for start in range(0, len(texts), CHUNK_SIZE):
        values = encoder.encode(texts[start:start + CHUNK_SIZE], is_query=is_query, use_cache=True,
                                cache_namespace=phase)
        chunks.append(values)
        stats = encoder.last_stats
        for key in ("items", "cache_hits", "truncated", "tokens", "recorded_original_inference_seconds"):
            summary[key] += stats[key]
        summary["uncached_item_seconds"].extend(stats["uncached_item_seconds"])
        summary["uncached_inference_seconds"] += sum(stats["uncached_item_seconds"])
        for key, count in stats["recorded_threads_item_counts"].items():
            summary["recorded_threads_item_counts"][key] = summary["recorded_threads_item_counts"].get(key, 0) + count
        summary["wall_seconds_this_invocation"] = time.perf_counter() - started
        summary["embedding_fingerprint"] = stats["embedding_fingerprint"]
        LOGGER.info("Qwen %s: %d/%d, cached=%d, truncated=%d, elapsed=%.1fs",
                    phase, summary["items"], len(texts), summary["cache_hits"], summary["truncated"],
                    summary["wall_seconds_this_invocation"])
        write_json(output_dir / f"{phase}_progress.json", {
            "protocol_sha256": protocol_hash, "phase": phase, "total": len(texts),
            **{key: value for key, value in summary.items() if key != "uncached_item_seconds"},
        })
    latencies = summary.pop("uncached_item_seconds")
    summary["uncached_item_latency_seconds"] = {
        "count": len(latencies),
        "p50": float(np.quantile(latencies, 0.5)) if latencies else None,
        "p95": float(np.quantile(latencies, 0.95)) if latencies else None,
    }
    return np.concatenate(chunks), summary


def run_modern_experiment(data_dir: Path, cache_dir: Path, output_dir: Path,
                          threads: int = 2, max_tokens: int = 128, encode_extra_train: bool = False,
                          backend: str = "cpu", device: str = "Vulkan0") -> dict:
    """Evaluate all 300 SciFact test queries against all 5,183 documents.

    No sampling, candidate restriction, label-based document injection, tuning,
    or teacher-score fitting is performed. An interrupted encoding resumes from
    per-text checkpoints; historical inference cost and new work are separate.
    """
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    dataset = load_dataset(Path(data_dir))
    documents = sorted(dataset.docs, key=lambda row: str(row["_id"]))
    query_ids = sorted(dataset.qrels.get("test", {}))
    train_ids = sorted(dataset.qrels.get("train", {})) if encode_extra_train else []
    if len(documents) != EXPECTED_DOCUMENTS or len(query_ids) != EXPECTED_TEST_QUERIES:
        raise ValueError("This experiment requires the complete 5,183-document, 300-test-query BEIR SciFact dataset")
    if any(not any(rel > 0 for rel in dataset.qrels["test"][qid].values()) for qid in query_ids):
        raise ValueError("Every declared SciFact test query must have a positive relevance judgment")
    if encode_extra_train and (len(train_ids) != EXPECTED_TRAIN_QUERIES or set(train_ids) & set(query_ids)):
        raise ValueError("Optional teacher encodings require all 809 official train query IDs, disjoint from test IDs")
    archived_corpus_hash = dataset.metadata.get("corpus_sha256")
    if archived_corpus_hash and archived_corpus_hash != dataset.corpus_hash:
        raise ValueError("Dataset corpus no longer matches its downloaded manifest")
    document_texts = [doc["title"] + "\n" + doc["text"] for doc in documents]
    query_texts = [dataset.queries[qid] for qid in query_ids]
    encoder = QwenEmbeddingEncoder(cache_dir, threads=threads, max_tokens=max_tokens, backend=backend, device=device)
    spec = {
        "schema": 1, "method": METHOD, "encoder": dict(encoder.metadata),
        "source_corpus_sha256": dataset.corpus_hash,
        "source_queries_sha256": digest(dataset.queries),
        "source_test_qrels_sha256": digest(dataset.qrels["test"]),
        "ordered_document_ids_sha256": digest([str(doc["_id"]) for doc in documents]),
        "ordered_test_query_ids": query_ids,
        "extra_train_encoding": {"enabled": encode_extra_train, "ordered_query_ids": train_ids,
                                 "scope": "Query text only, separately cached; never included in test quality metrics"},
        "document_count": len(documents), "query_count": len(query_ids),
        "document_text": "title + newline + abstract; no document instruction",
        "query_text": "Instruct: <fixed scientific evidence instruction>\\nQuery:<claim>",
        "ranking_scope": "Every test query scores the entire original corpus using exact normalized dot products",
        "saved_ranking_depth": 50, "tie_break": "document ID lexicographic ascending",
        "ndcg_gain": "Original qrel relevance, linear graded gain (trec_eval ndcg_cut convention)",
        "checkpoint_chunk_size": CHUNK_SIZE,
        "selection": "All 300 official test queries and all 5,183 corpus documents; no test-driven model selection",
        "token_budget_comparison": (
            "128 tokens by default for both Qwen and the student, but tokenizers differ; "
            "Qwen query instructions consume part of its limit. Equal token counts are not equal text coverage."
        ),
        "quality_claim": "Q8_0 and explicit truncation are part of this local model configuration; not an official full-precision score",
    }
    protocol_path = output_dir / "protocol.json"
    spec_hash = digest(spec)
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["spec_sha256"] != spec_hash or digest(protocol["spec"]) != spec_hash:
            raise ValueError("Frozen Qwen protocol differs; use a fresh output directory")
    else:
        protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
                    "spec_sha256": spec_hash, "spec": spec}
        # Freeze input identities and inference choices before starting any model
        # or looking at output rankings.
        write_json(protocol_path, protocol)
    protocol_hash = _sha256(protocol_path)
    report = {
        "protocol_sha256": protocol_hash, "spec_sha256": spec_hash,
        "scope": "Complete SciFact test; exact retrieval over complete original corpus",
        "environment": {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy")},
        "memory_before": _memory_snapshot(),
        "limitations": [
            "This is a pretrained quantized model baseline; no Qwen weights were trained here.",
            "Model pretraining contamination is not independently audited.",
            "128-token budgets do not imply identical text coverage across the two tokenizers.",
            "Cached historical inference timings may span runs; cache hits are never presented as fresh model throughput.",
            "Dot-product-only ranking latency excludes model loading, document indexing, and query encoding.",
        ],
    }
    started = time.perf_counter()
    try:
        doc_vectors, doc_stats = _encode_all(encoder, document_texts, is_query=False, phase="documents",
                                             output_dir=output_dir, protocol_hash=protocol_hash)
        query_vectors, query_stats = _encode_all(encoder, query_texts, is_query=True, phase="test_queries",
                                                 output_dir=output_dir, protocol_hash=protocol_hash)
        report["encoder"] = encoder.metadata
        report["encoding"] = {"documents": doc_stats, "test_queries": query_stats}
        if encode_extra_train:
            train_vectors, train_stats = _encode_all(
                encoder, [dataset.queries[qid] for qid in train_ids], is_query=True, phase="train_queries",
                output_dir=output_dir, protocol_hash=protocol_hash,
            )
            report["encoding"]["train_queries"] = train_stats
            teacher_dir = cache_dir / "modern" / "teacher_vectors" / spec_hash
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_path = teacher_dir / "vectors.npz"
            temporary = teacher_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, document_ids=np.asarray([str(doc["_id"]) for doc in documents]),
                                    document_vectors=doc_vectors, train_query_ids=np.asarray(train_ids),
                                    train_query_vectors=train_vectors)
            os.replace(temporary, teacher_path)
            teacher_metadata = {
                "path_relative_to_cache": teacher_path.relative_to(cache_dir).as_posix(),
                "sha256": _sha256(teacher_path), "protocol_sha256": protocol_hash,
                "spec_sha256": spec_hash, "document_count": len(documents), "train_query_count": len(train_ids),
                "embedding_dimension": int(doc_vectors.shape[1]),
                "test_query_vectors_included": False,
                "purpose": "Frozen teacher cosine scores for separately preregistered train-only negative filtering",
            }
            write_json(teacher_dir / "manifest.json", teacher_metadata)
            report["teacher_vector_artifact"] = teacher_metadata
        report["memory_after_encoding"] = _memory_snapshot(encoder.process)
    finally:
        encoder.close()
    # Release Qwen before the full-corpus numerical ranking stage.
    results = evaluate_vectors(dataset, doc_vectors, query_vectors, query_ids, output_dir / "runs", METHOD)
    report["result"] = results
    report["wall_seconds_this_invocation"] = time.perf_counter() - started
    report["artifacts"] = {
        "trec": {"path": f"runs/{METHOD}.trec", "sha256": _sha256(output_dir / "runs" / f"{METHOD}.trec")},
        "per_query": {"path": f"runs/{METHOD}.per_query.json",
                      "sha256": _sha256(output_dir / "runs" / f"{METHOD}.per_query.json")},
    }
    write_json(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/modern/scifact-full"))
    parser.add_argument("--threads", type=int, default=2, choices=(1, 2))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--encode-extra-train", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--device", default="Vulkan0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_modern_experiment(args.data_dir, args.cache_dir, args.output_dir, args.threads, args.max_tokens,
                                   args.encode_extra_train, args.backend, args.device)
    print(json.dumps({"metrics": result["result"]["metrics"], "report": str(args.output_dir / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
