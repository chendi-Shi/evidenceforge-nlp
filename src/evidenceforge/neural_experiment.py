"""Frozen-protocol neural adaptation and full-corpus transfer evaluation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import time

import numpy as np

from .data import load_dataset
from .evaluation import macro_metrics, paired_bootstrap, query_metrics
from .model_assets import download_float_minilm
from .transfer_data import download_external, load_external

LOGGER = logging.getLogger(__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cached_variants(encoder, texts: list[str], cache_dir: Path, *, batch_size: int = 8, chunk_size: int = 64) -> dict[str, np.ndarray]:
    """Checkpoint bounded chunks; validate identity, shapes and norms on reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = digest({"encoder": encoder.metadata, "texts": texts, "chunk_size": chunk_size})
    directory = cache_dir / identity
    directory.mkdir(exist_ok=True)
    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        path = directory / f"{start:07d}.npz"
        checksum_path = path.with_suffix(".sha256")
        if path.exists() and checksum_path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum_path.read_text().strip():
                raise ValueError(f"Corrupt embedding chunk: {path}")
            with np.load(path, allow_pickle=False) as loaded:
                vectors = {name: loaded[name] for name in loaded.files}
        else:
            vectors = encoder.encode(chunk, batch_size=batch_size)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **vectors)
            os.replace(temporary, path)
            checksum_path.write_text(hashlib.sha256(path.read_bytes()).hexdigest(), encoding="ascii")
        if parts and set(vectors) != set(parts):
            raise ValueError("Embedding variant schema changed between chunks")
        for name, values in vectors.items():
            if values.ndim != 2 or len(values) != len(chunk) or not np.isfinite(values).all():
                raise ValueError(f"Invalid embedding chunk: {name}")
            if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4):
                raise ValueError(f"Non-normalized embedding chunk: {name}")
            parts.setdefault(name, []).append(values)
        if start % (chunk_size * 4) == 0 or start + chunk_size >= len(texts):
            LOGGER.info("Shared-prefix encodings: %s/%s texts", min(start + chunk_size, len(texts)), len(texts))
    write_json(directory / "manifest.json", {"identity": identity, "encoder": encoder.metadata, "texts": len(texts),
                                               "chunk_size": chunk_size})
    return {name: np.concatenate(values) for name, values in parts.items()}


def evaluate_vectors(dataset, doc_vectors: np.ndarray, query_vectors: np.ndarray, query_ids: list[str],
                     output_dir: Path, method: str, split: str = "test") -> dict:
    """Rank every document; save explicit TREC ranks and graded per-query metrics."""
    documents = sorted(dataset.docs, key=lambda row: str(row["_id"]))
    ids = np.asarray([str(row["_id"]) for row in documents])
    if len(doc_vectors) != len(ids) or len(query_vectors) != len(query_ids):
        raise ValueError("Vector/ID alignment mismatch")
    if doc_vectors.ndim != 2 or query_vectors.ndim != 2 or doc_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Embedding dimensions do not match")
    if not np.isfinite(doc_vectors).all() or not np.isfinite(query_vectors).all():
        raise ValueError("Nonfinite evaluation vectors")
    if len(query_ids) != len(set(query_ids)) or any(qid not in dataset.qrels[split] for qid in query_ids):
        raise ValueError("Duplicate or unjudged evaluation queries")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, elapsed = [], []
    run_path = output_dir / f"{method}.trec"
    with run_path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid, vector in zip(query_ids, query_vectors):
            started = time.perf_counter()
            scores = doc_vectors @ vector
            order = np.argsort(-scores, kind="stable")[:50]
            elapsed.append(time.perf_counter() - started)
            ranking = ids[order].tolist()
            metrics = query_metrics(ranking, dataset.qrels[split][qid])
            rows.append({"query_id": qid, **metrics})
            for rank, index in enumerate(order, 1):
                handle.write(f"{qid} Q0 {ids[index]} {rank} {float(scores[index]):.10g} {method}\n")
    write_json(output_dir / f"{method}.per_query.json", rows)
    return {"metrics": macro_metrics(rows), "per_query": rows, "query_count": len(rows),
            "corpus_size": len(ids), "ranking_scope": "entire corpus", "split": split,
            "tie_break": "document ID lexicographic ascending", "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            "dot_product_only_seconds": {"p50": float(np.quantile(elapsed, .5)), "p95": float(np.quantile(elapsed, .95))}}


def _versions() -> dict:
    names = ("torch", "transformers", "safetensors", "numpy", "tokenizers", "onnxruntime")
    return {name: importlib.metadata.version(name) for name in names}


def run_adaptation_experiment(data_dir: Path, transfer_dir: Path, cache_dir: Path, output_dir: Path,
                             *, stage: str = "all", epochs: int = 2, max_train_queries: int = 10000,
                             threads: int = 2) -> dict:
    from .adaptation import ARMS, AdaptationConfig, train_adaptation

    if stage not in {"prepare", "train", "evaluate", "all"}:
        raise ValueError("Unknown adaptation stage")
    config = AdaptationConfig(epochs=epochs, max_train_queries=max_train_queries, threads=threads)
    config.validate()
    output_dir = Path(output_dir)
    base_dir = download_float_minilm(cache_dir)
    scifact = load_dataset(data_dir)
    transfer = download_external("nfcorpus", transfer_dir)
    spec = {"schema": 2, "training": asdict(config), "source_corpus_sha256": scifact.corpus_hash,
            "source_train_qrels_sha256": digest(scifact.qrels["train"]),
            "source_queries_sha256": digest(scifact.queries), "source_test_qrels_sha256": digest(scifact.qrels["test"]),
            "transfer_corpus_sha256": transfer.corpus_hash,
            "ndcg_gain": "Original qrel relevance, linear graded gain (trec_eval ndcg_cut convention)",
            "base_assets": json.loads((base_dir / "assets_manifest.json").read_text()),
            "arms": ["base", *ARMS], "selection": "All predeclared final-epoch adapters are evaluated; no best-test selection",
            "evaluation": "Full source test and full NFCorpus test, full corpus exact cosine ranking, graded nDCG",
            "primary_comparison": "filtered minus retrieved nDCG@10; report even when null or negative",
            "secondary_comparisons": "Each arm minus same-float/same-token-length base",
            "teacher": "Frozen INT8 MiniLM retrieval encoder; similarity is a risk proxy, not a ground-truth false-negative label",
            "transfer_scope": "NFCorpus tests query/task distribution transfer within overlapping biomedical subject matter",
            "seed_scope": "One optimization seed; query bootstrap does not estimate training variance"}
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["spec_sha256"] != digest(spec):
            raise ValueError("Experiment protocol differs; select a fresh output directory")
    else:
        protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "spec_sha256": digest(spec), "spec": spec}
        write_json(protocol_path, protocol)
    if stage == "prepare":
        return protocol
    if stage in {"train", "all"}:
        from .retrieval import RetrievalIndex
        index = RetrievalIndex(scifact, cache_dir, threads=threads).build()
        train_adaptation(scifact, index, base_dir, output_dir / "training", config)
        del index
    if stage == "train":
        return json.loads((output_dir / "training" / "training_report.json").read_text())
    from .adaptation import MultiAdapterEncoder
    adapters = {arm: output_dir / "training" / arm for arm in ARMS}
    plan = json.loads((output_dir / "training" / "training_plan.json").read_text(encoding="utf-8"))
    plan_hash = plan["plan_sha256"]
    if digest({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan_hash:
        raise ValueError("Training plan content hash mismatch")
    if digest(plan["config"]) != digest(asdict(config)) or plan["corpus_sha256"] != scifact.corpus_hash:
        raise ValueError("Training plan does not match the frozen experiment protocol")
    if plan["external_teacher_scores"]:
        raise ValueError("This protocol requires the predeclared frozen MiniLM teacher")
    selected = plan["train_query_ids"]
    if (plan["selected_train_qrels_sha256"] != digest({qid: scifact.qrels["train"][qid] for qid in selected})
            or plan["selected_train_queries_sha256"] != digest({qid: scifact.queries[qid] for qid in selected})):
        raise ValueError("Training input identity mismatch")
    frozen = {}
    for arm, directory in adapters.items():
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (digest(metadata["config"]) != digest(asdict(config))
                or metadata["identity"]["plan_sha256"] != plan_hash or metadata["arm"] != arm):
            raise ValueError("Adapter training identity does not match this experiment")
        frozen[arm] = metadata["adapter_sha256"]
    frozen_path = output_dir / "frozen_adapters.json"
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text(encoding="utf-8"))
        if previous["adapters"] != frozen or previous["protocol_sha256"] != protocol_hash:
            raise ValueError("Frozen evaluation model identity changed")
    else:
        write_json(frozen_path, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                                "adapters": frozen, "protocol_sha256": protocol_hash})
    encoder = MultiAdapterEncoder(base_dir, adapters, config=config)
    report = {"protocol_sha256": protocol_hash, "spec_sha256": protocol["spec_sha256"], "encoder": encoder.metadata, "environment": _versions(),
              "python": platform.python_version(), "datasets": {}, "comparisons": {},
              "limitations": ["Single optimization seed and limited source training data.",
                              "Similarity filtering may remove informative true negatives; no manual false-negative labels.",
                              "NFCorpus and SciFact share biomedical subject matter; this is not unrelated-domain transfer.",
                              "Query bootstrap is neither family-clustered nor adjusted for multiple comparisons.",
                              "Source benchmark has six normalized train/test claim overlaps, retained and disclosed in v1."]}
    started = time.perf_counter()
    for name, dataset in (("scifact", scifact), ("nfcorpus", load_external("nfcorpus", transfer_dir))):
        documents = sorted(dataset.docs, key=lambda row: str(row["_id"]))
        query_ids = sorted(qid for qid, rels in dataset.qrels["test"].items() if any(rel > 0 for rel in rels.values()))
        texts = [doc["title"] + "\n" + doc["text"] for doc in documents] + [dataset.queries[qid] for qid in query_ids]
        LOGGER.info("Evaluating %s: %s documents, %s queries, all four encoder variants", name, len(documents), len(query_ids))
        variants = cached_variants(encoder, texts, Path(cache_dir) / "adaptation_vectors")
        methods = {}
        for arm, values in variants.items():
            methods[arm] = evaluate_vectors(dataset, values[:len(documents)], values[len(documents):], query_ids,
                                            output_dir / name / "runs", arm)
            # Reuse the evaluated vectors in CLI/API inference without re-encoding.
            from .retrieval import embedding_cache_key
            key = embedding_cache_key(dataset.corpus_hash, encoder.metadata[arm])
            cache_path = Path(cache_dir) / f"embeddings-{key}.npy"
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, values[:len(documents)], allow_pickle=False)
            os.replace(temporary, cache_path)
        comparisons = {}
        for treatment, reference in [("filtered", "retrieved"), *[(arm, "base") for arm in ARMS]]:
            comparisons[f"{treatment}_minus_{reference}"] = paired_bootstrap(
                [row["ndcg@10"] for row in methods[treatment]["per_query"]],
                [row["ndcg@10"] for row in methods[reference]["per_query"]])
        report["datasets"][name] = {"corpus_sha256": dataset.corpus_hash, "methods": methods}
        report["comparisons"][name] = comparisons
        report["evaluation_seconds"] = time.perf_counter() - started
        write_json(output_dir / "report.json", report)
    return report
