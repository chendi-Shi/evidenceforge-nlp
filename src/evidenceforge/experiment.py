"""One-command, auditable SciFact retrieval/reranking experiment."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np

from .data import download_scifact, load_dataset
from .evaluation import METRIC_NAMES, candidate_ceiling, macro_metrics, paired_bootstrap, query_metrics
from .ranker import BASE_CONFIG, GRID, POOL_SIZE, SEED, EvidenceRanker, normalized_claim
from .retrieval import RetrievalIndex

METHODS = ("bm25", "dense", "rrf", "learned", "learned_no_dense")
LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _latency(values: list[float]) -> dict[str, float]:
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.quantile(values, 0.5)), "p95_ms": float(np.quantile(values, 0.95))}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(data_dir: Path, cache_dir: Path, output_dir: Path, threads: int = 2) -> dict:
    """Tune on grouped official-train split, refit, then evaluate frozen models.

    The benchmark does not promise that learning improves an already strong
    retriever. The validation-selected recommendation is frozen before test access.
    """
    data_dir, cache_dir, output_dir = map(Path, (data_dir, cache_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "BEIR SciFact",
        "dataset_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "primary_metric": "ndcg@10", "secondary_metrics": ["mrr@10", "recall@10", "recall@50"],
        "methods": list(METHODS), "candidate_pool_size": POOL_SIZE, "seed": SEED,
        "hyperparameter_grid": [dict(BASE_CONFIG, **config) for config in GRID],
        "selection": "Evidence/normalized-claim grouped 80/20 partition of official train; maximize validation nDCG@10; refit on all official train.",
        "test_policy": "One frozen-model evaluation of all official test queries. No test-driven model or hyperparameter selection.",
        "bootstrap": {"unit": "query", "samples": 2000, "seed": SEED, "confidence": 0.95, "multiple_comparison_correction": False},
    }
    _write_json(output_dir / "protocol.json", protocol)
    started = time.perf_counter()
    download_scifact(data_dir)
    dataset = load_dataset(data_dir)
    if "train" not in dataset.qrels or "test" not in dataset.qrels:
        raise ValueError("Benchmark requires official train and test qrels")
    index = RetrievalIndex(dataset, cache_dir=cache_dir, dense=True, threads=threads)
    LOGGER.info("Building index over all %s corpus documents", len(dataset.docs))
    index.build()
    LOGGER.info("Starting full-feature ranker training")
    ranker = EvidenceRanker(include_dense=True)
    ranker.fit(dataset, index, threads=threads)
    ranker.save(output_dir / "model")
    LOGGER.info("Starting no-dense/RRF-feature ablation training")
    ablation = EvidenceRanker(include_dense=False)
    ablation.fit(dataset, index, threads=threads)
    ablation.save(output_dir / "ablation_model")
    frozen_at = datetime.now(timezone.utc).isoformat()
    recommended_method = ranker.metadata["validation_recommendation"]
    frozen = {
        "frozen_at_utc": frozen_at, "recommended_method": recommended_method,
        "model_sha256": _file_hash(output_dir / "model" / "model.txt"),
        "ablation_model_sha256": _file_hash(output_dir / "ablation_model" / "model.txt"),
        "selected_config": ranker.metadata["selected_config"], "ablation_selected_config": ablation.metadata["selected_config"],
        "protocol_sha256": _file_hash(output_dir / "protocol.json"),
    }
    _write_json(output_dir / "frozen_models.json", frozen)
    LOGGER.info("Models frozen; validation recommendation: %s. Beginning test evaluation", recommended_method)

    test_qids = sorted(qid for qid, rels in dataset.qrels["test"].items() if any(int(rel) > 0 for rel in rels.values()))
    if not test_qids:
        raise ValueError("The official test split contains no positively judged queries")
    rows, errors, ceilings = [], [], []
    latency = {method: [] for method in METHODS}
    all_metrics = {method: [] for method in METHODS}
    runs = {method: [] for method in METHODS}
    document_titles = {str(doc["_id"]): doc["title"] for doc in dataset.docs}
    evaluation_started = time.perf_counter()
    for query_number, qid in enumerate(test_qids, 1):
        query, qrels = dataset.queries[qid], dataset.qrels["test"][qid]
        rankings = {}
        for method in ("bm25", "dense", "rrf"):
            t0 = time.perf_counter()
            rankings[method] = index.search(query, method=method, k=POOL_SIZE)
            latency[method].append((time.perf_counter() - t0) * 1000)
        for method, model in (("learned", ranker), ("learned_no_dense", ablation)):
            t0 = time.perf_counter()
            rankings[method] = model.rerank(query, rankings["rrf"])
            latency[method].append(latency["rrf"][-1] + (time.perf_counter() - t0) * 1000)
        metrics = {method: query_metrics(rankings[method], qrels) for method in METHODS}
        ceiling = candidate_ceiling(rankings["rrf"], qrels)
        ceilings.append(ceiling)
        rows.append({"query_id": qid, "query": query, "relevant_document_ids": sorted(d for d, rel in qrels.items() if int(rel) > 0),
                     "metrics": metrics, "candidate_ceiling": ceiling})
        for method in METHODS:
            all_metrics[method].append(metrics[method])
            score_key = {"bm25": "bm25_score", "dense": "dense_score", "rrf": "rrf_score"}.get(method, "ltr_score")
            for position, hit in enumerate(rankings[method], start=1):
                runs[method].append(f"{qid} Q0 {hit['doc_id']} {position} {float(hit[score_key]):.17g} evidenceforge_{method}")
        if metrics["learned"]["recall@10"] < 1.0 or metrics["learned"]["ndcg@10"] < metrics["rrf"]["ndcg@10"]:
            if ceiling["candidate_recall@50"] == 0:
                category = "candidate_retrieval_miss"
            elif metrics["learned"]["recall@10"] == 0:
                category = "reranking_miss"
            elif metrics["learned"]["ndcg@10"] < metrics["rrf"]["ndcg@10"]:
                category = "regression_against_rrf"
            else:
                category = "partial_evidence_recall"
            errors.append({"query_id": qid, "query": query, "category": category, "metrics": metrics,
                           "candidate_ceiling": ceiling,
                           "gold": [{"doc_id": doc, "title": document_titles.get(doc, ""), "relevance": rel} for doc, rel in qrels.items() if int(rel) > 0],
                           "top10": {method: [{"doc_id": hit["doc_id"], "title": hit["title"], "relevant": int(qrels.get(str(hit["doc_id"]), 0)) > 0} for hit in rankings[method][:10]] for method in METHODS}})
        if query_number % 50 == 0 or query_number == len(test_qids):
            LOGGER.info("Test evaluation: %s/%s queries", query_number, len(test_qids))
    evaluation_seconds = time.perf_counter() - evaluation_started
    summary = {method: macro_metrics(all_metrics[method]) for method in METHODS}
    comparisons = {}
    for method in ("bm25", "dense", "learned", "learned_no_dense"):
        comparisons[f"{method}_minus_rrf"] = {
            metric: paired_bootstrap([m[metric] for m in all_metrics[method]], [m[metric] for m in all_metrics["rrf"]])
            for metric in METRIC_NAMES
        }
    comparisons["learned_minus_learned_no_dense"] = {
        metric: paired_bootstrap([m[metric] for m in all_metrics["learned"]], [m[metric] for m in all_metrics["learned_no_dense"]])
        for metric in METRIC_NAMES
    }
    train_qids = sorted(dataset.qrels["train"])
    train_claims = {normalized_claim(dataset.queries[qid]) for qid in train_qids}
    test_claims = {normalized_claim(dataset.queries[qid]) for qid in test_qids}
    report = {
        "dataset": "BEIR SciFact", "corpus_size": len(dataset.docs), "train_query_count": len(train_qids), "test_query_count": len(test_qids),
        "metrics": summary, "recommended_method": recommended_method, "recommendation_basis": "Frozen validation comparison; not selected using test metrics.",
        "candidate_ceiling": {key: float(np.mean([row[key] for row in ceilings])) for key in ceilings[0]},
        "paired_bootstrap": comparisons, "latency": {method: _latency(values) for method, values in latency.items()},
        "latency_protocol": "Sequential single-query warm-corpus retrieval, no network, fixed CPU threads. Learned latency is measured RRF retrieval plus reranking; model/index build excluded. No encoder-result cache; RRF/dense queries include encoding. Methods run in a fixed order, so timing is descriptive, not a controlled performance comparison.",
        "ranker_training": ranker.metadata, "ablation_training": ablation.metadata,
        "retrieval": index.metadata, "dataset_manifest": getattr(dataset, "metadata", {}), "frozen_models": frozen, "error_count": len(errors),
        "split_audit": {"official_train_test_query_id_overlap": len(set(train_qids) & set(test_qids)),
                        "official_train_test_normalized_claim_overlap": len(train_claims & test_claims),
                        "note": "Official test partition preserved. Shared corpus documents are allowed in retrieval; normalized-claim overlap is reported as an evaluation limitation when nonzero."},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "lightgbm": lightgbm.__version__, "threads": threads},
        "source_sha256": {path.name: _file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "evaluation_seconds": evaluation_seconds, "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "SciFact is a small, English scientific-evidence retrieval benchmark; results do not establish performance on private company documents or Chinese queries.",
            "The dense encoder is pretrained and frozen. This experiment trains LambdaMART only; it does not fine-tune an LLM or train a cross-encoder.",
            "Public encoder pretraining data can overlap public benchmarks; no claim of contamination-free neural evaluation is made.",
            "Unjudged documents are treated as nonrelevant; evidence judgments can be incomplete.",
            "The official corpus is available during indexing/training; this is a transductive document retrieval benchmark.",
            "Bootstrap intervals resample queries, not related claim groups, and do not adjust for multiple comparisons.",
            "No-dense ablation removes dense and RRF features but keeps the same hybrid candidate pool; it does not ablate dense candidate generation.",
        ],
    }
    run_dir = output_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    for method, lines in runs.items():
        (run_dir / f"{method}.trec").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    _write_json(output_dir / "errors.json", errors)
    _write_json(output_dir / "report.json", report)
    LOGGER.info("Benchmark artifacts saved to %s; total %.1f seconds", output_dir, report["elapsed_seconds"])
    return report
