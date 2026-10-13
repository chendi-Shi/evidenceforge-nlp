"""Retrieval metrics and paired uncertainty estimates, independent of model code."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

METRIC_NAMES = ("ndcg@10", "mrr@10", "recall@10", "recall@50")


def _unique_ids(ranking: Sequence[str | dict]) -> list[str]:
    """A duplicate must never receive relevance credit twice."""
    return list(dict.fromkeys(str(x["doc_id"] if isinstance(x, dict) else x) for x in ranking))


def query_metrics(ranking: Sequence[str | dict], qrels: Mapping[str, int]) -> dict[str, float]:
    ids = _unique_ids(ranking)
    relevant = {str(k): int(v) for k, v in qrels.items() if int(v) > 0}
    ideal = sorted(relevant.values(), reverse=True)[:10]
    # trec_eval / BEIR ndcg_cut uses the original relevance grades as gains.
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    dcg = sum(relevant.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ids[:10]))
    denominator = len(relevant)
    return {
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "mrr@10": next((1.0 / (i + 1) for i, doc in enumerate(ids[:10]) if doc in relevant), 0.0),
        "recall@10": sum(doc in relevant for doc in ids[:10]) / denominator if denominator else 0.0,
        "recall@50": sum(doc in relevant for doc in ids[:50]) / denominator if denominator else 0.0,
    }


def macro_metrics(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        raise ValueError("Cannot evaluate an empty query set")
    return {name: float(np.mean([row[name] for row in per_query])) for name in METRIC_NAMES}


def candidate_ceiling(candidates: Sequence[str | dict], qrels: Mapping[str, int]) -> dict[str, float]:
    """Oracle reordering of retrieved candidates; never used during inference."""
    ids = _unique_ids(candidates)
    oracle = sorted(ids, key=lambda doc: (-int(qrels.get(doc, 0)), doc))
    metrics = query_metrics(oracle, qrels)
    return {"oracle_ndcg@10": metrics["ndcg@10"], "candidate_recall@50": metrics["recall@50"]}


def paired_bootstrap(
    treatment: Sequence[float], reference: Sequence[float], *, seed: int = 42, samples: int = 2000
) -> dict[str, float | int]:
    """Percentile CI for a macro-metric difference using matched query resampling.

    This is a query-level (not claim-family-clustered) interval and does not correct
    for multiple comparisons. Both limitations are included in benchmark reports.
    """
    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Bootstrap requires nonempty, equally sized paired vectors")
    if samples < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Bootstrap inputs must be finite and samples positive")
    delta = a - b
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    # Bounded memory even if the benchmark is expanded beyond SciFact.
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        estimates[start : start + count] = delta[rng.integers(0, len(delta), size=(count, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"delta": float(delta.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "samples": samples, "seed": seed}
