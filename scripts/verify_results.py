"""Audit saved benchmark numbers from TREC rankings and qrels, without model inference.

This script intentionally uses only the standard library and independently
implements the metric arithmetic, rather than importing evidenceforge.evaluation.
Run: python scripts/verify_results.py --data-dir data/scifact --results-dir results/scifact
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


METRICS = ("ndcg@10", "mrr@10", "recall@10", "recall@50")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    judgments: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid qrels columns: {path}")
        for line, row in enumerate(reader, 2):
            qid, doc_id = row["query-id"], row["corpus-id"]
            relevance = int(row["score"])
            if not qid or not doc_id or relevance < 0:
                raise ValueError(f"Invalid qrel at {path}:{line}")
            group = judgments.setdefault(qid, {})
            if doc_id in group:
                raise ValueError(f"Duplicate qrel at {path}:{line}: {qid}/{doc_id}")
            group[doc_id] = relevance
    return judgments


def read_trec(path: Path, query_ids: set[str], document_ids: set[str]) -> dict[str, list[str]]:
    """Use explicit ranks, including when scores tie or file lines are shuffled."""
    groups: dict[str, dict[int, tuple[str, float]]] = {}
    seen: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6 or fields[1] != "Q0":
                raise ValueError(f"Invalid six-column TREC row at {path}:{line_number}")
            qid, _, doc_id, rank_text, score_text, _ = fields
            rank, score = int(rank_text), float(score_text)
            if qid not in query_ids or doc_id not in document_ids:
                raise ValueError(f"Unknown query/document at {path}:{line_number}: {qid}/{doc_id}")
            if rank < 1 or not math.isfinite(score):
                raise ValueError(f"Invalid rank/score at {path}:{line_number}")
            group, docs = groups.setdefault(qid, {}), seen.setdefault(qid, set())
            if rank in group:
                raise ValueError(f"Duplicate rank for query {qid} at {path}:{line_number}")
            if doc_id in docs:
                raise ValueError(f"Duplicate document for query {qid} at {path}:{line_number}")
            group[rank] = (doc_id, score)
            docs.add(doc_id)
    rankings = {}
    for qid, ranks in groups.items():
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError(f"Noncontiguous ranks for query {qid} in {path}")
        ordered = [ranks[rank] for rank in sorted(ranks)]
        if any(ordered[i][1] < ordered[i + 1][1] for i in range(len(ordered) - 1)):
            raise ValueError(f"Scores increase with rank for query {qid} in {path}")
        rankings[qid] = [doc_id for doc_id, _ in ordered]
    return rankings


def calculate_metrics(ranking: list[str], qrels: dict[str, int]) -> dict[str, float]:
    relevant = {doc: rel for doc, rel in qrels.items() if rel > 0}
    ideal = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(sorted(relevant.values(), reverse=True)[:10], 1))
    observed = sum((2**relevant.get(doc, 0) - 1) / math.log2(rank + 1) for rank, doc in enumerate(ranking[:10], 1))
    first = next((rank for rank, doc in enumerate(ranking[:10], 1) if doc in relevant), None)
    return {
        "ndcg@10": observed / ideal if ideal else 0.0,
        "mrr@10": 1.0 / first if first is not None else 0.0,
        "recall@10": len(set(ranking[:10]) & relevant.keys()) / len(relevant) if relevant else 0.0,
        "recall@50": len(set(ranking[:50]) & relevant.keys()) / len(relevant) if relevant else 0.0,
    }


def verify_results(data_dir: Path, results_dir: Path, *, tolerance: float = 1e-10) -> dict:
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Tolerance must be finite and nonnegative")
    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    document_ids = set()
    with (data_dir / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc_id = str(json.loads(line)["_id"])
            if doc_id in document_ids:
                raise ValueError(f"Duplicate corpus document: {doc_id}")
            document_ids.add(doc_id)
    all_qrels = read_qrels(data_dir / "qrels" / "test.tsv")
    qrels = {qid: rels for qid, rels in all_qrels.items() if any(rel > 0 for rel in rels.values())}
    if not qrels:
        raise ValueError("No positive test judgments")
    if any(doc not in document_ids for rels in qrels.values() for doc in rels):
        raise ValueError("Test qrels reference a missing corpus document")
    discrepancies, verified, rankings_by_method = [], {}, {}

    def compare(field: str, expected, actual) -> None:
        if not isinstance(expected, (int, float)) or not math.isfinite(expected) or not math.isclose(expected, actual, rel_tol=0.0, abs_tol=tolerance):
            discrepancies.append({"field": field, "reported": expected, "recomputed": actual})

    compare("corpus_size", report.get("corpus_size"), len(document_ids))
    compare("test_query_count", report.get("test_query_count"), len(qrels))
    compare("train_query_count", report.get("train_query_count"), len(read_qrels(data_dir / "qrels" / "train.tsv")))
    reported_methods = report.get("metrics")
    if not isinstance(reported_methods, dict) or not reported_methods:
        raise ValueError("Report does not contain per-method aggregate metrics")
    expected_depth = min(int(report.get("ranker_training", {}).get("candidate_pool_size", 50)), len(document_ids))
    for method, reported_metrics in reported_methods.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", method):
            raise ValueError(f"Invalid run method name: {method!r}")
        rankings = read_trec(results_dir / "runs" / f"{method}.trec", set(qrels), document_ids)
        rankings_by_method[method] = rankings
        if any(len(ranking) > expected_depth for ranking in rankings.values()):
            raise ValueError(f"Run {method} exceeds its documented candidate depth")
        if method in {"dense", "rrf", "learned", "learned_no_dense"}:
            for qid in qrels:
                if len(rankings.get(qid, [])) != expected_depth:
                    raise ValueError(f"Run {method} has missing/truncated candidates for query {qid}")
        per_query = [calculate_metrics(rankings.get(qid, []), rels) for qid, rels in sorted(qrels.items())]
        verified[method] = {metric: math.fsum(row[metric] for row in per_query) / len(per_query) for metric in METRICS}
        for metric in METRICS:
            compare(f"metrics.{method}.{metric}", reported_metrics.get(metric), verified[method][metric])
    ceilings = None
    if "rrf" in rankings_by_method:
        oracle_metrics = []
        for qid, rels in sorted(qrels.items()):
            pool = rankings_by_method["rrf"].get(qid, [])
            oracle_metrics.append(calculate_metrics(sorted(pool, key=lambda doc: (-rels.get(doc, 0), doc)), rels))
        ceilings = {
            "oracle_ndcg@10": math.fsum(row["ndcg@10"] for row in oracle_metrics) / len(oracle_metrics),
            "candidate_recall@50": math.fsum(row["recall@50"] for row in oracle_metrics) / len(oracle_metrics),
        }
        for metric, value in ceilings.items():
            compare(f"candidate_ceiling.{metric}", report.get("candidate_ceiling", {}).get(metric), value)
    return {"ok": not discrepancies, "test_query_count": len(qrels), "corpus_size": len(document_ids),
            "tolerance": tolerance, "metrics": verified, "candidate_ceiling": ceilings, "discrepancies": discrepancies,
            "scope": "Independent aggregate relevance metrics and candidate ceiling from explicit TREC ranks and actual qrels; does not rerun inference or verify timing/bootstrap intervals."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/scifact"))
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    try:
        result = verify_results(args.data_dir, args.results_dir, tolerance=args.tolerance)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
