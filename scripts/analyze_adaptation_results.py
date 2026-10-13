"""Post-hoc neural retrieval analysis from saved rankings; never loads a model.

Primary results remain the frozen full-test protocol. The normalized-claim
exclusion is a sensitivity analysis against the entire official SciFact train
split, including the project's internal validation partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re

from verify_results import METRICS, calculate_metrics, read_qrels, read_trec

ARMS = ("base", "random", "retrieved", "filtered")
PAIRS = (("filtered", "retrieved"), *((arm, "base") for arm in ARMS[1:]))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized_claim(text: str) -> str:
    """Exact rule from evidenceforge.ranker; copied to avoid model dependencies."""
    return " ".join(re.sub(r"\d+", "<num>", token)
                    for token in re.findall(r"\b\w+\b", text.lower(), re.UNICODE))


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["_id"])
            if not key or key in rows or not isinstance(row.get("text"), str):
                raise ValueError(f"Invalid or duplicate JSONL row: {path}/{key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"Empty JSONL file: {path}")
    return rows


def corpus_hash(documents: dict[str, dict]) -> str:
    """Same canonical corpus identity as Dataset.corpus_hash, without importing it."""
    digest = hashlib.sha256()
    for key in sorted(documents):
        row = {field: str(documents[key].get(field, "")) for field in ("_id", "title", "text")}
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate(rows: dict[str, dict], query_ids: list[str]) -> dict[str, float]:
    if not query_ids:
        raise ValueError("Cannot analyze an empty query subset")
    return {metric: math.fsum(rows[qid][metric] for qid in query_ids) / len(query_ids)
            for metric in METRICS}


def paired_bootstrap(treatment: list[float], reference: list[float], *, samples: int = 2000,
                     seed: int = 42) -> dict:
    """Independent bounded-memory percentile bootstrap with linear quantiles."""
    if not treatment or len(treatment) != len(reference) or samples < 1:
        raise ValueError("Bootstrap needs paired nonempty inputs and positive samples")
    if not all(math.isfinite(value) for value in treatment + reference):
        raise ValueError("Bootstrap inputs must be finite")
    differences = [a - b for a, b in zip(treatment, reference)]
    rng = random.Random(seed)
    estimates = sorted(math.fsum(rng.choices(differences, k=len(differences))) / len(differences)
                       for _ in range(samples))

    def quantile(probability):
        position = (samples - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return estimates[lower] + (estimates[upper] - estimates[lower]) * (position - lower)

    return {"delta": math.fsum(differences) / len(differences), "ci95_low": quantile(.025),
            "ci95_high": quantile(.975), "samples": samples, "seed": seed,
            "unit": "paired query", "rng": "stdlib random.Random.choices",
            "note": "Independent resampling: RNG differs from the primary NumPy bootstrap; endpoints can differ."}


def compare(rows: dict[str, dict[str, dict]], query_ids: list[str], *, samples: int, seed: int) -> dict:
    comparisons = {}
    for treatment, reference in PAIRS:
        differences = [rows[treatment][qid]["ndcg@10"] - rows[reference][qid]["ndcg@10"]
                       for qid in query_ids]
        comparisons[f"{treatment}_minus_{reference}"] = {
            **paired_bootstrap([rows[treatment][qid]["ndcg@10"] for qid in query_ids],
                               [rows[reference][qid]["ndcg@10"] for qid in query_ids],
                               samples=samples, seed=seed),
            "wins": sum(value > 1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
            "losses": sum(value < -1e-12 for value in differences), "tie_tolerance": 1e-12}
    return comparisons


def normalized_overlap(queries: dict[str, dict], official_train_qrels: dict, test_query_ids: list[str]) -> dict:
    train_ids = sorted(qid for qid, rels in official_train_qrels.items() if any(rel > 0 for rel in rels.values()))
    signatures: dict[str, list[str]] = {}
    for qid in train_ids:
        signatures.setdefault(normalized_claim(queries[qid]["text"]), []).append(qid)
    excluded = []
    retained = []
    for qid in test_query_ids:
        signature = normalized_claim(queries[qid]["text"])
        matches = signatures.get(signature, [])
        if matches:
            excluded.append({"query_id": qid, "query": queries[qid]["text"],
                             "normalized_claim": signature, "matching_official_train_query_ids": matches})
        else:
            retained.append(qid)
    return {"official_train_query_count": len(train_ids), "excluded": excluded,
            "excluded_count": len(excluded), "retained_query_ids": retained,
            "retained_count": len(retained)}


def close_enough(expected, actual, context: str) -> None:
    if not isinstance(expected, (int, float)) or not math.isfinite(expected) or not math.isclose(
            expected, actual, abs_tol=1e-10, rel_tol=0):
        raise ValueError(f"Report/ranking mismatch: {context}")


def analyze_dataset(name: str, data_dir: Path, results_dir: Path, saved: dict, *, samples: int,
                    seed: int, case_count: int) -> dict:
    documents = read_jsonl(data_dir / "corpus.jsonl")
    if corpus_hash(documents) != saved["corpus_sha256"]:
        raise ValueError(f"Corpus identity mismatch: {name}")
    queries = read_jsonl(data_dir / "queries.jsonl")
    judgments = {qid: rels for qid, rels in read_qrels(data_dir / "qrels" / "test.tsv").items()
                 if any(rel > 0 for rel in rels.values())}
    query_ids = sorted(judgments)
    if not query_ids or any(qid not in queries for qid in judgments):
        raise ValueError(f"Missing positive test queries: {name}")
    if any(doc not in documents for rels in judgments.values() for doc in rels):
        raise ValueError(f"Missing qrel document: {name}")
    if set(saved["methods"]) != set(ARMS):
        raise ValueError(f"Expected all four methods: {name}")
    rows, rankings, runs, metrics = {}, {}, {}, {}
    for arm in ARMS:
        report = saved["methods"][arm]
        path = results_dir / name / "runs" / f"{arm}.trec"
        runs[arm] = sha256(path)
        if runs[arm] != report["run_sha256"]:
            raise ValueError(f"Run byte hash mismatch: {name}/{arm}")
        rankings[arm] = read_trec(path, set(query_ids), set(documents))
        if set(rankings[arm]) != set(query_ids) or any(
                len(ranking) != min(50, len(documents)) for ranking in rankings[arm].values()):
            raise ValueError(f"Missing or truncated test ranking: {name}/{arm}")
        rows[arm] = {qid: calculate_metrics(rankings[arm][qid], judgments[qid]) for qid in query_ids}
        reported_rows = {row["query_id"]: row for row in report["per_query"]}
        if len(reported_rows) != len(report["per_query"]) or set(reported_rows) != set(query_ids):
            raise ValueError(f"Duplicate or missing report query: {name}/{arm}")
        close_enough(report["query_count"], len(query_ids), f"{name}/{arm}/query_count")
        close_enough(report["corpus_size"], len(documents), f"{name}/{arm}/corpus_size")
        metrics[arm] = aggregate(rows[arm], query_ids)
        for metric in METRICS:
            close_enough(report["metrics"][metric], metrics[arm][metric], f"{name}/{arm}/{metric}")
            for qid in query_ids:
                close_enough(reported_rows[qid][metric], rows[arm][qid][metric], f"{name}/{arm}/{qid}/{metric}")

    def example(qid: str) -> dict:
        def listed(doc_id: str) -> dict:
            return {"doc_id": doc_id, "title": documents[doc_id].get("title", ""),
                    "judged_relevance": judgments[qid].get(doc_id, 0)}
        return {"query_id": qid, "query": queries[qid]["text"],
                "ndcg_delta_filtered_minus_retrieved": rows["filtered"][qid]["ndcg@10"] - rows["retrieved"][qid]["ndcg@10"],
                "metrics": {arm: rows[arm][qid] for arm in ARMS},
                "positive_judgments": [listed(doc) for doc, grade in sorted(judgments[qid].items()) if grade > 0],
                "top10": {arm: [listed(doc) for doc in rankings[arm][qid][:10]] for arm in ("retrieved", "filtered")}}

    deltas = {qid: rows["filtered"][qid]["ndcg@10"] - rows["retrieved"][qid]["ndcg@10"] for qid in query_ids}
    gains = sorted((qid for qid in query_ids if deltas[qid] > 1e-12), key=lambda qid: (-deltas[qid], qid))[:case_count]
    losses = sorted((qid for qid in query_ids if deltas[qid] < -1e-12), key=lambda qid: (deltas[qid], qid))[:case_count]
    result = {"query_count": len(query_ids), "corpus_size": len(documents), "metrics": metrics,
              "comparisons_ndcg@10": compare(rows, query_ids, samples=samples, seed=seed),
              "largest_gains": [example(qid) for qid in gains], "largest_regressions": [example(qid) for qid in losses],
              "case_selection": "Post-hoc extremes of filtered minus retrieved nDCG@10; descriptive, not representative.",
              "input_sha256": {"corpus.jsonl": sha256(data_dir / "corpus.jsonl"),
                               "queries.jsonl": sha256(data_dir / "queries.jsonl"),
                               "qrels/test.tsv": sha256(data_dir / "qrels" / "test.tsv"), "runs": runs}}
    if name == "scifact":
        train_path = data_dir / "qrels" / "train.tsv"
        official_train = read_qrels(train_path)
        if any(qid not in queries for qid in official_train):
            raise ValueError("Official train qrels reference missing query text")
        overlap = normalized_overlap(queries, official_train, query_ids)
        retained = overlap["retained_query_ids"]
        result["post_hoc_normalized_overlap_sensitivity"] = {
            "status": "post-hoc sensitivity, not a replacement for the frozen primary protocol",
            "exclusion_reference": "Entire official train split, including the internal validation partition; not only the 647 optimization queries.",
            "normalization": "Unicode word tokens, lowercase, each digit sequence replaced by <num>; same rule as evidenceforge.ranker.normalized_claim.",
            "scope": "Only exact equality after normalization; shared evidence documents and semantic paraphrases are not excluded.",
            **overlap, "metrics": {arm: aggregate(rows[arm], retained) for arm in ARMS},
            "comparisons_ndcg@10": compare(rows, retained, samples=samples, seed=seed)}
        result["input_sha256"]["qrels/train.tsv"] = sha256(train_path)
    return result


def analyze(results_dir: Path, source_dir: Path, transfer_dir: Path, *, samples: int = 2000,
            seed: int = 42, case_count: int = 5) -> dict:
    if samples < 1 or case_count < 0:
        raise ValueError("Samples must be positive and case count nonnegative")
    path = results_dir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if set(report["datasets"]) != {"scifact", "nfcorpus"}:
        raise ValueError("Both complete datasets must be present; evaluation may still be running")
    return {"analysis": "Independent post-hoc analysis from complete saved TREC rankings and qrels",
            "report_sha256": sha256(path), "protocol_sha256": report["protocol_sha256"],
            "datasets": {name: analyze_dataset(name, directory, results_dir, report["datasets"][name],
                                               samples=samples, seed=seed, case_count=case_count)
                         for name, directory in (("scifact", source_dir), ("nfcorpus", transfer_dir))},
            "limitations": ["One optimization seed; paired query bootstrap does not estimate training variance.",
                            "Intervals are not claim-family-clustered or corrected for multiple comparisons.",
                            "Post-hoc exclusion and extreme examples do not change the frozen primary evaluation.",
                            "Unjudged documents receive zero relevance for metrics; this does not prove non-relevance.",
                            "Filtering uses a false-negative risk proxy, not verified false-negative labels; changes also alter negative hardness.",
                            "SciFact and NFCorpus overlap in biomedical subject matter; transfer is not across unrelated domains.",
                            "This script checks corpus identity, run hashes and relevance metrics, not adapter training or model inference."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/adaptation"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--transfer-dir", type=Path, default=Path("data/nfcorpus"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-count", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.results_dir, args.source_dir, args.transfer_dir,
                         samples=args.samples, seed=args.seed, case_count=args.case_count)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        resolved = args.output.resolve()
        protected_roots = (args.source_dir.resolve(), args.transfer_dir.resolve(),
                           *((args.results_dir / name).resolve() for name in ("training", "scifact", "nfcorpus")))
        protected_names = {"report.json", "protocol.json", "protocol.initial-draft.json", "protocol_amendment.json", "frozen_adapters.json"}
        if any(resolved.is_relative_to(root) for root in protected_roots) or (
                resolved.parent == args.results_dir.resolve() and resolved.name in protected_names):
            parser.exit(1, "Analysis output must not overwrite data, model training artifacts, or the frozen experiment.\n")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
