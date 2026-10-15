"""Describe post-hoc v3 query extremes from audited, saved rankings (no models).

Outputs IDs, scores and judged-positive ranks, never source query/document text.
Default is stdout only; --output exclusively creates a new file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from verify_results import calculate_metrics, read_qrels, read_trec

SEEDS = ("17", "42", "73")
COMPLETE = "student evaluation complete; no Qwen target-corpus score implied"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def jsonl_ids(path):
    ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                key = str(json.loads(line)["_id"])
                if not key or key in ids:
                    raise ValueError("Empty or duplicate data ID")
                ids.add(key)
    return ids


def select_extremes(differences, count):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Case count must be a positive integer")
    if not differences or any(not math.isfinite(value) for value in differences.values()):
        raise ValueError("Query differences must be nonempty and finite")
    gains = sorted((qid for qid, value in differences.items() if value > 1e-12),
                   key=lambda qid: (-differences[qid], qid))[:count]
    regressions = sorted((qid for qid, value in differences.items() if value < -1e-12),
                         key=lambda qid: (differences[qid], qid))[:count]
    return {"largest_improvements": gains, "largest_regressions": regressions}


def analyze(results_dir, data_dir, *, case_count=2):
    results, data = Path(results_dir), Path(data_dir)
    # Check completeness before accessing target data. This is descriptive analysis
    # after the separate audit, not a substitute for the full weight/training audit.
    report = read_json(results / "report.json")
    audit = read_json(results / "independent_audit.json")
    if report["status"] != COMPLETE or audit.get("status") != "passed" or not audit.get("ok"):
        raise ValueError("Completed evaluation and passed independent audit are required")
    if audit["spec_sha256"] != report["spec_sha256"]:
        raise ValueError("Audit/report protocol mismatch")
    for arm, seeds in (("base", ("frozen",)), ("kd", SEEDS), ("kd_anchor", SEEDS)):
        if set(report["runs"][arm]) != set(seeds):
            raise ValueError("Missing or unexpected comparison seed")
    inputs = {"report.json": sha256(results / "report.json"),
              "independent_audit.json": sha256(results / "independent_audit.json"), "data": {}, "runs": {}}
    for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        observed = sha256(data / filename)
        if observed != report["dataset"]["file_sha256"][filename]:
            raise ValueError(f"Target data hash mismatch: {filename}")
        inputs["data"][filename] = observed
    documents, queries = jsonl_ids(data / "corpus.jsonl"), jsonl_ids(data / "queries.jsonl")
    qrels = read_qrels(data / "qrels/test.tsv")
    if (set(qrels) != queries or len(queries) != report["dataset"]["queries"]
            or len(documents) != report["dataset"]["documents"]
            or any(not any(rel > 0 for rel in rels.values()) or not set(rels) <= documents
                   for rels in qrels.values())):
        raise ValueError("Target query/document coverage mismatch")
    query_ids = sorted(queries)
    scores, ranks = {}, {}
    for arm, seeds in (("base", ("frozen",)), ("kd", SEEDS), ("kd_anchor", SEEDS)):
        scores[arm], ranks[arm] = {}, {}
        for seed in seeds:
            saved = report["runs"][arm][seed]
            path = results / "scidocs" / arm / seed / "ranking.trec"
            run_hash = sha256(path)
            if run_hash != saved["run_sha256"] or not saved["ignore_identical_ids"]:
                raise ValueError("Ranking hash or self-match policy mismatch")
            rankings = read_trec(path, queries, documents)
            if set(rankings) != queries or any(qid in ranking or len(ranking) != min(
                    50, len(documents) - int(qid in documents)) for qid, ranking in rankings.items()):
                raise ValueError("Ranking coverage, depth or self-match mismatch")
            per_query = {row["query_id"]: row for row in saved["per_query"]}
            if set(per_query) != queries or len(per_query) != len(saved["per_query"]):
                raise ValueError("Missing or duplicate per-query result")
            scores[arm][seed], ranks[arm][seed] = {}, {}
            for qid in query_ids:
                metrics = calculate_metrics(rankings[qid], qrels[qid])
                for metric, value in metrics.items():
                    reported = per_query[qid][metric]
                    if not isinstance(reported, (int, float)) or not math.isfinite(reported) or not math.isclose(
                            value, reported, abs_tol=1e-10, rel_tol=0):
                        raise ValueError("TREC/per-query metric mismatch")
                scores[arm][seed][qid] = metrics["ndcg@10"]
                ranks[arm][seed][qid] = {doc: index for index, doc in enumerate(rankings[qid], 1)}
            inputs["runs"][f"scidocs/{arm}/{seed}/ranking.trec"] = run_hash

    def case(qid, differences):
        values = {arm: {seed: rows[qid] for seed, rows in entries.items()} for arm, entries in scores.items()}
        relevant = []
        for doc, grade in sorted(qrels[qid].items()):
            if grade > 0:
                relevant.append({"document_id": doc, "relevance": grade,
                                 "ranks": {arm: {seed: rows[qid].get(doc) for seed, rows in entries.items()}
                                           for arm, entries in ranks.items()}})
        return {"query_id": qid, "mean_paired_ndcg_delta": differences[qid], "ndcg@10": values,
                "judged_positive_document_ranks": relevant}

    comparisons = {}
    for reference in ("kd", "base"):
        differences = {qid: math.fsum(scores["kd_anchor"][seed][qid]
                                     - scores[reference]["frozen" if reference == "base" else seed][qid]
                                     for seed in SEEDS) / len(SEEDS) for qid in query_ids}
        extremes = select_extremes(differences, case_count)
        comparisons[f"kd_anchor_minus_{reference}"] = {
            "mean_query_delta": math.fsum(differences.values()) / len(query_ids),
            "wins": sum(delta > 1e-12 for delta in differences.values()),
            "ties": sum(abs(delta) <= 1e-12 for delta in differences.values()),
            "losses": sum(delta < -1e-12 for delta in differences.values()),
            **{kind: [case(qid, differences) for qid in selected] for kind, selected in extremes.items()}}
    return {"status": "complete", "ok": True, "spec_sha256": report["spec_sha256"],
            "analysis_type": "Post-hoc descriptive query extremes; no retraining, selection or causal inference",
            "selection_rule": "For each comparison, average the paired per-query nDCG@10 differences across seeds 17/42/73; sort positive differences descending and negative differences ascending, ties by query ID; keep case_count from each tail.",
            "case_count_per_tail": case_count, "query_count": len(query_ids), "training_seeds": list(SEEDS),
            "tie_tolerance": 1e-12, "rank_null_meaning": "Not in the saved top-50; exact rank beyond 50 is unknown.",
            "input_sha256": inputs, "comparisons": comparisons,
            "limitations": ["Selected extremes are not representative samples and have no inferential confidence intervals.",
                            "Base is one frozen run reused across paired seeds; observations are not seed x query independent replicates.",
                            "Judged positives are citation-retrieval labels, not exhaustive semantic relevance or verified causal explanations.",
                            "No query or document text is redistributed; use IDs to inspect authorized local data.",
                            "Metric/ranking checks here cover only base, kd and kd_anchor; the separate independent audit covers all 13 runs."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/distillation"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/scidocs"))
    parser.add_argument("--case-count", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Explicit new output file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError("Output already exists; refusing to overwrite")
        result = analyze(args.results_dir, args.data_dir, case_count=args.case_count)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        else:
            print(serialized, end="")
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "failed", "ok": False, "reason": str(error)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
