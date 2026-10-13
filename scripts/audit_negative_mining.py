"""Audit a saved sampling plan without training, relabeling, or test judgments.

Usage: python scripts/audit_negative_mining.py --plan PATH --output PATH
Optional --data-dir reads only qrels/train.tsv to count multi-positive queries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(ordered), "min": ordered[0], "median": statistics.median(ordered),
        "mean": statistics.mean(ordered), "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def _finite(value, label):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}")
    return result


def audit_plan(plan: dict, *, train_qrels: dict | None = None) -> dict:
    """Describe treatment differences; do not infer false-negative ground truth."""
    required_arms = ("random", "retrieved", "filtered")
    arms = {arm: {row["query_id"]: row for row in plan["arms"][arm]} for arm in required_arms}
    query_ids = set(plan["train_query_ids"])
    if any(set(rows) != query_ids for rows in arms.values()):
        raise ValueError("Sampling arms must contain identical training query IDs")
    mining = {row["query_id"]: row for row in plan["mining"]}
    if set(mining) != query_ids:
        raise ValueError("Mining rows do not cover exactly the selected training queries")
    if any(len(plan["arms"][arm]) != len(query_ids) for arm in required_arms) or len(plan["mining"]) != len(query_ids):
        raise ValueError("Duplicate training query records")

    counts = {name: 0 for name in (
        "candidates", "family_flag", "teacher_flag", "both_flags", "family_only_flag",
        "teacher_only_flag", "neither_flag", "missing_teacher_score", "filtered_random_fallback_negatives",
    )}
    candidate_scores, candidate_gaps, per_query = [], [], []
    selected = {
        arm: {"negatives": 0, "outside_mined_pool": 0, "family_flag": 0, "teacher_flag": 0,
              "teacher_scores": [], "score_minus_positive": [], "nonpositive_pool_ranks": [], "rrf_ranks": []}
        for arm in required_arms
    }
    shadow = {
        arm: {"queries_changed_from_retrieved": 0, "replaced_negatives": 0, "unfilled_slots": 0}
        for arm in ("family_only", "teacher_only", "combined")
    }
    changed_queries = replaced_negatives = total_slots = 0
    observed_rrf_ranks = 0
    for qid in sorted(query_ids):
        record = mining[qid]
        positive = record["positive_doc_id"]
        positive_score = _finite(record["positive_teacher_score"], "positive score")
        cutoff = _finite(record["filter_cutoff"], "filter cutoff")
        if positive_score is None or cutoff is None:
            raise ValueError("Positive score and filter cutoff are required")
        candidates, ordered = {}, []
        for position, row in enumerate(record["candidates"], 1):
            doc_id = row["doc_id"]
            if doc_id in candidates or doc_id == positive:
                raise ValueError("Duplicate candidate or chosen positive in the negative pool")
            score = _finite(row.get("teacher_score"), "candidate score")
            family = row["decision"] == "related_training_claim_positive"
            teacher = score is not None and score >= cutoff
            rrf_rank = row.get("rrf_rank", row.get("retrieval_rank"))
            if rrf_rank is not None and (not isinstance(rrf_rank, int) or rrf_rank < 1):
                raise ValueError("An original retrieval rank must be a positive integer")
            observed_rrf_ranks += int(rrf_rank is not None)
            candidates[doc_id] = {
                "score": score, "family": family, "teacher": teacher,
                "pool_rank": position, "rrf_rank": rrf_rank,
            }
            ordered.append(doc_id)
            counts["candidates"] += 1
            counts["family_flag"] += int(family)
            counts["teacher_flag"] += int(teacher)
            counts["both_flags"] += int(family and teacher)
            counts["family_only_flag"] += int(family and not teacher)
            counts["teacher_only_flag"] += int(teacher and not family)
            counts["neither_flag"] += int(not family and not teacher)
            counts["missing_teacher_score"] += int(score is None)
            if score is not None:
                candidate_scores.append(score)
                candidate_gaps.append(score - positive_score)
        retrieved = arms["retrieved"][qid]["negative_doc_ids"]
        filtered = arms["filtered"][qid]["negative_doc_ids"]
        k = len(retrieved)
        total_slots += k
        changed = set(retrieved) != set(filtered)
        replaced = len(set(retrieved) - set(filtered))
        changed_queries += int(changed)
        replaced_negatives += replaced
        counts["filtered_random_fallback_negatives"] += record.get("fallback_count", 0)
        for arm in required_arms:
            example = arms[arm][qid]
            ids = example["negative_doc_ids"]
            if len(ids) != k or len(ids) != len(set(ids)) or positive in ids or example["positive_doc_id"] != positive:
                raise ValueError("Arms must share the positive and the same number of distinct negatives")
            for doc_id in ids:
                stats = selected[arm]
                stats["negatives"] += 1
                if doc_id not in candidates:
                    stats["outside_mined_pool"] += 1
                    continue
                candidate = candidates[doc_id]
                stats["family_flag"] += int(candidate["family"])
                stats["teacher_flag"] += int(candidate["teacher"])
                stats["nonpositive_pool_ranks"].append(candidate["pool_rank"])
                if candidate["rrf_rank"] is not None:
                    stats["rrf_ranks"].append(candidate["rrf_rank"])
                if candidate["score"] is not None:
                    stats["teacher_scores"].append(candidate["score"])
                    stats["score_minus_positive"].append(candidate["score"] - positive_score)
        shadow_ids = {
            "family_only": [doc for doc in ordered if not candidates[doc]["family"]][:k],
            "teacher_only": [doc for doc in ordered if not candidates[doc]["teacher"]][:k],
            "combined": [doc for doc in ordered if not candidates[doc]["family"] and not candidates[doc]["teacher"]][:k],
        }
        for arm, ids in shadow_ids.items():
            shadow[arm]["queries_changed_from_retrieved"] += int(set(ids) != set(retrieved))
            shadow[arm]["replaced_negatives"] += len(set(retrieved) - set(ids))
            shadow[arm]["unfilled_slots"] += k - len(ids)
        per_query.append({
            "query_id": qid, "negative_set_changed": changed, "replaced_negatives": replaced,
            "candidate_count": len(candidates), "fallback_count": record.get("fallback_count", 0),
            "family_flag_count": sum(row["family"] for row in candidates.values()),
            "teacher_flag_count": sum(row["teacher"] for row in candidates.values()),
            "shadow_candidate_ids": shadow_ids,
        })
    for stats in selected.values():
        for key in ("teacher_scores", "score_minus_positive", "nonpositive_pool_ranks", "rrf_ranks"):
            stats[key] = _distribution(stats[key])
    count = len(query_ids)
    for stats in shadow.values():
        stats["query_change_rate"] = stats["queries_changed_from_retrieved"] / count if count else 0
        stats["replacement_rate"] = stats["replaced_negatives"] / total_slots if total_slots else 0
    holdout = {
        "status": "not_run",
        "reason": (
            "The saved plan excludes all known query positives from candidate decisions and contains "
            "only one chosen-positive teacher score. It cannot support an independent masked-positive "
            "audit. Do not substitute the anchor's score or family labels as hidden ground truth."
        ),
        "multi_positive_training_queries": None,
        "other_known_positives_available": None,
        "false_negative_precision": None,
    }
    if train_qrels is not None:
        if not query_ids.issubset(train_qrels):
            raise ValueError("Train qrels do not cover the plan's selected training query IDs")
        positives = {qid: {doc for doc, score in train_qrels[qid].items() if score > 0} for qid in query_ids}
        if any(mining[qid]["positive_doc_id"] not in positives[qid] for qid in query_ids):
            raise ValueError("Plan anchor is not a known training positive")
        holdout["multi_positive_training_queries"] = sum(len(ids) > 1 for ids in positives.values())
        holdout["other_known_positives_available"] = sum(max(0, len(ids) - 1) for ids in positives.values())
    return {
        "schema_version": 1, "training_plan_sha256": plan.get("plan_sha256"), "training_query_count": count,
        "teacher": plan.get("teacher"), "config": plan.get("config"), "decisions": counts,
        "candidate_teacher_scores": _distribution(candidate_scores),
        "candidate_score_minus_positive": _distribution(candidate_gaps),
        "filtered_vs_retrieved": {
            "queries_changed": changed_queries, "query_change_rate": changed_queries / count if count else 0,
            "negative_slots": total_slots, "replaced_negatives": replaced_negatives,
            "replacement_rate": replaced_negatives / total_slots if total_slots else 0,
        },
        "selected_negatives_by_arm": selected, "shadow_rules_without_training": shadow,
        "original_rrf_rank_coverage": observed_rrf_ranks / counts["candidates"] if counts["candidates"] else 0,
        "masked_positive_audit": holdout, "per_query": per_query,
        "limitations": [
            "Family and teacher flags are separate, potentially overlapping heuristic interventions; neither establishes false-negative truth.",
            "Shadow rules change candidate selection only. They are not trained counterfactual models or causal estimates of retrieval quality.",
            "Shadow pools are not randomly refilled. Unfilled slots are reported explicitly, so shortage is not mistaken for teacher filtering.",
            "Nonpositive-pool ranks exclude known query positives and are not original RRF ranks. Original ranks are reported only when saved in the plan.",
            "Scores and ranks for negatives outside the mined pool are missing, not zero. Random-arm score summaries condition on pool overlap and are not representative of all random negatives.",
            "An improvement in retrieval metrics could come from changed hardness or family-label protection; no denoising-precision claim is supported.",
            "Only source train qrels may be read for eligibility counts. No validation/test judgments, model weights, or gradient updates are used.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, help="Optional source dataset directory; read qrels/train.tsv only")
    args = parser.parse_args()
    payload = args.plan.read_bytes()
    plan = json.loads(payload)
    qrels = None
    if args.data_dir:
        qrels = {}
        with (args.data_dir / "qrels/train.tsv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    result = audit_plan(plan, train_qrels=qrels)
    result["training_plan_file_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "training_queries": result["training_query_count"],
                      "replacement_rate": result["filtered_vs_retrieved"]["replacement_rate"],
                      "masked_positive_audit": result["masked_positive_audit"]["status"]}))


if __name__ == "__main__":
    main()
