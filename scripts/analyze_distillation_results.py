"""Model-free secondary analysis of completed, frozen v3 JSON results.

Uses only the Python standard library. Does not open target data, TREC files,
weights, or model code; run audit_distillation.py separately for independent
metric/weight verification. With no --output, the CLI writes only to stdout.
Missing result files produce pending (exit 2), never synthetic measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ARMS = ("ce", "kd", "anchor", "kd_anchor")
SEEDS = ("17", "42", "73")
METRICS = ("ndcg@10", "mrr@10", "recall@10", "recall@50")
DRIFT_KEYS = ("query_mean_cosine_distance_to_base", "document_mean_cosine_distance_to_base")
PAIRING_FIELDS = (
    "query_ids_sha256", "document_ids_sha256", "qrels_sha256", "query_count", "corpus_size",
    "ignore_identical_ids", "score_function", "tie_break", "saved_ranking_depth",
)
COMPLETE_STATUS = "student evaluation complete; no Qwen target-corpus score implied"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def checked_metrics(metrics):
    result = {name: finite(metrics[name], name) for name in METRICS}
    if any(not 0 <= value <= 1 + 1e-12 for value in result.values()):
        raise ValueError("Retrieval metrics must be in [0, 1]")
    return result


def checked_rows(result):
    rows = {}
    for row in result["per_query"]:
        qid = row["query_id"]
        if not isinstance(qid, str) or not qid or qid in rows:
            raise ValueError("Duplicate or invalid query ID")
        rows[qid] = checked_metrics(row)
    if not rows or len(rows) != result["query_count"]:
        raise ValueError("Per-query rows and query_count disagree")
    if digest(sorted(rows)) != result["query_ids_sha256"]:
        raise ValueError("Per-query rows and query identity disagree")
    metrics = checked_metrics(result["metrics"])
    for name in METRICS:
        if not math.isclose(statistics.fmean(row[name] for row in rows.values()), metrics[name],
                            rel_tol=0, abs_tol=1e-10):
            raise ValueError("Aggregate metrics disagree with per-query rows")
    return rows


def descriptive(values):
    values = list(values)
    return {"seed_count": len(values), "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "interpretation": "Descriptive across training-order seeds; not a confidence interval"}


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def paired_intervals(differences, *, samples=2000, bootstrap_seed=42):
    """Bootstrap matched query contrasts; hold one draw, not a samples x queries matrix.

    Each input vector has the same canonical query order. Within a call, all
    contrasts reuse a draw; no training seeds are combined in any bootstrap.
    Python's random.Random is explicit, so finite-draw endpoints may differ from
    the independent audit or production primary CI, which are separate outputs.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer")
    vectors = {key: [finite(value, "Query contrast") for value in values]
               for key, values in differences.items()}
    lengths = {len(values) for values in vectors.values()}
    if not vectors or len(lengths) != 1 or 0 in lengths:
        raise ValueError("Bootstrap needs nonempty, equally sized paired query contrasts")
    count = lengths.pop()
    rng = random.Random(bootstrap_seed)
    estimates = {key: [] for key in vectors}
    positions = range(count)
    for _ in range(samples):
        selected = rng.choices(positions, k=count)
        for key, values in vectors.items():
            estimates[key].append(math.fsum(values[index] for index in selected) / count)
    result = {}
    for key, values in vectors.items():
        ordered = sorted(estimates[key])
        result[key] = {
            "delta": statistics.fmean(values), "ci95_low": percentile(ordered, .025),
            "ci95_high": percentile(ordered, .975), "query_count": count, "samples": samples,
            "bootstrap_seed": bootstrap_seed, "generator": "Python random.Random.choices",
            "interval": "Paired-query percentile bootstrap, linear quantiles",
            "uncertainty_unit": "Paired query, conditional on the fixed fitted models in this seed",
        }
    return result


def source_selection(report, baseline):
    selection = report["selection"]
    trajectory = []
    last_step = 0
    for row in selection["validations"]:
        step = row["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= last_step:
            raise ValueError("Source validation steps must be positive, unique and increasing")
        last_step = step
        drift = {key: finite(row["drift"][key], "Source drift") for key in DRIFT_KEYS}
        if any(not -1e-6 <= value <= 2 + 1e-6 for value in drift.values()):
            raise ValueError("Source cosine distance must be in [0, 2] within floating tolerance")
        trajectory.append({"step": step, "metrics": checked_metrics(row["metrics"]),
                           "source_drift": drift})
    if not trajectory or last_step > report["progress"]["completed_steps"]:
        raise ValueError("Source validation trajectory is missing or exceeds trained steps")
    # Python max keeps the first record on ties: the predeclared earliest-step rule.
    selected = max(trajectory, key=lambda row: row["metrics"]["ndcg@10"])
    if (selection["best_step"] != selected["step"] or
            not math.isclose(finite(selection["best_ndcg"], "Selected source nDCG"),
                             selected["metrics"]["ndcg@10"], rel_tol=0, abs_tol=1e-10)):
        raise ValueError("Selected checkpoint disagrees with the source-only earliest-best rule")
    return {
        "selected_step": selected["step"], "selected_metrics": selected["metrics"],
        "selected_minus_source_base": {
            name: selected["metrics"][name] - baseline["metrics"][name] for name in METRICS},
        "selected_source_drift": selected["source_drift"], "validation_trajectory": trajectory,
        "selection_rule": "Highest recorded source-validation nDCG@10; earliest step wins ties",
        "scope": "Fixed shared source validation-document pool; not full MS MARCO evaluation",
    }


def analyze(results_dir, *, samples=2000, bootstrap_seed=42):
    """Analyze the complete fixed 4-arm x 3-seed experiment, or return pending.

    Malformed completed inputs raise ValueError. Completion refers to this
    secondary analysis only; it does not replace independent TREC/weight audit.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer")
    root = Path(results_dir)
    required = [Path("report.json"), Path("source_validation_base.json")]
    required += [Path("training") / arm / seed / "report.json" for arm in ARMS for seed in SEEDS]
    missing = [path.as_posix() for path in required if not (root / path).is_file()]
    if missing:
        return {"status": "pending", "ok": False, "missing_files": missing,
                "reason": "Complete target and source/training reports are required; no results inferred"}
    inputs, hashes = {}, {}
    for relative in required:
        raw = (root / relative).read_bytes()
        inputs[relative.as_posix()] = json.loads(raw)
        hashes[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    try:
        return _analyze(inputs, hashes, samples=samples, bootstrap_seed=bootstrap_seed)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Malformed completed result schema: {error}") from error


def _analyze(inputs, hashes, *, samples, bootstrap_seed):
    report, baseline = inputs["report.json"], inputs["source_validation_base.json"]
    spec = report["spec_sha256"]
    if not isinstance(spec, str) or not spec or baseline["spec_sha256"] != spec:
        raise ValueError("Source baseline and target report protocol identities differ")
    if report["status"] != COMPLETE_STATUS:
        raise ValueError("Root report does not declare the completed student evaluation")
    checked_rows(baseline)
    runs = report["runs"]
    if set(runs) != {"base", *ARMS} or set(runs["base"]) != {"frozen"}:
        raise ValueError("Complete target report requires base/frozen and all four arms")
    if any(set(runs[arm]) != set(SEEDS) for arm in ARMS):
        raise ValueError("All four arms must have exactly matching training seeds 17, 42, 73")
    base = runs["base"]["frozen"]
    base_rows = checked_rows(base)
    query_ids = sorted(base_rows)
    pairing = {field: base[field] for field in PAIRING_FIELDS}
    checked = {}
    for arm, entries in runs.items():
        for seed, result in entries.items():
            rows = checked_rows(result)
            if set(rows) != set(base_rows) or any(result[field] != value for field, value in pairing.items()):
                raise ValueError("Target runs must use identical query/document IDs, judgments and policies")
            identity = result["identity"]
            if (identity["spec_sha256"] != spec or identity["arm"] != arm or identity["seed"] != seed):
                raise ValueError("Target arm/seed/protocol identity mismatch")
            if any(identity[field] != base["identity"][field]
                   for field in ("dataset", "queries_sha256", "qrels_sha256")):
                raise ValueError("Target dataset/query/qrels identity mismatch")
            if arm == "base" and identity["weights"] != "base":
                raise ValueError("Frozen base weights identity mismatch")
            checked[arm, seed] = rows
    per_seed = {}
    for seed in SEEDS:
        arms, orders = {}, set()
        for arm in ARMS:
            training = inputs[f"training/{arm}/{seed}/report.json"]
            plan = training["plan"]
            if plan["spec_sha256"] != spec or plan["arm"] != arm or str(plan["seed"]) != seed:
                raise ValueError("Training arm/seed/protocol identity mismatch")
            orders.add(plan["order_sha256"])
            if runs[arm][seed]["identity"]["weights"] != training["selected_metadata_sha256"]:
                raise ValueError("Target report and selected training checkpoint identities differ")
            metrics = checked_metrics(runs[arm][seed]["metrics"])
            arms[arm] = {
                "target_metrics": metrics,
                "target_minus_base": {name: metrics[name] - base["metrics"][name] for name in METRICS},
                "source_validation": source_selection(training, baseline),
            }
        if len(orders) != 1:
            raise ValueError("All arms within a seed must use the same training row order")
        differences = {
            f"{arm}_minus_base": [checked[arm, seed][qid]["ndcg@10"] - base_rows[qid]["ndcg@10"]
                                 for qid in query_ids] for arm in ARMS}
        differences["interaction"] = [
            checked["kd_anchor", seed][qid]["ndcg@10"] - checked["kd", seed][qid]["ndcg@10"]
            - checked["anchor", seed][qid]["ndcg@10"] + checked["ce", seed][qid]["ndcg@10"]
            for qid in query_ids]
        intervals = paired_intervals(differences, samples=samples, bootstrap_seed=bootstrap_seed)
        for arm in ARMS:
            arms[arm]["target_minus_base_ndcg@10_interval"] = intervals[f"{arm}_minus_base"]
        per_seed[seed] = {
            "training_seed": seed, "arms": arms,
            "interaction": {
                "formula": "(kd_anchor - kd) - (anchor - ce)",
                "metrics": {name: arms["kd_anchor"]["target_metrics"][name] - arms["kd"]["target_metrics"][name]
                            - arms["anchor"]["target_metrics"][name] + arms["ce"]["target_metrics"][name]
                            for name in METRICS},
                "ndcg@10_interval": intervals["interaction"], "role": "Predeclared secondary contrast",
            },
        }
    arm_summaries = {}
    for arm in ARMS:
        rows = [per_seed[seed]["arms"][arm] for seed in SEEDS]
        arm_summaries[arm] = {
            "target_metrics": {name: descriptive(row["target_metrics"][name] for row in rows) for name in METRICS},
            "target_minus_base": {name: descriptive(row["target_minus_base"][name] for row in rows) for name in METRICS},
            "source_selected_minus_base": {
                name: descriptive(row["source_validation"]["selected_minus_source_base"][name] for row in rows)
                for name in METRICS},
            "selected_steps": {seed: per_seed[seed]["arms"][arm]["source_validation"]["selected_step"]
                               for seed in SEEDS},
        }
    return {
        "status": "complete", "ok": True, "spec_sha256": spec,
        "analysis_scope": "Secondary analysis of supplied JSON measurements; not independent metric/weight audit",
        "input_sha256": hashes, "dataset": report["dataset"], "training_seeds": list(SEEDS),
        "target_base_metrics": base["metrics"], "source_base_metrics": baseline["metrics"],
        "target_pairing": pairing, "per_seed": per_seed,
        "descriptive_across_seeds": {"arms": arm_summaries, "interaction": {
            name: descriptive(per_seed[seed]["interaction"]["metrics"][name] for seed in SEEDS)
            for name in METRICS}},
        "notes": [
            "Three seeds vary training batch order; pretrained initialization is shared and dropout is disabled.",
            "No seed x query pooling. Sample SD across three seeds is descriptive, not a confidence interval.",
            "Query intervals condition on the fixed fitted models; they do not measure training variance or correct multiple comparisons.",
            "source_drift is the source-validation query/document cosine distance to the original base. Target drift was not measured.",
            "Different best steps can explain different drift. The full recorded source trajectories permit same-step inspection.",
            "Checkpoints are selected only on source validation; target scores never change selection. Source gains may coexist with target losses.",
            "A positive interaction is a difference of differences, not proof of beating base or of generalization improvement.",
            "The primary kd_anchor-minus-kd result and independent TREC/weight audit remain separate artifacts.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/distillation"))
    parser.add_argument("--output", type=Path, help="New file to create exclusively; otherwise stdout only")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = analyze(args.results_dir, samples=args.samples, bootstrap_seed=args.bootstrap_seed)
    except (ValueError, OSError) as error:
        result = {"status": "failed", "ok": False, "reason": str(error)}
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        except FileExistsError:
            print(json.dumps({"status": "failed", "ok": False,
                              "reason": f"Output already exists; refusing to overwrite: {args.output}"}))
            return 1
        except OSError as error:
            print(json.dumps({"status": "failed", "ok": False, "reason": str(error)}))
            return 1
    else:
        print(serialized, end="")
    return {"complete": 0, "pending": 2, "failed": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
