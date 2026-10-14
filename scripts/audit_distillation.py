"""Independently audit completed v3 artifacts without inference or downloads.

Exit 0 = passed, 2 = pending/incomplete, 1 = inconsistent/invalid. By default
this is read-only; --output explicitly writes the audit, never a model report.
Only standard-library modules and NumPy are used; training code is not imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from verify_results import METRICS, calculate_metrics, read_qrels, read_trec

ARMS = {"ce": [0.0, 0.0], "kd": [1.0, 0.0], "anchor": [0.0, 10.0], "kd_anchor": [1.0, 10.0]}
DOCUMENTS, QUERIES, TRAIN_QUERIES, VALIDATION_QUERIES = 25657, 1000, 12000, 1000
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 2000, 42
CODE_FILES = ("distill_experiment.py", "distill_student.py", "distill_data.py", "distill_eval.py")
SCOPE = (
    "Independent saved-ranking metrics and per-seed paired-query bootstrap; canonical protocol, "
    "source/target identities, 12 declared training traces, checkpoint selection and weight-file SHA256. "
    "Does not rerun inference/training, prove computation outside saved top-50 rankings, authenticate "
    "historical preregistration/freeze times, validate timing, inspect tensor values, or establish model quality "
    "from synthetic tests. Query intervals are conditional on each pair of fitted models, not training-seed "
    "uncertainty or a multiple-comparison correction. Local hashes establish consistency, not authenticity."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def child(directory, filename):
    require(isinstance(filename, str) and filename not in ("", ".", "..")
            and not any(char in filename for char in "/\\:"), "Unsafe artifact filename")
    path = Path(directory) / filename
    require(not path.is_symlink(), "Symlink artifact is not permitted")
    return path


def close(actual, expected, label, tolerance):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), f"{label}: numeric mismatch")


def metrics_equal(actual, expected, label, tolerance):
    for metric in METRICS:
        close(actual[metric], expected[metric], f"{label}/{metric}", tolerance)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def corpus_identity(path):
    """Sort a small ID/offset index, retaining no full corpus in memory."""
    offsets = {}
    with Path(path).open("rb") as handle:
        while True:
            offset, line = handle.tell(), handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["_id"])
            require(key and key not in offsets, "Duplicate/empty corpus ID")
            offsets[key] = offset
        content = hashlib.sha256()
        for key in sorted(offsets):
            handle.seek(offsets[key])
            row = json.loads(handle.readline())
            normalized = {name: str(row.get(name, "")) for name in ("_id", "title", "text")}
            content.update(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode())
            content.update(b"\n")
    return set(offsets), content.hexdigest()


def paired_interval(differences):
    """Independent percentile bootstrap, never pooling seed/query observations."""
    values = np.asarray(differences, dtype=np.float64)
    require(values.ndim == 1 and len(values) and np.isfinite(values).all(), "Invalid bootstrap pairs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        estimates[index] = values[rng.integers(len(values), size=len(values))].mean()
    low, high = np.quantile(estimates, [.025, .975])
    return {"delta": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}


def verify_weights(directory, spec, arm, seed, count):
    metadata = read_json(directory / "metadata.json")
    require(sha256(child(directory, metadata["filename"])) == metadata["suffix_sha256"], "Weight SHA mismatch")
    expected_config = {**spec["student_configs"][arm], "seed": seed}
    require(metadata["config"] == expected_config, "Checkpoint config differs from protocol")
    identity = metadata["identity"]
    for key, expected in (("base_model_sha256", spec["base_model_sha256"]),
                          ("config_sha256", spec["base_config_sha256"]),
                          ("tokenizer_sha256", spec["base_tokenizer_sha256"]),
                          ("max_length", spec["config"]["student_tokens"]),
                          ("threads", spec["config"]["threads"])):
        require(identity[key] == expected, f"Checkpoint identity mismatch: {key}")
    require(metadata["trainable_parameter_count"] == count and count > 0, "Parameter-count mismatch")
    require(metadata["parameter_sha256"] != metadata["initial_parameter_sha256"], "Declared weights never changed")
    return metadata


def verify_training(results, frozen, spec, source_ids, tolerance):
    steps = math.ceil(TRAIN_QUERIES / spec["config"]["batch_size"])
    validation_steps = list(range(spec["config"]["validation_every"], steps + 1,
                                  spec["config"]["validation_every"]))
    if steps not in validation_steps:
        validation_steps.append(steps)
    teacher_hashes, identities, verified = set(), set(), {}
    for key, frozen_report in frozen["runs"].items():
        arm, seed_text = key.split("/")
        seed, directory = int(seed_text), results / "training" / key
        report = read_json(directory / "report.json")
        require(report == frozen_report, f"Frozen training report mismatch: {key}")
        plan = report["plan"]
        order = np.random.default_rng(seed).permutation(len(source_ids))
        require(plan["spec_sha256"] == frozen["spec_sha256"] and plan["arm"] == arm and plan["seed"] == seed
                and plan["order_sha256"] == digest([source_ids[index] for index in order]), "Training plan identity mismatch")
        teacher_hashes.add(plan["teacher_sha256"])
        require(report["progress"] == {"completed_steps": steps, "nonzero_gradient_steps": steps,
                                       "epoch": 0, "row_cursor": TRAIN_QUERIES}, "Incomplete training counters")
        trace = read_json(directory / "trace.json")
        require([row["step"] for row in trace] == list(range(1, steps + 1)), "Training trace missing/duplicate/out-of-order step")
        count = report["trainable_parameters"]
        for row in trace:
            require(row["epoch"] == 0 and row["trainable_parameter_count"] == count and row["dropout_enabled"] is False,
                    "Training trace metadata mismatch")
            for name in ("loss", "ce", "kd", "anchor", "gradient_l2_before_clip"):
                require(isinstance(row[name], (float, int)) and not isinstance(row[name], bool)
                        and math.isfinite(row[name]), "Nonfinite/non-numeric training trace")
            require(row["gradient_l2_before_clip"] > 0, "Training step without nonzero gradient")
        selection = read_json(directory / "selection.json")
        require(selection == report["selection"], "Selection/report mismatch")
        require([row["step"] for row in selection["validations"]] == validation_steps, "Missing validation checkpoint")
        best = max(selection["validations"], key=lambda row: row["metrics"]["ndcg@10"])
        for row in selection["validations"]:
            for metric in METRICS:
                require(math.isfinite(row["metrics"][metric]) and 0 <= row["metrics"][metric] <= 1 + tolerance,
                        "Invalid source validation metric")
        require(selection["best_step"] == best["step"], "Selected checkpoint is not earliest source-validation maximum")
        close(selection["best_ndcg"], best["metrics"]["ndcg@10"], "Selected validation score", tolerance)
        require(sha256(directory / "selected/metadata.json") == report["selected_metadata_sha256"], "Selected metadata SHA mismatch")
        selected = verify_weights(directory / "selected", spec, arm, seed, count)
        final = verify_weights(directory / "final", spec, arm, seed, count)
        require(final == report["final_suffix"], "Final checkpoint/report mismatch")
        identities.add(digest(selected["identity"]))
        verified[key] = {"steps": steps, "selected_step": selection["best_step"],
                         "trace_sha256": sha256(directory / "trace.json"), "selected_weight_sha256": selected["suffix_sha256"]}
    require(len(identities) == 1 and len(teacher_hashes) == 1, "Training runs have different teacher/base identities")
    return verified


def verify_run(directory, result, identity, queries, documents, qrels, tolerance):
    require(result["identity"] == identity, "Evaluation report identity mismatch")
    expected_fields = {"query_count": len(queries), "corpus_size": len(documents), "saved_ranking_depth": 50,
                       "query_ids_sha256": digest(sorted(queries)), "document_ids_sha256": digest(sorted(documents)),
                       "qrels_sha256": digest(qrels), "ignore_identical_ids": True,
                       "tie_break": "document ID lexicographic ascending", "score_function": "normalized dot product (cosine)",
                       "ranking_scope": "all supplied documents"}
    require(all(result[key] == value for key, value in expected_fields.items()), "Ranking metadata mismatch")
    trec = child(directory, result["run_filename"])
    per_query = child(directory, result["per_query_filename"])
    require(sha256(trec) == result["run_sha256"] and sha256(per_query) == result["per_query_sha256"], "Ranking artifact SHA mismatch")
    rankings = read_trec(trec, set(queries), documents)
    require(set(rankings) == set(queries) and all(len(rankings[qid]) == min(50, len(documents) - (qid in documents))
            and qid not in rankings[qid] for qid in queries), "Missing/truncated ranking or forbidden self-match")
    depths = [len(ranking) for ranking in rankings.values()]
    require(result["actual_depth_min"] == min(depths) and result["actual_depth_max"] == max(depths), "Ranking depth mismatch")
    recomputed = {qid: calculate_metrics(rankings[qid], qrels[qid]) for qid in sorted(queries)}
    for label, rows in (("report", result["per_query"]), ("artifact", read_json(per_query))):
        ids = [row["query_id"] for row in rows]
        require(len(ids) == len(set(ids)) and set(ids) == set(queries), "Duplicate/missing per-query ID")
        for row in rows:
            metrics_equal(row, recomputed[row["query_id"]], f"{label}/{row['query_id']}", tolerance)
    aggregate = {metric: math.fsum(row[metric] for row in recomputed.values()) / len(queries) for metric in METRICS}
    metrics_equal(result["metrics"], aggregate, "aggregate", tolerance)
    return {"metrics": aggregate, "per_query": recomputed, "run_sha256": result["run_sha256"]}


def verify_summary(saved, runs, tolerance):
    require(set(saved["arms"]) == set(runs), "Summary arm coverage mismatch")
    for arm, seeds in runs.items():
        summary = saved["arms"][arm]
        require(summary["seed_count"] == len(seeds) and set(summary["per_seed_metrics"]) == set(seeds), "Summary seed coverage mismatch")
        for seed, run in seeds.items():
            metrics_equal(summary["per_seed_metrics"][seed], run["metrics"], f"summary/{arm}/{seed}", tolerance)
        for metric in METRICS:
            values = [run["metrics"][metric] for run in seeds.values()]
            close(summary["metrics"][metric]["mean"], float(np.mean(values)), "Seed mean", tolerance)
            if len(values) == 1:
                require(summary["metrics"][metric]["sample_std"] is None, "Single base is not multiple training seeds")
            else:
                close(summary["metrics"][metric]["sample_std"], float(np.std(values, ddof=1)), "Seed sample SD", tolerance)
    primary = saved["primary"]
    require(primary["treatment"] == "kd_anchor" and primary["reference"] == "kd" and primary["metric"] == "ndcg@10"
            and set(primary["per_seed"]) == set(runs["kd"]), "Primary comparison identity mismatch")
    intervals = {}
    for seed in sorted(runs["kd"]):
        treatment, reference = runs["kd_anchor"][seed]["per_query"], runs["kd"][seed]["per_query"]
        interval = paired_interval([treatment[qid]["ndcg@10"] - reference[qid]["ndcg@10"] for qid in sorted(treatment)])
        row = primary["per_seed"][seed]
        require(row["training_seed"] == seed and row["query_count"] == len(treatment)
                and row["samples"] == BOOTSTRAP_SAMPLES and row["seed"] == BOOTSTRAP_SEED, "Bootstrap identity mismatch")
        for field in ("delta", "ci95_low", "ci95_high"):
            close(row[field], interval[field], f"bootstrap/{seed}/{field}", tolerance)
        intervals[seed] = interval
    deltas = [row["delta"] for row in intervals.values()]
    close(primary["mean_delta"], float(np.mean(deltas)), "Primary seed mean", tolerance)
    close(primary["sample_std_delta"], float(np.std(deltas, ddof=1)), "Primary seed SD", tolerance)
    return intervals


def _verify(results, data, source, repo, tolerance):
    protocol_path = results / "protocol.json"
    if not protocol_path.is_file():
        return {"status": "pending", "missing": [str(protocol_path)]}
    protocol = read_json(protocol_path)
    spec, spec_hash = protocol["spec"], digest(protocol["spec"])
    require(protocol["spec_sha256"] == spec_hash, "Protocol canonical spec hash mismatch")
    config = spec["config"]
    require(spec["version"] == 3 and spec["arms"] == ARMS and config["epochs"] == 1
            and config["train_queries"] == TRAIN_QUERIES and config["validation_queries"] == VALIDATION_QUERIES,
            "Unexpected v3 experiment design")
    seeds = config["seeds"]
    require(len(seeds) == 3 and len(set(seeds)) == 3 and all(type(seed) is int for seed in seeds), "Expected three unique training seeds")
    keys = {f"{arm}/{seed}" for arm in ARMS for seed in seeds}
    required = [results / "frozen_students.json", results / "report.json"]
    for key in keys:
        required.extend(results / "training" / key / name for name in
                        ("report.json", "trace.json", "selection.json", "selected/metadata.json", "final/metadata.json"))
    required.extend(results / "scidocs" / key / "report.json" for key in ["base/frozen", *sorted(keys)])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"status": "pending", "spec_sha256": spec_hash, "missing": missing,
                "note": "Final artifacts incomplete; target data was not opened and no quality result was certified."}
    require(set(spec["implementation_sha256"]) == set(CODE_FILES), "Protocol implementation coverage mismatch")
    for name in CODE_FILES:
        require(sha256(repo / "src/evidenceforge" / name) == spec["implementation_sha256"][name], "Frozen implementation SHA mismatch")
    train, validation = source / "prepared/train.jsonl", source / "prepared/validation.jsonl"
    require(sha256(train) == spec["train_sha256"] and sha256(validation) == spec["validation_sha256"], "Source split SHA mismatch")
    source_ids = [row["query_id"] for row in jsonl(train)]
    validation_ids = [row["query_id"] for row in jsonl(validation)]
    require(len(source_ids) == len(set(source_ids)) == TRAIN_QUERIES
            and len(validation_ids) == len(set(validation_ids)) == VALIDATION_QUERIES
            and not set(source_ids).intersection(validation_ids), "Source query coverage/leakage mismatch")
    frozen, report = read_json(results / "frozen_students.json"), read_json(results / "report.json")
    require(frozen["spec_sha256"] == report["spec_sha256"] == spec_hash and set(frozen["runs"]) == keys, "Frozen protocol/run identity mismatch")
    training = verify_training(results, frozen, spec, source_ids, tolerance)
    # No held-out file is touched until final artifacts and all training runs pass.
    manifest = read_json(data / "manifest.json")
    require(report["dataset"] == manifest and manifest["dataset"] == "BEIR/SCIDOCS"
            and manifest["archive_md5"] == "38121350fc3a4d2f48850f6aff52e4a9", "SCIDOCS manifest identity mismatch")
    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        require(sha256(data / name) == manifest["file_sha256"][name]
                and (data / name).stat().st_size == manifest["file_bytes"][name], "SCIDOCS file SHA/size mismatch")
    documents, corpus_hash = corpus_identity(data / "corpus.jsonl")
    queries = {}
    for row in jsonl(data / "queries.jsonl"):
        key = str(row["_id"])
        require(key and key not in queries, "Duplicate/empty target query ID")
        queries[key] = row["text"]
    qrels = read_qrels(data / "qrels/test.tsv")
    require(len(documents) == DOCUMENTS and len(queries) == QUERIES and set(qrels) == set(queries)
            and all(any(rel > 0 for rel in rels.values()) and set(rels) <= documents for rels in qrels.values()), "Incomplete SCIDOCS data")
    require(manifest["corpus_sha256"] == corpus_hash and manifest["queries_sha256"] == digest(queries)
            and manifest["test_qrels_sha256"] == digest(qrels), "SCIDOCS canonical identity mismatch")
    require(set(report["runs"]) == {"base", *ARMS}, "Final report arm coverage mismatch")
    runs = {}
    for arm in ["base", *ARMS]:
        expected_seeds = {"frozen"} if arm == "base" else {str(seed) for seed in seeds}
        require(set(report["runs"][arm]) == expected_seeds, "Final report seed coverage mismatch")
        runs[arm] = {}
        for seed in sorted(expected_seeds):
            directory = results / "scidocs" / arm / seed
            result = read_json(directory / "report.json")
            require(result == report["runs"][arm][seed], "Final/individual evaluation report mismatch")
            identity = {"spec_sha256": spec_hash, "dataset": corpus_hash, "queries_sha256": digest(queries),
                        "qrels_sha256": digest(qrels), "arm": arm, "seed": seed,
                        "weights": "base" if arm == "base" else frozen["runs"][f"{arm}/{seed}"]["selected_metadata_sha256"]}
            runs[arm][seed] = verify_run(directory, result, identity, queries, documents, qrels, tolerance)
    intervals = verify_summary(report["summary"], runs, tolerance)
    return {"status": "passed", "spec_sha256": spec_hash, "protocol_sha256": sha256(protocol_path),
            "training": training, "corpus_size": len(documents), "test_query_count": len(queries),
            "metrics": {arm: {seed: run["metrics"] for seed, run in entries.items()} for arm, entries in runs.items()},
            "primary_paired_bootstrap": intervals}


def verify(results_dir, data_dir, source_dir, *, repo_root=None, tolerance=1e-10):
    try:
        require(math.isfinite(tolerance) and tolerance >= 0, "Invalid tolerance")
        result = _verify(Path(results_dir), Path(data_dir), Path(source_dir),
                         Path(repo_root) if repo_root else Path(__file__).resolve().parents[1], tolerance)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        result = {"status": "failed", "error": str(error)}
    return {**result, "ok": result["status"] == "passed", "scope": SCOPE, "tolerance": tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/distillation"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/scidocs"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/distillation"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only check (also the default)")
    mode.add_argument("--output", type=Path, help="Explicitly write a separate audit JSON")
    args = parser.parse_args()
    result = verify(args.results_dir, args.data_dir, args.source_dir)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        # Exclusive creation prevents accidentally replacing a frozen report/protocol.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    raise SystemExit({"passed": 0, "failed": 1, "pending": 2}[result["status"]])


if __name__ == "__main__":
    main()
