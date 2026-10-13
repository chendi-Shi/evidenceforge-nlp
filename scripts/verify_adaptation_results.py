"""Independent standard-library audit of neural runs, protocol and adapter bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from verify_results import METRICS, calculate_metrics, read_qrels, read_trec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def verify(results_dir: Path, source_dir: Path, transfer_dir: Path) -> dict:
    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((results_dir / "frozen_adapters.json").read_text(encoding="utf-8"))
    errors, recomputed, per_query = [], {}, {}
    protocol_hash = sha(results_dir / "protocol.json")
    if report["protocol_sha256"] != protocol_hash or frozen["protocol_sha256"] != protocol_hash:
        errors.append("Protocol byte hash mismatch")
    protocol = json.loads((results_dir / "protocol.json").read_text(encoding="utf-8"))
    spec = protocol["spec"]
    if protocol["spec_sha256"] != canonical_digest(spec) or report["spec_sha256"] != protocol["spec_sha256"]:
        errors.append("Protocol spec hash mismatch")
    plan = json.loads((results_dir / "training" / "training_plan.json").read_text(encoding="utf-8"))
    plan_hash = canonical_digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan["plan_sha256"] != plan_hash:
        errors.append("Training plan content hash mismatch")
    config = spec["training"]
    config_hash = canonical_digest(config)
    if plan["config"] != config:
        errors.append("Training plan/protocol config mismatch")
    if plan["corpus_sha256"] != spec["source_corpus_sha256"]:
        errors.append("Training plan/protocol source corpus mismatch")
    train_ids, validation_ids = plan["train_query_ids"], plan["validation_query_ids"]
    if (not train_ids or not validation_ids or len(train_ids) != len(set(train_ids))
            or len(validation_ids) != len(set(validation_ids)) or set(train_ids) & set(validation_ids)):
        errors.append("Training/validation plan IDs are empty, duplicated or overlapping")
    expected_steps = math.ceil(len(train_ids) / config["batch_size"]) * config["epochs"]
    expected_arms = {"random", "retrieved", "filtered"}
    if set(frozen["adapters"]) != expected_arms or set(plan["arms"]) != expected_arms:
        errors.append("Missing/unexpected training arm")
    initial_hashes, base_identities = set(), set()
    positives = {}
    for arm in ("random", "retrieved", "filtered"):
        examples = plan["arms"][arm]
        example_ids = [row["query_id"] for row in examples]
        if len(example_ids) != len(set(example_ids)) or set(example_ids) != set(train_ids):
            errors.append(f"{arm}: training examples do not cover the selected queries exactly")
        positives[arm] = {row["query_id"]: row["positive_doc_id"] for row in examples}
        directory = results_dir / "training" / arm
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        identity, base = metadata["identity"], metadata["base_encoder"]
        if metadata["arm"] != arm or identity["arm"] != arm:
            errors.append(f"{arm}: training arm identity mismatch")
        if metadata["config"] != config or identity["config_sha256"] != config_hash:
            errors.append(f"{arm}: training config differs from protocol")
        if identity["plan_sha256"] != plan_hash:
            errors.append(f"{arm}: training plan identity mismatch")
        if (metadata["training_query_count"] != len(train_ids)
                or metadata["validation_query_count"] != len(validation_ids)):
            errors.append(f"{arm}: training/validation query count differs from plan")
        if identity["initial_adapter_sha256"] != metadata["initial_trainable_sha256"]:
            errors.append(f"{arm}: initialization identity mismatch")
        initial_hashes.add(metadata["initial_trainable_sha256"])
        if identity["base_encoder"] != base:
            errors.append(f"{arm}: base encoder metadata/identity mismatch")
        base_identities.add(canonical_digest(base))
        asset_fields = {"base_model_sha256": "model.safetensors", "config_sha256": "config.json",
                        "tokenizer_sha256": "tokenizer.json"}
        if (base["model_id"] != spec["base_assets"]["model_id"] or base["max_length"] != config["max_length"]
                or any(base[field] != spec["base_assets"]["sha256"][filename] for field, filename in asset_fields.items())):
            errors.append(f"{arm}: base encoder differs from protocol assets or token length")
        if metadata["adapter_sha256"] != sha(directory / "adapter.safetensors") or frozen["adapters"][arm] != metadata["adapter_sha256"]:
            errors.append(f"{arm}: adapter hash mismatch")
        if (metadata["completed_steps"] != expected_steps or metadata["parameter_delta_l2"] <= 0
                or metadata["nonzero_gradient_steps"] <= 0 or metadata["initial_trainable_sha256"] == metadata["final_trainable_sha256"]):
            errors.append(f"{arm}: incomplete or unchanged training")
    if len(initial_hashes) != 1:
        errors.append("Training arms did not share the same initialization")
    if len(base_identities) != 1:
        errors.append("Training arms did not share the same base encoder identity")
    if any(positives[arm] != positives["random"] for arm in ("retrieved", "filtered")):
        errors.append("Training arms did not share the same positive examples")
    if set(report["datasets"]) != {"scifact", "nfcorpus"}:
        errors.append("Missing/unexpected evaluation dataset")
    for name, data_dir in (("scifact", source_dir), ("nfcorpus", transfer_dir)):
        documents = {str(json.loads(line)["_id"]) for line in (data_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
        judgments = {q: rel for q, rel in read_qrels(data_dir / "qrels" / "test.tsv").items() if any(v > 0 for v in rel.values())}
        recomputed[name], per_query[name] = {}, {}
        if set(report["datasets"][name]["methods"]) != {"base", "random", "retrieved", "filtered"}:
            errors.append(f"{name}: missing/unexpected method")
        for arm in ("base", "random", "retrieved", "filtered"):
            saved = report["datasets"][name]["methods"][arm]
            path = results_dir / name / "runs" / f"{arm}.trec"
            rankings = read_trec(path, set(judgments), documents)
            if set(rankings) != set(judgments) or any(len(rows) != min(50, len(documents)) for rows in rankings.values()):
                errors.append(f"{name}/{arm}: missing or truncated rankings")
            if saved["run_sha256"] != sha(path) or saved["corpus_size"] != len(documents) or saved["query_count"] != len(judgments):
                errors.append(f"{name}/{arm}: identity/count mismatch")
            rows = {qid: calculate_metrics(rankings.get(qid, []), rels) for qid, rels in sorted(judgments.items())}
            values = {metric: math.fsum(row[metric] for row in rows.values()) / len(rows) for metric in METRICS}
            recomputed[name][arm], per_query[name][arm] = values, rows
            saved_ids = [row["query_id"] for row in saved["per_query"]]
            if len(saved_ids) != len(set(saved_ids)) or set(saved_ids) != set(rows):
                errors.append(f"{name}/{arm}: per-query coverage or uniqueness mismatch")
            for row in saved["per_query"]:
                if row["query_id"] not in rows or any(not math.isclose(row[metric], rows[row["query_id"]][metric], abs_tol=1e-10, rel_tol=0) for metric in METRICS):
                    errors.append(f"{name}/{arm}: per-query mismatch")
            for metric in METRICS:
                if not math.isclose(values[metric], saved["metrics"][metric], abs_tol=1e-10, rel_tol=0):
                    errors.append(f"{name}/{arm}: {metric} mismatch")
        expected_pairs = {"filtered_minus_retrieved", "random_minus_base", "retrieved_minus_base", "filtered_minus_base"}
        comparisons = report["comparisons"][name]
        if set(comparisons) != expected_pairs:
            errors.append(f"{name}: missing/unexpected paired comparison")
        for pair in sorted(expected_pairs & set(comparisons)):
            comparison = comparisons[pair]
            treatment, reference = pair.split("_minus_")
            delta = math.fsum(per_query[name][treatment][qid]["ndcg@10"] - per_query[name][reference][qid]["ndcg@10"] for qid in sorted(judgments)) / len(judgments)
            if not math.isclose(delta, comparison["delta"], abs_tol=1e-10, rel_tol=0):
                errors.append(f"{name}/{pair}: paired mean mismatch")
    return {"ok": not errors, "errors": errors, "verified_metrics": recomputed, "aggregate_metric_count": 32,
            "scope": "TREC rankings, all per-query/aggregate relevance metrics, paired mean differences, canonical protocol/plan hashes, matched initialization/config/base identities in training metadata, adapter/run byte hashes and declared training completion; not bootstrap endpoints, model inference, optimizer replay or timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/adaptation"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--transfer-dir", type=Path, default=Path("data/nfcorpus"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.results_dir, args.source_dir, args.transfer_dir)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result = {"ok": False, "error": str(exc)}
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
