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


def verify(results_dir: Path, source_dir: Path, transfer_dir: Path) -> dict:
    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((results_dir / "frozen_adapters.json").read_text(encoding="utf-8"))
    errors, recomputed, per_query = [], {}, {}
    protocol_hash = sha(results_dir / "protocol.json")
    if report["protocol_sha256"] != protocol_hash or frozen["protocol_sha256"] != protocol_hash:
        errors.append("Protocol byte hash mismatch")
    protocol = json.loads((results_dir / "protocol.json").read_text(encoding="utf-8"))
    if report["spec_sha256"] != protocol["spec_sha256"]:
        errors.append("Protocol spec hash mismatch")
    for arm in ("random", "retrieved", "filtered"):
        directory = results_dir / "training" / arm
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["adapter_sha256"] != sha(directory / "adapter.safetensors") or frozen["adapters"][arm] != metadata["adapter_sha256"]:
            errors.append(f"{arm}: adapter hash mismatch")
        expected_steps = math.ceil(metadata["training_query_count"] / metadata["config"]["batch_size"]) * metadata["config"]["epochs"]
        if (metadata["completed_steps"] != expected_steps or metadata["parameter_delta_l2"] <= 0
                or metadata["nonzero_gradient_steps"] <= 0 or metadata["initial_trainable_sha256"] == metadata["final_trainable_sha256"]):
            errors.append(f"{arm}: incomplete or unchanged training")
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
            if len(saved["per_query"]) != len(rows):
                errors.append(f"{name}/{arm}: per-query count mismatch")
            for row in saved["per_query"]:
                if row["query_id"] not in rows or any(not math.isclose(row[metric], rows[row["query_id"]][metric], abs_tol=1e-10, rel_tol=0) for metric in METRICS):
                    errors.append(f"{name}/{arm}: per-query mismatch")
            for metric in METRICS:
                if not math.isclose(values[metric], saved["metrics"][metric], abs_tol=1e-10, rel_tol=0):
                    errors.append(f"{name}/{arm}: {metric} mismatch")
        for pair, comparison in report["comparisons"][name].items():
            treatment, reference = pair.split("_minus_")
            delta = math.fsum(per_query[name][treatment][qid]["ndcg@10"] - per_query[name][reference][qid]["ndcg@10"] for qid in sorted(judgments)) / len(judgments)
            if not math.isclose(delta, comparison["delta"], abs_tol=1e-10, rel_tol=0):
                errors.append(f"{name}/{pair}: paired mean mismatch")
    return {"ok": not errors, "errors": errors, "verified_metrics": recomputed, "aggregate_metric_count": 32,
            "scope": "TREC rankings, all per-query/aggregate relevance metrics, paired mean differences, protocol/adapter/run hashes and training completion; not bootstrap endpoints, model inference or timing."}


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
