"""Audit saved full-SciFact Qwen rankings and artifact identities without inference.

Only standard-library modules are used. This verifies the saved top-50 rankings,
not whether the model actually scored documents outside those saved rankings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_adaptation_results import corpus_hash, read_jsonl
from verify_results import METRICS, calculate_metrics, read_qrels, read_trec

EXPECTED_DOCUMENTS = 5183
EXPECTED_TEST_QUERIES = 300
METHOD = "qwen3_06b_q8"
MODEL_IDENTITY = {
    "model_id": "Qwen/Qwen3-Embedding-0.6B-GGUF",
    "model_revision": "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
    "model_file": "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "model_sha256": "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
    "runtime": "llama.cpp", "runtime_version": "b11580",
    "pooling": "last", "normalization": "L2", "dimension": 1024,
}
RUNTIME_ARCHIVES = {
    "cpu": "430efe5b4f10db617ae0f5732f7e9360fa7be4344d1d97688e09bbd3b5b084da",
    "vulkan": "6e520f0b8bbc6608b53d70065ed80ee90d33acf1636f886d965e0c87f708dff2",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_digest(value) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(results_dir: Path, data_dir: Path, *, tolerance: float = 1e-10) -> dict:
    """Fail closed on missing, altered, inconsistent or incomplete artifacts."""
    require(math.isfinite(tolerance) and tolerance >= 0, "Tolerance must be finite and nonnegative")
    results_dir, data_dir = Path(results_dir), Path(data_dir)
    protocol_path = results_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    spec = protocol["spec"]
    spec_hash, protocol_hash = canonical_digest(spec), sha256(protocol_path)
    require(protocol["spec_sha256"] == spec_hash == report["spec_sha256"], "Protocol spec hash mismatch")
    require(report["protocol_sha256"] == protocol_hash, "Protocol byte hash mismatch")
    require(spec["schema"] == 1 and spec["method"] == METHOD, "Unexpected protocol schema or method")
    require(spec["saved_ranking_depth"] == 50, "Expected the frozen top-50 ranking depth")

    encoder = spec["encoder"]
    require(all(encoder.get(key) == value for key, value in MODEL_IDENTITY.items()),
            "Protocol differs from the pinned Qwen model/runtime identity")
    require(encoder.get("backend") in RUNTIME_ARCHIVES
            and encoder.get("runtime_archive_sha256") == RUNTIME_ARCHIVES[encoder["backend"]],
            "Protocol runtime archive identity mismatch")
    # Starting the local server appends this operational field to the report.
    # All other fields, including thread count and tokenization, must match.
    require("server_command" not in encoder, "Frozen protocol unexpectedly contains server_command")
    reported_encoder = {key: value for key, value in report["encoder"].items() if key != "server_command"}
    require(reported_encoder == encoder, "Report encoder metadata differs from protocol")

    documents = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    qrels = read_qrels(data_dir / "qrels" / "test.tsv")
    query_ids, document_ids = sorted(qrels), sorted(documents)
    require(len(documents) == EXPECTED_DOCUMENTS and len(qrels) == EXPECTED_TEST_QUERIES,
            "Expected complete 5,183-document, 300-test-query SciFact data")
    require(all(qid in queries and any(rel > 0 for rel in rels.values()) for qid, rels in qrels.items()),
            "Missing query text or positive test judgment")
    require(all(doc_id in documents for rels in qrels.values() for doc_id in rels),
            "Qrels reference a missing document")
    require(spec["document_count"] == len(documents) and spec["query_count"] == len(qrels),
            "Protocol data count mismatch")
    require(spec["source_corpus_sha256"] == corpus_hash(documents), "Corpus identity mismatch")
    require(spec["source_queries_sha256"] == canonical_digest({qid: row["text"] for qid, row in queries.items()}),
            "Query text identity mismatch")
    require(spec["source_test_qrels_sha256"] == canonical_digest(qrels), "Test qrels identity mismatch")
    require(spec["ordered_document_ids_sha256"] == canonical_digest(document_ids), "Document ID identity mismatch")
    require(spec["ordered_test_query_ids"] == query_ids, "Test query ID coverage, order or uniqueness mismatch")

    paths = {"trec": f"runs/{METHOD}.trec", "per_query": f"runs/{METHOD}.per_query.json"}
    hashes = {}
    require(set(report["artifacts"]) == set(paths), "Unexpected or missing result artifact")
    for kind, relative_path in paths.items():
        artifact = report["artifacts"][kind]
        require(artifact["path"] == relative_path, f"Unexpected {kind} artifact path")
        hashes[kind] = sha256(results_dir / relative_path)
        require(artifact["sha256"] == hashes[kind], f"{kind} artifact byte hash mismatch")
    saved = report["result"]
    require(saved["run_sha256"] == hashes["trec"], "Result run byte hash mismatch")
    require(saved["corpus_size"] == len(documents) and saved["query_count"] == len(qrels),
            "Result data count mismatch")
    require(saved["split"] == "test" and saved["ranking_scope"] == "entire corpus",
            "Unexpected result split or declared ranking scope")
    require(saved["tie_break"] == spec["tie_break"] == "document ID lexicographic ascending",
            "Declared tie-breaking policy mismatch")
    rankings = read_trec(results_dir / paths["trec"], set(query_ids), set(document_ids))
    require(set(rankings) == set(query_ids) and all(len(rows) == min(50, len(documents)) for rows in rankings.values()),
            "Missing or truncated rankings")
    rows = {qid: calculate_metrics(rankings[qid], qrels[qid]) for qid in query_ids}

    def compare_metrics(reported, actual, location):
        for metric in METRICS:
            value = reported.get(metric)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    and math.isclose(value, actual[metric], rel_tol=0, abs_tol=tolerance),
                    f"{location}: {metric} metric mismatch")

    def check_per_query(saved_rows, location):
        require(isinstance(saved_rows, list), f"{location}: expected per-query list")
        ids = [row["query_id"] for row in saved_rows]
        require(len(ids) == len(set(ids)) and set(ids) == set(query_ids),
                f"{location}: duplicate or missing per-query ID")
        for row in saved_rows:
            compare_metrics(row, rows[row["query_id"]], f"{location}/{row['query_id']}")

    check_per_query(saved["per_query"], "report")
    artifact_rows = json.loads((results_dir / paths["per_query"]).read_text(encoding="utf-8"))
    check_per_query(artifact_rows, "per_query artifact")
    metrics = {metric: math.fsum(row[metric] for row in rows.values()) / len(rows) for metric in METRICS}
    require(set(saved["metrics"]) == set(METRICS), "Missing or unexpected aggregate metric")
    compare_metrics(saved["metrics"], metrics, "aggregate")
    return {
        "ok": True, "method": METHOD, "corpus_size": len(documents), "test_query_count": len(qrels),
        "ranking_depth": min(50, len(documents)), "verified_metrics": metrics, "tolerance": tolerance,
        "protocol_sha256": protocol_hash, "spec_sha256": spec_hash, "artifact_sha256": hashes,
        "scope": (
            "Independent metric arithmetic from saved explicit TREC ranks and actual test qrels, every saved "
            "per-query metric and all four aggregates, canonical protocol and data identities, artifact byte hashes, "
            "and consistency of declared pinned encoder/runtime metadata. Does not rerun model inference, inspect "
            "model/runtime binaries, prove computation over documents outside the saved rankings, verify historical "
            "preregistration timing, or validate latency/throughput."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/modern/scifact-full"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.results_dir, args.data_dir)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result = {"ok": False, "error": str(exc)}
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
