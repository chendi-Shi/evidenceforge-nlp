import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def audit(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_modern_results.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("modern_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Tiny synthetic fixtures exercise the production checks without models.
    monkeypatch.setattr(module, "EXPECTED_DOCUMENTS", 3)
    monkeypatch.setattr(module, "EXPECTED_TEST_QUERIES", 2)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixture_results(audit, root):
    data, results = root / "data", root / "results"
    (data / "qrels").mkdir(parents=True)
    (results / "runs").mkdir(parents=True)
    documents = {key: {"_id": key, "title": key, "text": f"Abstract {key}"} for key in ("d1", "d2", "d3")}
    queries = {key: {"_id": key, "text": f"Claim {key}"} for key in ("q1", "q2", "train")}
    for name, rows in (("corpus", documents), ("queries", queries)):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows.values()), encoding="utf-8")
    (data / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t2\nq2\td2\t1\n", encoding="utf-8")
    qrels = audit.read_qrels(data / "qrels" / "test.tsv")
    encoder = {**audit.MODEL_IDENTITY, "backend": "cpu", "runtime_archive_sha256": audit.RUNTIME_ARCHIVES["cpu"],
               "threads": 2, "max_tokens": 128, "append_eos": True}
    spec = {
        "schema": 1, "method": audit.METHOD, "encoder": encoder, "saved_ranking_depth": 50,
        "document_count": 3, "query_count": 2, "source_corpus_sha256": audit.corpus_hash(documents),
        "source_queries_sha256": audit.canonical_digest({key: row["text"] for key, row in queries.items()}),
        "source_test_qrels_sha256": audit.canonical_digest(qrels),
        "ordered_document_ids_sha256": audit.canonical_digest(sorted(documents)),
        "ordered_test_query_ids": sorted(qrels), "tie_break": "document ID lexicographic ascending",
    }
    protocol = {"spec": spec, "spec_sha256": audit.canonical_digest(spec), "registered_at_utc": "synthetic"}
    write_json(results / "protocol.json", protocol)
    ranking = {"q1": ["d1", "d2", "d3"], "q2": ["d2", "d1", "d3"]}
    trec_path = results / "runs" / f"{audit.METHOD}.trec"
    trec_path.write_text("".join(f"{qid} Q0 {doc} {rank} {4-rank} {audit.METHOD}\n"
                                 for qid, docs in ranking.items() for rank, doc in enumerate(docs, 1)), encoding="utf-8")
    rows = [{"query_id": qid, **audit.calculate_metrics(ranking[qid], qrels[qid])} for qid in sorted(qrels)]
    per_query_path = results / "runs" / f"{audit.METHOD}.per_query.json"
    write_json(per_query_path, list(reversed(rows)))
    report = {
        "protocol_sha256": audit.sha256(results / "protocol.json"), "spec_sha256": protocol["spec_sha256"],
        "encoder": {**encoder, "server_command": ["llama-server.exe", "--embedding"]},
        "artifacts": {key: {"path": f"runs/{path.name}", "sha256": audit.sha256(path)}
                      for key, path in (("trec", trec_path), ("per_query", per_query_path))},
        "result": {"run_sha256": audit.sha256(trec_path), "corpus_size": 3, "query_count": 2, "split": "test",
                   "ranking_scope": "entire corpus", "tie_break": spec["tie_break"], "per_query": rows,
                   "metrics": {metric: sum(row[metric] for row in rows) / len(rows) for metric in audit.METRICS}},
    }
    write_json(results / "report.json", report)
    return data, results, protocol, report


def test_independent_audit_accepts_reordered_rows_and_runtime_command(audit, tmp_path):
    data, results, _, report = fixture_results(audit, tmp_path)
    verified = audit.verify(results, data)
    assert verified["ok"]
    assert verified["verified_metrics"] == report["result"]["metrics"]
    assert verified["test_query_count"] == 2
    assert "outside the saved rankings" in verified["scope"]


@pytest.mark.parametrize("location", ["report", "artifact"])
def test_duplicate_per_query_rejected_even_with_valid_byte_hash(audit, tmp_path, location):
    data, results, _, report = fixture_results(audit, tmp_path)
    duplicate = [report["result"]["per_query"][0]] * 2
    if location == "report":
        report["result"]["per_query"] = duplicate
    else:
        path = results / report["artifacts"]["per_query"]["path"]
        write_json(path, duplicate)
        report["artifacts"]["per_query"]["sha256"] = audit.sha256(path)
    write_json(results / "report.json", report)
    with pytest.raises(ValueError, match="duplicate or missing per-query"):
        audit.verify(results, data)


def test_tampered_spec_rejected_even_when_protocol_byte_hash_updated(audit, tmp_path):
    data, results, protocol, report = fixture_results(audit, tmp_path)
    protocol["spec"]["encoder"]["max_tokens"] = 256
    write_json(results / "protocol.json", protocol)
    report["protocol_sha256"] = audit.sha256(results / "protocol.json")
    write_json(results / "report.json", report)
    with pytest.raises(ValueError, match="spec hash mismatch"):
        audit.verify(results, data)


def test_missing_ranking_rejected_even_with_matching_hashes(audit, tmp_path):
    data, results, _, report = fixture_results(audit, tmp_path)
    path = results / report["artifacts"]["trec"]["path"]
    path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
    report["artifacts"]["trec"]["sha256"] = report["result"]["run_sha256"] = audit.sha256(path)
    write_json(results / "report.json", report)
    with pytest.raises(ValueError, match="Missing or truncated rankings"):
        audit.verify(results, data)


@pytest.mark.parametrize("change, message", [
    ("encoder", "encoder metadata"), ("query", "Query text identity"), ("metric", "metric mismatch"),
])
def test_identity_and_metric_tampering_rejected(audit, tmp_path, change, message):
    data, results, _, report = fixture_results(audit, tmp_path)
    if change == "encoder":
        report["encoder"]["threads"] = 1
    elif change == "query":
        path = data / "queries.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("Claim q1", "Changed q1"), encoding="utf-8")
    else:
        report["result"]["metrics"]["ndcg@10"] = 0.0
    write_json(results / "report.json", report)
    with pytest.raises(ValueError, match=message):
        audit.verify(results, data)


def test_cli_returns_nonzero_and_json_for_missing_results(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_modern_results.py"
    output = tmp_path / "audit.json"
    run = subprocess.run([sys.executable, str(script), "--results-dir", str(tmp_path), "--output", str(output)],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert json.loads(run.stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert not json.loads(run.stdout)["ok"]
