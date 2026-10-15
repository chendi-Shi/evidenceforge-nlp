"""Synthetic boundary checks for post-hoc case analysis; no model or real data."""
import importlib.util
import json
import math
from pathlib import Path

import pytest


@pytest.fixture
def analysis(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_distillation_failures.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("distillation_failures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extremes_keep_both_signs_and_deterministic_ties(analysis):
    assert analysis.select_extremes({"z": .5, "a": .5, "loss": -.4, "tie": 0}, 1) == {
        "largest_improvements": ["a"], "largest_regressions": ["loss"]}
    for values, count in (({}, 1), ({"a": float("nan")}, 1), ({"a": .1}, 0)):
        with pytest.raises(ValueError):
            analysis.select_extremes(values, count)


@pytest.fixture
def completed(analysis, tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    (data / "qrels").mkdir(parents=True)
    results.mkdir()
    (data / "corpus.jsonl").write_text(''.join(json.dumps({"_id": f"d{x}", "text": "PRIVATE TEXT"}) + '\n' for x in (1, 2)), encoding="utf-8")
    (data / "queries.jsonl").write_text('{"_id":"q","text":"PRIVATE QUERY"}\n', encoding="utf-8")
    (data / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\td1\t1\n", encoding="utf-8")
    report = {"status": analysis.COMPLETE, "spec_sha256": "spec", "runs": {},
              "dataset": {"queries": 1, "documents": 2, "file_sha256": {
                  name: analysis.sha256(data / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")}}}
    for arm, seeds in (("base", ("frozen",)), ("kd", analysis.SEEDS), ("kd_anchor", analysis.SEEDS)):
        report["runs"][arm] = {}
        for seed in seeds:
            ranking = ["d2", "d1"] if arm == "kd" else ["d1", "d2"]
            run = results / "scidocs" / arm / seed / "ranking.trec"
            run.parent.mkdir(parents=True)
            run.write_text(''.join(f"q Q0 {doc} {rank} {3-rank} test\n" for rank, doc in enumerate(ranking, 1)), encoding="utf-8")
            report["runs"][arm][seed] = {"run_sha256": analysis.sha256(run), "ignore_identical_ids": True,
                                         "per_query": [{"query_id": "q", **analysis.calculate_metrics(ranking, {"d1": 1})}]}
    (results / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (results / "independent_audit.json").write_text('{"status":"passed","ok":true,"spec_sha256":"spec"}', encoding="utf-8")
    return results, data, report


def test_recomputed_cases_expose_ids_ranks_and_no_text(analysis, completed):
    results, data, _ = completed
    output = analysis.analyze(results, data)
    case = output["comparisons"]["kd_anchor_minus_kd"]["largest_improvements"][0]
    assert case["mean_paired_ndcg_delta"] == pytest.approx(1 - 1 / math.log2(3))
    ranks = case["judged_positive_document_ranks"][0]["ranks"]
    assert ranks["kd"]["17"] == 2 and ranks["kd_anchor"]["17"] == 1
    assert output["comparisons"]["kd_anchor_minus_base"]["ties"] == 1
    assert "PRIVATE" not in json.dumps(output)
    assert str(results) not in json.dumps(output)


def test_missing_seed_rejected_before_target_read(analysis, completed):
    results, _, report = completed
    del report["runs"]["kd"]["42"]
    (results / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison seed"):
        analysis.analyze(results, results / "does-not-exist")


def test_metric_tampering_rejected(analysis, completed):
    results, data, report = completed
    report["runs"]["kd"]["42"]["per_query"][0]["ndcg@10"] = 0
    (results / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="metric mismatch"):
        analysis.analyze(results, data)


def test_explicit_output_cannot_overwrite_existing_file(analysis, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original", encoding="utf-8")
    assert analysis.main(["--output", str(output), "--results-dir", str(tmp_path / "missing")]) == 1
    assert output.read_text(encoding="utf-8") == "original"
