import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def analysis(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_adaptation_results.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("adaptation_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalized_claim_excludes_official_train_including_validation(analysis):
    queries = {"train": {"text": "IL-2 boosts T cells!"}, "validation": {"text": "Dose 10 mg."},
               "a": {"text": "il 8 boosts T cells"}, "b": {"text": "Dose 20 MG"},
               "c": {"text": "Different claim."}}
    result = analysis.normalized_overlap(queries, {"train": {"d": 1}, "validation": {"e": 1}}, ["a", "b", "c"])
    assert result["official_train_query_count"] == 2
    assert result["retained_query_ids"] == ["c"]
    assert [row["matching_official_train_query_ids"] for row in result["excluded"]] == [["train"], ["validation"]]
    assert analysis.normalized_claim("β2: ÉTUDES_12!") == "β<num> études_<num>"


def test_paired_bootstrap_known_difference_and_validation(analysis):
    result = analysis.paired_bootstrap([.5, .75, 1.], [.25, .5, .75], samples=100)
    assert result["delta"] == result["ci95_low"] == result["ci95_high"] == .25
    assert analysis.paired_bootstrap([0, 1], [1, 0], samples=100) == analysis.paired_bootstrap([0, 1], [1, 0], samples=100)
    for a, b in (([], []), ([1], [1, 2]), ([float("nan")], [0])):
        with pytest.raises(ValueError):
            analysis.paired_bootstrap(a, b)


def synthetic_dataset(analysis, root):
    data, results = root / "data", root / "results"
    (data / "qrels").mkdir(parents=True)
    runs = results / "scifact" / "runs"
    runs.mkdir(parents=True)
    documents = {key: {"_id": key, "title": key, "text": key} for key in ("d1", "d2", "d3")}
    queries = {key: {"_id": key, "text": text} for key, text in
               (("train", "Dose 10"), ("q1", "Dose 20"), ("q2", "Gain"), ("q3", "Loss"))}
    for name, rows in (("corpus", documents), ("queries", queries)):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows.values()), encoding="utf-8")
    (data / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\ntrain\td1\t1\n", encoding="utf-8")
    (data / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t2\nq2\td2\t1\nq3\td1\t1\n", encoding="utf-8")
    qrels = analysis.read_qrels(data / "qrels" / "test.tsv")
    saved = {"corpus_sha256": analysis.corpus_hash(documents), "methods": {}}
    for arm in analysis.ARMS:
        rankings = {"q1": ["d1", "d2", "d3"], "q2": ["d2", "d1", "d3"], "q3": ["d1", "d2", "d3"]}
        if arm == "filtered":
            rankings.update(q2=["d1", "d2", "d3"], q3=["d3", "d2", "d1"])
        path = runs / f"{arm}.trec"
        path.write_text("".join(f"{qid} Q0 {doc} {rank} {4-rank} {arm}\n"
                                for qid in rankings for rank, doc in enumerate(rankings[qid], 1)), encoding="utf-8")
        rows = {qid: analysis.calculate_metrics(rankings[qid], qrels[qid]) for qid in rankings}
        saved["methods"][arm] = {"run_sha256": analysis.sha256(path), "query_count": 3, "corpus_size": 3,
                                 "metrics": analysis.aggregate(rows, sorted(rows)),
                                 "per_query": [{"query_id": qid, **rows[qid]} for qid in reversed(sorted(rows))]}
    return data, results, saved


def test_analysis_pairs_by_query_and_keeps_post_hoc_separate(analysis, tmp_path):
    data, results, saved = synthetic_dataset(analysis, tmp_path)
    output = analysis.analyze_dataset("scifact", data, results, saved, samples=50, seed=42, case_count=5)
    comparison = output["comparisons_ndcg@10"]["filtered_minus_retrieved"]
    assert (comparison["wins"], comparison["ties"], comparison["losses"]) == (1, 1, 1)
    assert output["query_count"] == 3
    assert output["largest_gains"][0]["query_id"] == "q2"
    assert output["largest_regressions"][0]["query_id"] == "q3"
    sensitivity = output["post_hoc_normalized_overlap_sensitivity"]
    assert sensitivity["retained_query_ids"] == ["q2", "q3"]
    assert sensitivity["excluded_count"] == 1
    assert sensitivity["metrics"]["filtered"]["ndcg@10"] == .75
    assert output["metrics"]["filtered"]["ndcg@10"] == pytest.approx(5 / 6)


def test_analysis_rejects_duplicate_report_rows_and_tampered_rankings(analysis, tmp_path):
    data, results, saved = synthetic_dataset(analysis, tmp_path)
    original = saved["methods"]["base"]["per_query"]
    saved["methods"]["base"]["per_query"] = [original[0], original[0], original[2]]
    with pytest.raises(ValueError, match="Duplicate or missing"):
        analysis.analyze_dataset("scifact", data, results, saved, samples=5, seed=42, case_count=1)
    saved["methods"]["base"]["per_query"] = original
    (results / "scifact" / "runs" / "base.trec").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="byte hash mismatch"):
        analysis.analyze_dataset("scifact", data, results, saved, samples=5, seed=42, case_count=1)


def test_analysis_refuses_incomplete_report(analysis, tmp_path):
    (tmp_path / "report.json").write_text('{"datasets": {"scifact": {}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Both complete"):
        analysis.analyze(tmp_path, tmp_path, tmp_path)
