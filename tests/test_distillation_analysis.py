"""Small synthetic JSON fixtures; never access real target data or load models."""
import importlib.util
import json
from pathlib import Path
import random
import statistics

import pytest


@pytest.fixture
def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_distillation_results.py"
    spec = importlib.util.spec_from_file_location("distillation_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def completed(analysis, tmp_path):
    spec = analysis.digest("synthetic protocol")

    def result(values, arm, seed, *, source=False):
        ids = [f"{'source' if source else 'target'}-{index}" for index in range(3)]
        rows = [{"query_id": qid, **{name: value for name in analysis.METRICS}}
                for qid, value in zip(ids, values, strict=True)]
        return {
            "metrics": {name: statistics.fmean(values) for name in analysis.METRICS},
            "per_query": rows, "query_count": 3, "corpus_size": 9,
            "query_ids_sha256": analysis.digest(sorted(ids)), "document_ids_sha256": "documents",
            "qrels_sha256": "qrels", "ignore_identical_ids": not source,
            "score_function": "normalized dot product (cosine)",
            "tie_break": "document ID lexicographic ascending", "saved_ranking_depth": 50,
            "identity": {"spec_sha256": spec, "arm": arm, "seed": seed,
                         "weights": "base" if arm == "base" else f"selected-{arm}-{seed}",
                         "dataset": "synthetic corpus", "queries_sha256": "queries", "qrels_sha256": "qrels"},
        }

    baseline = result([.2, .4, .6], "base", "frozen", source=True)
    baseline["spec_sha256"] = spec
    write_json(tmp_path / "source_validation_base.json", baseline)
    base_values = [.4, .6, .5]
    report = {"spec_sha256": spec, "status": analysis.COMPLETE_STATUS,
              "dataset": {"name": "synthetic fixture, not a real benchmark"},
              "runs": {"base": {"frozen": result(base_values, "base", "frozen")}}}
    for seed, interaction in zip(analysis.SEEDS, [.05, .10, -.05], strict=True):
        offsets = {"ce": [-.1, -.1, -.1], "kd": [-.1, 0, -.05], "anchor": [-.03, -.05, -.07],
                   "kd_anchor": [interaction, interaction + .02, interaction - .02]}
        for arm in analysis.ARMS:
            values = [base + offset for base, offset in zip(base_values, offsets[arm], strict=True)]
            report["runs"].setdefault(arm, {})[seed] = result(values, arm, seed)
            validations = []
            for step, improvement in ((250, .1), (500, .15), (750, .12)):
                validations.append({"step": step,
                                    "metrics": {name: baseline["metrics"][name] + improvement
                                                for name in analysis.METRICS},
                                    "drift": {key: step / 10000 for key in analysis.DRIFT_KEYS}})
            training = {"plan": {"spec_sha256": spec, "seed": int(seed), "arm": arm,
                                 "order_sha256": f"order-{seed}"},
                        "selected_metadata_sha256": f"selected-{arm}-{seed}",
                        "progress": {"completed_steps": 750},
                        "selection": {"best_step": 500, "best_ndcg": validations[1]["metrics"]["ndcg@10"],
                                      "validations": validations}}
            write_json(tmp_path / "training" / arm / seed / "report.json", training)
    write_json(tmp_path / "report.json", report)
    return tmp_path


def test_per_seed_factorial_sign_and_descriptive_sd(analysis, completed):
    result = analysis.analyze(completed, samples=50)
    assert result["status"] == "complete" and result["ok"] is True
    for seed, expected in zip(analysis.SEEDS, [.05, .1, -.05], strict=True):
        row = result["per_seed"][seed]
        assert row["interaction"]["metrics"]["ndcg@10"] == pytest.approx(expected)
        assert row["interaction"]["ndcg@10_interval"]["delta"] == pytest.approx(expected)
        assert row["interaction"]["ndcg@10_interval"]["query_count"] == 3
        assert row["arms"]["ce"]["target_minus_base"]["ndcg@10"] == pytest.approx(-.1)
    summary = result["descriptive_across_seeds"]["interaction"]["ndcg@10"]
    assert summary["seed_count"] == 3
    assert summary["mean"] == pytest.approx(statistics.mean([.05, .1, -.05]))
    assert summary["sample_sd"] == pytest.approx(statistics.stdev([.05, .1, -.05]))
    assert "ci95_low" not in summary


def test_pair_by_query_id_not_row_position_and_keep_training_seeds(analysis, completed):
    before = analysis.analyze(completed, samples=40)
    path = completed / "report.json"
    report = read_json(path)
    rng = random.Random(99)
    for entries in report["runs"].values():
        for run in entries.values():
            rng.shuffle(run["per_query"])
    report["runs"]["kd"] = dict(reversed(list(report["runs"]["kd"].items())))
    write_json(path, report)
    after = analysis.analyze(completed, samples=40)
    assert after["per_seed"] == before["per_seed"]
    assert after["descriptive_across_seeds"] == before["descriptive_across_seeds"]


def test_source_gain_target_regression_and_complete_source_drift_trajectory(analysis, completed):
    result = analysis.analyze(completed, samples=20)
    row = result["per_seed"]["17"]["arms"]["ce"]
    source = row["source_validation"]
    assert source["selected_minus_source_base"]["ndcg@10"] == pytest.approx(.15)
    assert row["target_minus_base_ndcg@10_interval"]["ci95_high"] == pytest.approx(-.1)
    assert source["selected_step"] == 500
    assert [record["step"] for record in source["validation_trajectory"]] == [250, 500, 750]
    assert source["selected_source_drift"][analysis.DRIFT_KEYS[0]] == pytest.approx(.05)
    assert "source_drift" in source["validation_trajectory"][0]
    assert "target_drift" not in row
    assert any("Target drift was not measured" in note for note in result["notes"])
    assert any("batch order" in note for note in result["notes"])


@pytest.mark.parametrize("mutation", ["missing_arm", "missing_seed", "different_seed"])
def test_missing_group_or_unpaired_seed_rejected(analysis, completed, mutation):
    path = completed / "report.json"
    report = read_json(path)
    if mutation == "missing_arm":
        del report["runs"]["anchor"]
    elif mutation == "missing_seed":
        del report["runs"]["kd"]["42"]
    else:
        report["runs"]["kd"]["99"] = report["runs"]["kd"].pop("42")
    write_json(path, report)
    with pytest.raises(ValueError, match="four arms|matching training seeds"):
        analysis.analyze(completed, samples=2)


def test_different_queries_rejected_even_when_counts_and_aggregates_match(analysis, completed):
    path = completed / "report.json"
    report = read_json(path)
    row = report["runs"]["anchor"]["42"]
    row["per_query"][0]["query_id"] = "different-query"
    row["query_ids_sha256"] = analysis.digest(sorted(item["query_id"] for item in row["per_query"]))
    write_json(path, report)
    with pytest.raises(ValueError, match="identical query/document"):
        analysis.analyze(completed, samples=2)


@pytest.mark.parametrize("mutation,expected", [
    ("duplicate_query", "Duplicate"), ("aggregate", "Aggregate"), ("nan", "finite"),
    ("qrels", "identical query/document"), ("seed_identity", "identity mismatch"),
    ("selected_weights", "checkpoint identities"),
])
def test_inconsistent_completed_target_inputs_rejected(analysis, completed, mutation, expected):
    path = completed / "report.json"
    report = read_json(path)
    row = report["runs"]["kd"]["42"]
    if mutation == "duplicate_query":
        row["per_query"][1]["query_id"] = row["per_query"][0]["query_id"]
    elif mutation == "aggregate":
        row["metrics"]["ndcg@10"] += .1
    elif mutation == "nan":
        row["per_query"][0]["ndcg@10"] = float("nan")
    elif mutation == "qrels":
        row["qrels_sha256"] = "different judgments"
    elif mutation == "seed_identity":
        row["identity"]["seed"] = "17"
    else:
        row["identity"]["weights"] = "different checkpoint"
    write_json(path, report)
    with pytest.raises(ValueError, match=expected):
        analysis.analyze(completed, samples=2)


def test_row_order_must_match_within_each_seed(analysis, completed):
    path = completed / "training/anchor/42/report.json"
    training = read_json(path)
    training["plan"]["order_sha256"] = "wrong row order"
    write_json(path, training)
    with pytest.raises(ValueError, match="same training row order"):
        analysis.analyze(completed, samples=2)


def test_selection_uses_only_earliest_best_source_step(analysis, completed):
    path = completed / "training/ce/17/report.json"
    training = read_json(path)
    selection = training["selection"]
    selection["validations"][0]["metrics"] = dict(selection["validations"][1]["metrics"])
    selection["best_step"] = 250
    write_json(path, training)
    result = analysis.analyze(completed, samples=2)
    assert result["per_seed"]["17"]["arms"]["ce"]["source_validation"]["selected_step"] == 250
    selection["best_step"] = 500
    write_json(path, training)
    with pytest.raises(ValueError, match="source-only earliest-best"):
        analysis.analyze(completed, samples=2)


def test_pending_does_not_read_partial_files_or_generate_default_artifact(analysis, tmp_path, capsys):
    # An unfinished/malformed root report must not be parsed before all files exist.
    (tmp_path / "report.json").write_text("not yet complete", encoding="utf-8")
    assert analysis.main(["--results-dir", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pending" and result["ok"] is False
    assert "source_validation_base.json" in result["missing_files"]
    assert "per_seed" not in result
    assert not (tmp_path / "analysis.json").exists()


def test_missing_training_file_is_pending(analysis, completed):
    (completed / "training/kd/73/report.json").unlink()
    result = analysis.analyze(completed)
    assert result["status"] == "pending" and not result["ok"]
    assert result["missing_files"] == ["training/kd/73/report.json"]


def test_bootstrap_constant_contrast_sign_and_reproducibility(analysis):
    first = analysis.paired_intervals({"positive": [.2] * 3, "negative": [-.1] * 3,
                                      "mixed": [-.2, .1, .4]}, samples=31)
    second = analysis.paired_intervals({"positive": [.2] * 3, "negative": [-.1] * 3,
                                       "mixed": [-.2, .1, .4]}, samples=31)
    assert first == second
    for name, expected in (("positive", .2), ("negative", -.1)):
        assert first[name]["delta"] == pytest.approx(expected)
        assert first[name]["ci95_low"] == pytest.approx(expected)
        assert first[name]["ci95_high"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="equally sized"):
        analysis.paired_intervals({"a": [1], "b": [1, 2]})
    with pytest.raises(ValueError, match="positive integer"):
        analysis.paired_intervals({"a": [1]}, samples=0)


def test_cli_explicit_output_and_failed_exit(analysis, completed, tmp_path):
    output = tmp_path / "derived/secondary.json"
    assert analysis.main(["--results-dir", str(completed), "--samples", "2", "--output", str(output)]) == 0
    assert read_json(output)["status"] == "complete"
    report = read_json(completed / "report.json")
    del report["runs"]["anchor"]
    write_json(completed / "report.json", report)
    failed_output = tmp_path / "derived/failed.json"
    assert analysis.main(["--results-dir", str(completed), "--output", str(failed_output)]) == 1
    assert read_json(failed_output)["ok"] is False


def test_cli_cannot_overwrite_existing_input(analysis, completed, capsys):
    source = completed / "report.json"
    before = source.read_bytes()
    assert analysis.main(["--results-dir", str(completed), "--samples", "2", "--output", str(source)]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "failed" and error["ok"] is False
    assert "refusing to overwrite" in error["reason"]
    assert source.read_bytes() == before
