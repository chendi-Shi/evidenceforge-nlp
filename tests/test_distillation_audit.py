"""Tiny fabricated artifacts test the auditor, never retrieval quality or training."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def audit(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_distillation.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("distillation_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in (("DOCUMENTS", 3), ("QUERIES", 2), ("TRAIN_QUERIES", 4), ("VALIDATION_QUERIES", 2)):
        monkeypatch.setattr(module, name, value)
    return module


def reference_summary(runs):
    """Vectorized bootstrap reference, separate from the audit implementation."""
    metrics = ("ndcg@10", "mrr@10", "recall@10", "recall@50")
    arms = {}
    for arm, entries in runs.items():
        arms[arm] = {"seed_count": len(entries), "per_seed_metrics": {s: r["metrics"] for s, r in entries.items()},
                     "metrics": {m: {"mean": float(np.mean([r["metrics"][m] for r in entries.values()])),
                                     "sample_std": float(np.std([r["metrics"][m] for r in entries.values()], ddof=1))
                                     if len(entries) > 1 else None} for m in metrics}}
    pairs = {}
    for seed in runs["kd"]:
        differences = np.asarray([a["ndcg@10"] - b["ndcg@10"] for a, b in
                                  zip(runs["kd_anchor"][seed]["per_query"], runs["kd"][seed]["per_query"], strict=True)])
        indices = np.random.default_rng(42).integers(len(differences), size=(2000, len(differences)))
        low, high = np.quantile(differences[indices].mean(axis=1), [.025, .975])
        pairs[seed] = {"delta": float(differences.mean()), "ci95_low": float(low), "ci95_high": float(high),
                       "samples": 2000, "seed": 42, "training_seed": seed, "query_count": len(differences)}
    deltas = [r["delta"] for r in pairs.values()]
    return {"arms": arms, "primary": {"treatment": "kd_anchor", "reference": "kd", "metric": "ndcg@10",
                                      "per_seed": pairs, "mean_delta": float(np.mean(deltas)),
                                      "sample_std_delta": float(np.std(deltas, ddof=1))}}


@pytest.fixture
def fixture(audit, tmp_path):
    results, data, source = tmp_path / "results", tmp_path / "data", tmp_path / "source"
    for directory in (results, data / "qrels", source / "prepared"):
        directory.mkdir(parents=True)
    for split, ids in (("train", ["s0", "s1", "s2", "s3"]), ("validation", ["v0", "v1"])):
        (source / "prepared" / f"{split}.jsonl").write_text(
            "".join(json.dumps({"query_id": key, "query": f"synthetic {key}"}) + "\n" for key in ids), encoding="utf-8")
    config = {"seeds": [17, 42, 73], "train_queries": 4, "validation_queries": 2, "batch_size": 2,
              "epochs": 1, "validation_every": 1, "student_tokens": 256, "threads": 2}
    code_hashes = {}
    for name in audit.CODE_FILES:
        path = tmp_path / "src/evidenceforge" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic fixture, not production code\n", encoding="utf-8")
        code_hashes[name] = audit.sha256(path)
    spec = {"version": 3, "config": config, "arms": audit.ARMS, "implementation_sha256": code_hashes,
            "train_sha256": audit.sha256(source / "prepared/train.jsonl"),
            "validation_sha256": audit.sha256(source / "prepared/validation.jsonl"),
            "base_model_sha256": "a" * 64, "base_config_sha256": "b" * 64, "base_tokenizer_sha256": "c" * 64,
            "student_configs": {arm: {"seed": 42, "kd_weight": weights[0], "anchor_weight": weights[1]}
                                for arm, weights in audit.ARMS.items()}}
    spec_hash = audit.digest(spec)
    write(results / "protocol.json", {"spec": spec, "spec_sha256": spec_hash})
    frozen = {"spec_sha256": spec_hash, "runs": {}}
    all_metrics = {name: 0.5 for name in audit.METRICS}
    for arm in audit.ARMS:
        for seed in config["seeds"]:
            key, directory = f"{arm}/{seed}", results / "training" / arm / str(seed)
            metadata = {}
            for kind in ("selected", "final"):
                (directory / kind).mkdir(parents=True)
                weights_path = directory / kind / "synthetic.safetensors"
                weights_path.write_bytes(b"SYNTHETIC BYTES: this is not a trained tensor file")
                metadata = {"filename": weights_path.name, "suffix_sha256": audit.sha256(weights_path),
                            "parameter_sha256": "d" * 64, "initial_parameter_sha256": "e" * 64,
                            "trainable_parameter_count": 2, "config": {**spec["student_configs"][arm], "seed": seed},
                            "identity": {"base_model_sha256": spec["base_model_sha256"], "config_sha256": spec["base_config_sha256"],
                                         "tokenizer_sha256": spec["base_tokenizer_sha256"], "max_length": 256, "threads": 2}}
                write(directory / kind / "metadata.json", metadata)
            selection = {"best_step": 1, "best_ndcg": .5,
                         "validations": [{"step": step, "metrics": dict(all_metrics)} for step in (1, 2)]}
            write(directory / "selection.json", selection)
            trace = [{"step": step, "epoch": 0, "trainable_parameter_count": 2, "dropout_enabled": False,
                      "loss": 1.0, "ce": 1.0, "kd": 0.0, "anchor": 0.0, "gradient_l2_before_clip": .1} for step in (1, 2)]
            write(directory / "trace.json", trace)
            order = np.random.default_rng(seed).permutation(4)
            training = {"plan": {"spec_sha256": spec_hash, "teacher_sha256": "f" * 64, "arm": arm, "seed": seed,
                                 "order_sha256": audit.digest([f"s{index}" for index in order])},
                        "progress": {"completed_steps": 2, "nonzero_gradient_steps": 2, "epoch": 0, "row_cursor": 4},
                        "selection": selection, "trainable_parameters": 2, "final_suffix": metadata,
                        "selected_metadata_sha256": audit.sha256(directory / "selected/metadata.json")}
            write(directory / "report.json", training)
            frozen["runs"][key] = training
    write(results / "frozen_students.json", frozen)
    docs = [{"_id": f"d{i}", "text": f"Synthetic document {i}", "title": ""} for i in range(1, 4)]
    queries = {"q1": "Synthetic first query", "q2": "Synthetic second query"}
    (data / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in reversed(docs)), encoding="utf-8")
    (data / "queries.jsonl").write_text("".join(json.dumps({"_id": k, "text": v}) + "\n" for k, v in queries.items()), encoding="utf-8")
    (data / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t2\nq2\td1\t1\n", encoding="utf-8")
    qrels = audit.read_qrels(data / "qrels/test.tsv")
    ids, corpus_hash = audit.corpus_identity(data / "corpus.jsonl")
    files = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    manifest = {"dataset": "BEIR/SCIDOCS", "archive_md5": "38121350fc3a4d2f48850f6aff52e4a9",
                "corpus_sha256": corpus_hash, "queries_sha256": audit.digest(queries), "test_qrels_sha256": audit.digest(qrels),
                "file_sha256": {name: audit.sha256(data / name) for name in files},
                "file_bytes": {name: (data / name).stat().st_size for name in files}}
    write(data / "manifest.json", manifest)
    runs = {}
    for arm in ["base", *audit.ARMS]:
        runs[arm] = {}
        for seed in (["frozen"] if arm == "base" else [str(s) for s in config["seeds"]]):
            directory = results / "scidocs" / arm / seed
            directory.mkdir(parents=True)
            rankings = {"q1": ["d1", "d2", "d3"], "q2": ["d2", "d1", "d3"]}
            if arm == "kd":
                rankings["q1"] = ["d3", "d2", "d1"]
            trec, per_path = directory / "ranking.trec", directory / "ranking.per_query.json"
            trec.write_text("".join(f"{qid} Q0 {doc} {rank} {4-rank} synthetic\n"
                                    for qid, ranking in rankings.items() for rank, doc in enumerate(ranking, 1)), encoding="utf-8")
            rows = [{"query_id": qid, **audit.calculate_metrics(ranking, qrels[qid])} for qid, ranking in rankings.items()]
            write(per_path, list(reversed(rows)))
            result = {"identity": {"spec_sha256": spec_hash, "dataset": corpus_hash, "queries_sha256": audit.digest(queries),
                                   "qrels_sha256": audit.digest(qrels), "arm": arm, "seed": seed,
                                   "weights": "base" if arm == "base" else frozen["runs"][f"{arm}/{seed}"]["selected_metadata_sha256"]},
                      "metrics": {m: sum(row[m] for row in rows) / 2 for m in audit.METRICS}, "per_query": rows,
                      "query_count": 2, "corpus_size": 3, "saved_ranking_depth": 50, "actual_depth_min": 3, "actual_depth_max": 3,
                      "ranking_scope": "all supplied documents", "ignore_identical_ids": True,
                      "tie_break": "document ID lexicographic ascending", "score_function": "normalized dot product (cosine)",
                      "query_ids_sha256": audit.digest(sorted(queries)), "document_ids_sha256": audit.digest(sorted(ids)),
                      "qrels_sha256": audit.digest(qrels), "run_filename": trec.name, "per_query_filename": per_path.name,
                      "run_sha256": audit.sha256(trec), "per_query_sha256": audit.sha256(per_path)}
            write(directory / "report.json", result)
            runs[arm][seed] = result
    write(results / "report.json", {"spec_sha256": spec_hash, "dataset": manifest, "runs": runs, "summary": reference_summary(runs)})
    return results, data, source, tmp_path


def check(audit, fixture):
    results, data, source, root = fixture
    return audit.verify(results, data, source, repo_root=root)


def change_run(audit, fixture, mutate):
    results = fixture[0]
    directory = results / "scidocs/kd/17"
    report = audit.read_json(directory / "report.json")
    mutate(report, directory)
    write(directory / "report.json", report)
    full = audit.read_json(results / "report.json")
    full["runs"]["kd"]["17"] = report
    write(results / "report.json", full)


def test_complete_synthetic_artifacts_pass_independently(audit, fixture):
    result = check(audit, fixture)
    assert result["status"] == "passed", result
    assert result["ok"] and len(result["training"]) == 12
    assert len(result["primary_paired_bootstrap"]) == 3
    assert result["metrics"]["kd"]["17"]["ndcg@10"] == .75
    assert result["primary_paired_bootstrap"]["17"]["delta"] == .25
    assert result["primary_paired_bootstrap"]["17"]["ci95_low"] == 0
    assert result["primary_paired_bootstrap"]["17"]["ci95_high"] == .5
    assert "not authenticity" in result["scope"]


def test_metric_definition_preserves_linear_grades(audit):
    result = audit.calculate_metrics(["d1", "d2"], {"d1": 1, "d2": 2})
    assert result["ndcg@10"] == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_pending_never_reads_target_or_requires_it(audit, fixture, monkeypatch):
    results, _, source, root = fixture
    (results / "report.json").unlink()
    original = audit.read_json
    def guarded(path):
        assert Path(path).is_relative_to(results)
        return original(path)
    monkeypatch.setattr(audit, "read_json", guarded)
    result = audit.verify(results, root / "nonexistent-sealed-target", source, repo_root=root)
    assert result["status"] == "pending" and not result["ok"]


@pytest.mark.parametrize("filename", ["trace.json", "selected/metadata.json", "report.json"])
def test_incomplete_training_is_pending(audit, fixture, filename):
    (fixture[0] / "training/kd/17" / filename).unlink()
    assert check(audit, fixture)["status"] == "pending"


@pytest.mark.parametrize("kind,expected", [
    ("spec", "spec hash"), ("code", "implementation SHA"), ("source", "Source split SHA"),
    ("weight", "Weight SHA"), ("trace", "trace missing"), ("gradient", "nonzero gradient"),
    ("bootstrap", "bootstrap/17"), ("mean", "Seed mean"),
])
def test_changed_artifacts_fail(audit, fixture, kind, expected):
    results, _, source, root = fixture
    if kind == "spec":
        path = results / "protocol.json"
        value = audit.read_json(path)
        value["spec"]["config"]["student_tokens"] += 1
        write(path, value)
    elif kind == "code":
        (root / "src/evidenceforge/distill_student.py").write_text("changed", encoding="utf-8")
    elif kind == "source":
        with (source / "prepared/train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
    elif kind == "weight":
        (results / "training/kd/17/selected/synthetic.safetensors").write_bytes(b"changed")
    elif kind in ("trace", "gradient"):
        path = results / "training/kd/17/trace.json"
        rows = audit.read_json(path)
        if kind == "trace":
            rows[1]["step"] = 1
        else:
            rows[1]["gradient_l2_before_clip"] = 0
        write(path, rows)
    else:
        path = results / "report.json"
        value = audit.read_json(path)
        if kind == "bootstrap":
            value["summary"]["primary"]["per_seed"]["17"]["ci95_low"] = .1
        else:
            value["summary"]["arms"]["kd"]["metrics"]["ndcg@10"]["mean"] = .1
        write(path, value)
    result = check(audit, fixture)
    assert result["status"] == "failed" and expected in result["error"], result


@pytest.mark.parametrize("kind,expected", [("duplicate", "Duplicate/missing per-query"),
                                           ("metric", "numeric mismatch"), ("ranking", "Missing/truncated ranking"),
                                           ("sha", "artifact SHA")])
def test_ranking_corruption_fails_even_if_report_copies_agree(audit, fixture, kind, expected):
    def mutate(report, directory):
        if kind == "duplicate":
            rows = [copy.deepcopy(report["per_query"][0])] * 2
            write(directory / report["per_query_filename"], rows)
            report["per_query_sha256"] = audit.sha256(directory / report["per_query_filename"])
        elif kind == "metric":
            report["metrics"]["ndcg@10"] = 0
        else:
            path = directory / report["run_filename"]
            path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
            if kind == "ranking":
                report["run_sha256"] = audit.sha256(path)
    change_run(audit, fixture, mutate)
    result = check(audit, fixture)
    assert result["status"] == "failed" and expected in result["error"], result


def test_forbidden_self_match_fails_with_matching_file_hash(audit, fixture):
    directory = fixture[0] / "scidocs/kd/17"
    result = audit.read_json(directory / "report.json")
    path = directory / result["run_filename"]
    path.write_text(path.read_text(encoding="utf-8").replace("d3", "q1"), encoding="utf-8")
    result["run_sha256"] = audit.sha256(path)
    documents = {"d1", "d2", "q1"}
    result["document_ids_sha256"] = audit.digest(sorted(documents))
    queries = {"q1": "", "q2": ""}
    qrels = audit.read_qrels(fixture[1] / "qrels/test.tsv")
    with pytest.raises(ValueError, match="forbidden self-match"):
        audit.verify_run(directory, result, result["identity"], queries, documents, qrels, 1e-10)


def test_bootstrap_reference_matches_odd_query_counts(audit):
    values = np.asarray([-.2, .1, .6])
    indices = np.random.default_rng(42).integers(3, size=(2000, 3))
    expected = np.quantile(values[indices].mean(axis=1), [.025, .975])
    actual = audit.paired_interval(values)
    assert [actual["ci95_low"], actual["ci95_high"]] == pytest.approx(expected)


def test_cli_read_only_pending_and_explicit_audit_file(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_distillation.py"
    args = [sys.executable, str(script), "--results-dir", str(tmp_path / "missing")]
    run = subprocess.run([*args, "--check"], capture_output=True, text=True, check=False)
    assert run.returncode == 2 and json.loads(run.stdout)["status"] == "pending"
    assert list(tmp_path.iterdir()) == []
    output = tmp_path / "synthetic-audit.json"
    run = subprocess.run([*args, "--output", str(output)], capture_output=True, text=True, check=False)
    assert run.returncode == 2 and json.loads(run.stdout) == json.loads(output.read_text(encoding="utf-8"))
