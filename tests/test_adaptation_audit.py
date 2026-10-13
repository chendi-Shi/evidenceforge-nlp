"""Synthetic audit artifacts exercise the acceptance gate without loading models."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def audit_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("adaptation_audit_under_test", scripts / "verify_adaptation_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path, audit_module):
    """Two small datasets with known perfect rankings and fake adapter bytes."""
    results = tmp_path / "results"
    source, transfer = tmp_path / "scifact", tmp_path / "nfcorpus"
    for data_dir in (source, transfer):
        (data_dir / "qrels").mkdir(parents=True)
        (data_dir / "corpus.jsonl").write_text(
            '{"_id":"d1","title":"A","text":"first"}\n'
            '{"_id":"d2","title":"B","text":"second"}\n', encoding="utf-8",
        )
        (data_dir / "qrels/test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq2\td2\t1\n", encoding="utf-8",
        )
    config = {"batch_size": 2, "epochs": 2, "max_length": 128, "seed": 42}
    base = {"model_id": "synthetic-base", "base_model_sha256": "base-weights",
            "config_sha256": "base-config", "tokenizer_sha256": "base-tokenizer", "max_length": 128}
    spec = {"training": config, "source_corpus_sha256": "source-corpus",
            "base_assets": {"model_id": base["model_id"], "sha256": {
                "model.safetensors": base["base_model_sha256"], "config.json": base["config_sha256"],
                "tokenizer.json": base["tokenizer_sha256"],
            }}}
    protocol = {"spec": spec, "spec_sha256": audit_module.canonical_digest(spec)}
    plan = {"config": config, "corpus_sha256": "source-corpus", "train_query_ids": ["train"],
            "validation_query_ids": ["validation"], "arms": {
                arm: [{"query_id": "train", "positive_doc_id": "d1", "negative_doc_ids": ["d2"]}]
                for arm in ("random", "retrieved", "filtered")
            }}
    plan["plan_sha256"] = audit_module.canonical_digest(plan)
    write_json(results / "protocol.json", protocol)
    write_json(results / "training/training_plan.json", plan)
    protocol_hash = audit_module.sha(results / "protocol.json")
    frozen = {"protocol_sha256": protocol_hash, "adapters": {}}
    for arm in ("random", "retrieved", "filtered"):
        directory = results / "training" / arm
        directory.mkdir(parents=True)
        (directory / "adapter.safetensors").write_bytes(f"synthetic adapter {arm}".encode())
        adapter_hash = audit_module.sha(directory / "adapter.safetensors")
        metadata = {
            "arm": arm, "config": copy.deepcopy(config), "base_encoder": copy.deepcopy(base),
            "identity": {"arm": arm, "config_sha256": audit_module.canonical_digest(config),
                         "plan_sha256": plan["plan_sha256"], "base_encoder": copy.deepcopy(base),
                         "initial_adapter_sha256": "same-initialization"},
            "adapter_sha256": adapter_hash, "training_query_count": 1, "validation_query_count": 1,
            "completed_steps": 2, "parameter_delta_l2": 1.0, "nonzero_gradient_steps": 2,
            "initial_trainable_sha256": "same-initialization", "final_trainable_sha256": f"changed-{arm}",
        }
        write_json(directory / "metadata.json", metadata)
        frozen["adapters"][arm] = adapter_hash
    write_json(results / "frozen_adapters.json", frozen)
    report = {"protocol_sha256": protocol_hash, "spec_sha256": protocol["spec_sha256"],
              "datasets": {}, "comparisons": {}}
    metrics = {metric: 1.0 for metric in audit_module.METRICS}
    for name in ("scifact", "nfcorpus"):
        methods = {}
        for arm in ("base", "random", "retrieved", "filtered"):
            path = results / name / "runs" / f"{arm}.trec"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"q1 Q0 d1 1 1.0 {arm}\nq1 Q0 d2 2 0.5 {arm}\n"
                            f"q2 Q0 d2 1 1.0 {arm}\nq2 Q0 d1 2 0.5 {arm}\n", encoding="utf-8")
            methods[arm] = {"run_sha256": audit_module.sha(path), "corpus_size": 2, "query_count": 2,
                            "metrics": dict(metrics), "per_query": [{"query_id": qid, **metrics} for qid in ("q1", "q2")]}
        report["datasets"][name] = {"methods": methods}
        report["comparisons"][name] = {pair: {"delta": 0.0} for pair in (
            "filtered_minus_retrieved", "random_minus_base", "retrieved_minus_base", "filtered_minus_base",
        )}
    write_json(results / "report.json", report)
    return results, source, transfer


def modify(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    write_json(path, value)


def test_valid_synthetic_artifacts_pass_without_model_dependencies(artifacts, audit_module):
    before = set(sys.modules)
    result = audit_module.verify(*artifacts)
    assert result["ok"], result["errors"]
    assert result["aggregate_metric_count"] == 32
    assert not ({"torch", "transformers", "safetensors"} & (set(sys.modules) - before))


def test_canonical_spec_is_recomputed_even_if_outer_hashes_are_updated(artifacts, audit_module):
    results, *_ = artifacts
    modify(results / "protocol.json", lambda p: p["spec"].update(source_corpus_sha256="changed"))
    new_file_hash = audit_module.sha(results / "protocol.json")
    for path in (results / "report.json", results / "frozen_adapters.json"):
        modify(path, lambda p: p.update(protocol_sha256=new_file_hash))
    assert "Protocol spec hash mismatch" in audit_module.verify(*artifacts)["errors"]


def test_training_plan_payload_must_match_its_declared_hash(artifacts, audit_module):
    results, *_ = artifacts
    modify(results / "training/training_plan.json", lambda p: p.update(added_untracked_decision=True))
    errors = audit_module.verify(*artifacts)["errors"]
    assert "Training plan content hash mismatch" in errors
    assert any("training plan identity mismatch" in error for error in errors)


def test_rehashed_plan_config_cannot_differ_from_protocol(artifacts, audit_module):
    results, *_ = artifacts

    def change(plan):
        plan["config"]["epochs"] = 3
        plan["plan_sha256"] = audit_module.canonical_digest({key: value for key, value in plan.items() if key != "plan_sha256"})

    modify(results / "training/training_plan.json", change)
    assert "Training plan/protocol config mismatch" in audit_module.verify(*artifacts)["errors"]


@pytest.mark.parametrize("field,expected", [
    ("initialization", "same initialization"),
    ("base", "same base encoder identity"),
    ("plan", "training plan identity mismatch"),
    ("config", "training config differs from protocol"),
])
def test_training_arms_cannot_silently_use_different_identities(artifacts, audit_module, field, expected):
    results, *_ = artifacts

    def change(metadata):
        if field == "initialization":
            metadata["initial_trainable_sha256"] = "different-initialization"
            metadata["identity"]["initial_adapter_sha256"] = "different-initialization"
        elif field == "base":
            metadata["base_encoder"]["base_model_sha256"] = "different-weights"
            metadata["identity"]["base_encoder"]["base_model_sha256"] = "different-weights"
        elif field == "plan":
            metadata["identity"]["plan_sha256"] = "different-plan"
        else:
            metadata["config"]["epochs"] = 3
            metadata["identity"]["config_sha256"] = audit_module.canonical_digest(metadata["config"])

    modify(results / "training/filtered/metadata.json", change)
    assert any(expected in error for error in audit_module.verify(*artifacts)["errors"])


def test_primary_comparison_cannot_be_omitted(artifacts, audit_module):
    results, *_ = artifacts
    modify(results / "report.json", lambda p: p["comparisons"]["nfcorpus"].pop("filtered_minus_retrieved"))
    assert "nfcorpus: missing/unexpected paired comparison" in audit_module.verify(*artifacts)["errors"]


def test_duplicate_query_rows_cannot_hide_an_omitted_query(artifacts, audit_module):
    results, *_ = artifacts

    def duplicate(report):
        rows = report["datasets"]["scifact"]["methods"]["base"]["per_query"]
        rows[1] = dict(rows[0])

    modify(results / "report.json", duplicate)
    assert "scifact/base: per-query coverage or uniqueness mismatch" in audit_module.verify(*artifacts)["errors"]


def test_adapter_bytes_are_bound_to_the_frozen_run(artifacts, audit_module):
    results, *_ = artifacts
    (results / "training/random/adapter.safetensors").write_bytes(b"changed")
    assert "random: adapter hash mismatch" in audit_module.verify(*artifacts)["errors"]
