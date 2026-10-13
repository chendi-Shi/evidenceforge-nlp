import json
from dataclasses import asdict

import numpy as np
import pytest

from evidenceforge.data import Dataset
from evidenceforge.neural_experiment import cached_variants, evaluate_vectors


def test_full_corpus_ranking_preserves_grades_and_ties(tmp_path):
    dataset = Dataset([{"_id": key, "text": key} for key in ["c", "b", "a"]], {"q": "test"},
                      {"test": {"q": {"b": 2, "c": 1}}})
    # Rows must follow canonical document-ID order, regardless of input order.
    vectors = np.array([[-1., 0.], [1., 0.], [0., 1.]])
    result = evaluate_vectors(dataset, vectors, np.array([[1., .5]]), ["q"], tmp_path, "model")
    assert result["metrics"]["ndcg@10"] == 1.0
    assert result["metrics"]["recall@50"] == 1.0
    assert [line.split()[2] for line in (tmp_path / "model.trec").read_text().splitlines()] == ["b", "c", "a"]
    evaluate_vectors(dataset, np.zeros((3, 2)), np.zeros((1, 2)), ["q"], tmp_path, "tie")
    assert [line.split()[2] for line in (tmp_path / "tie.trec").read_text().splitlines()] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_vectors(dataset, vectors, np.ones((2, 2)), ["q", "q"], tmp_path, "bad")


class CountingEncoder:
    def __init__(self):
        self.metadata = {"model_sha256": "test-model-only"}
        self.calls = 0

    def encode(self, texts, batch_size):
        self.calls += 1
        return {"base": np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))}


def test_chunk_resume_identity_and_corruption_detection(tmp_path):
    encoder = CountingEncoder()
    texts = ["first", "second", "third"]
    result = cached_variants(encoder, texts, tmp_path, chunk_size=2)
    assert result["base"].shape == (3, 2)
    assert encoder.calls == 2
    cached_variants(encoder, texts, tmp_path, chunk_size=2)
    assert encoder.calls == 2
    encoder.metadata["model_sha256"] = "different-adapter"
    cached_variants(encoder, texts, tmp_path, chunk_size=2)
    assert encoder.calls == 4
    encoder.metadata["model_sha256"] = "test-model-only"
    for directory in tmp_path.iterdir():
        if json.loads((directory / "manifest.json").read_text())["encoder"] == encoder.metadata:
            (directory / "0000000.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupt"):
        cached_variants(encoder, texts, tmp_path, chunk_size=2)


def prepared_protocol(tmp_path, monkeypatch):
    from evidenceforge import neural_experiment as module
    dataset = Dataset([{"_id": "doc", "title": "a", "text": "b"}], {"train": "source", "test": "target"},
                      {"train": {"train": {"doc": 1}}, "test": {"test": {"doc": 1}}})
    base = tmp_path / "base"
    base.mkdir()
    (base / "assets_manifest.json").write_text('{}')
    monkeypatch.setattr(module, "download_float_minilm", lambda path: base)
    monkeypatch.setattr(module, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(module, "download_external", lambda name, path: dataset)
    output = tmp_path / "result"
    arguments = (tmp_path / "source", tmp_path / "transfer", tmp_path / "cache", output)
    module.run_adaptation_experiment(*arguments, stage="prepare")
    return module, dataset, arguments


def test_frozen_protocol_rejects_changed_test_query(tmp_path, monkeypatch):
    module, dataset, arguments = prepared_protocol(tmp_path, monkeypatch)
    dataset.queries["test"] = "changed evaluation input"
    with pytest.raises(ValueError, match="protocol differs"):
        module.run_adaptation_experiment(*arguments, stage="prepare")


def test_evaluate_rejects_adapter_from_different_training_budget(tmp_path, monkeypatch):
    from evidenceforge.adaptation import AdaptationConfig
    module, dataset, arguments = prepared_protocol(tmp_path, monkeypatch)
    output = arguments[-1]
    config = asdict(AdaptationConfig(epochs=2, max_train_queries=10000))
    plan = {"config": config, "corpus_sha256": dataset.corpus_hash, "train_query_ids": ["train"],
            "external_teacher_scores": False,
            "selected_train_qrels_sha256": module.digest(dataset.qrels["train"]),
            "selected_train_queries_sha256": module.digest({"train": "source"})}
    plan["plan_sha256"] = module.digest(plan)
    module.write_json(output / "training" / "training_plan.json", plan)
    for arm in ("random", "retrieved", "filtered"):
        module.write_json(output / "training" / arm / "metadata.json", {
            "config": dict(config, epochs=3), "identity": {"plan_sha256": plan["plan_sha256"]}, "arm": arm})
    with pytest.raises(ValueError, match="Adapter training identity"):
        module.run_adaptation_experiment(*arguments, stage="evaluate")
