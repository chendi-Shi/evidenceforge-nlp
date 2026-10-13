import json
from types import SimpleNamespace

import numpy as np
import pytest

from evidenceforge.ranker import EvidenceRanker, extract_features, feature_names, grouped_train_validation_split


def test_claim_and_evidence_components_cannot_cross_split():
    queries = {"a": "Dose 10 improves recovery.", "b": "Dose 20 improves recovery!", "c": "Independent sentence", "d": "Other claim", "e": "Fifth claim"}
    qrels = {"a": {"d1": 1}, "b": {"d2": 1}, "c": {"d2": 1}, "d": {"d3": 1}, "e": {"d4": 1}}
    train, validation, metadata = grouped_train_validation_split(queries, qrels)
    assert (set("abc") <= set(train)) or (set("abc") <= set(validation))
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(queries)
    assert metadata["shared_positive_documents"] == metadata["shared_normalized_claims"] == 0
    assert (train, validation, metadata) == grouped_train_validation_split(queries, qrels)


def test_one_connected_component_fails_instead_of_leaking():
    with pytest.raises(ValueError, match="one evidence-connected group"):
        grouped_train_validation_split({"a": "a", "b": "b"}, {"a": {"same": 1}, "b": {"same": 1}})


def test_ablation_contains_no_dense_or_fusion_proxy():
    hit = {"doc_id": "a", "title": "Immune cells", "text": "Cells respond to treatment.", "bm25_score": 3.0,
           "bm25_rank": 4, "dense_score": 0.2, "dense_rank": 9, "rrf_score": 0.03}
    assert len(extract_features("Immune treatment", hit)) == len(feature_names())
    original = extract_features("Immune treatment", hit, include_dense=False)
    changed = dict(hit, dense_score=0.9, dense_rank=1, rrf_score=0.9)
    assert original == extract_features("Immune treatment", changed, include_dense=False)
    assert not any("dense" in name or "rrf" in name for name in feature_names(False))
    assert np.isfinite(extract_features("", {})).all()
    assert extract_features("unseen", dict(hit, bm25_score=0.0))[1] == 0.0


class FakeIndex:
    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = []
        self.metadata = {"corpus_sha256": "test-corpus", "model_revision": "test-revision"}

    def search(self, query, method="rrf", k=50):
        self.calls.append((query, k))
        return [{"doc_id": doc["_id"], "title": doc["title"], "text": doc["text"], "bm25_score": 1.0,
                 "bm25_rank": i + 1, "dense_score": 0.1, "dense_rank": i + 1, "rrf_score": 1 / (61 + i)}
                for i, doc in enumerate(self.dataset.docs[:k])]


def fixture_dataset():
    docs = [{"_id": f"d{i}", "title": f"Evidence {i}", "text": "Scientific result with measured effects"} for i in range(60)]
    queries = {f"q{i}": f"Unique claim about word{chr(97 + i)}" for i in range(12)}
    qrels = {"train": {f"q{i}": {f"d{48 + i}": 1} for i in range(12)}, "test": {"test": {"d0": 1}}}
    return SimpleNamespace(docs=docs, queries=queries, qrels=qrels)


def test_training_never_reads_test_and_roundtrip_preserves_predictions(tmp_path):
    dataset = fixture_dataset()
    # A fail-on-access test dictionary demonstrates train fitting cannot inspect test labels.
    class TrainOnly(dict):
        def __getitem__(self, key):
            if key != "train":
                raise AssertionError("Test split accessed during fitting")
            return super().__getitem__(key)
    dataset.qrels = TrainOnly(dataset.qrels)
    index = FakeIndex(dataset)
    model = EvidenceRanker()
    metadata = model.fit(dataset, index, threads=1)
    assert metadata["positive_injection_count"] == 10
    assert metadata["final_fit_query_count"] == 12
    natural_pool = index.search(dataset.queries["q0"])
    before = model.rerank(dataset.queries["q0"], natural_pool)
    assert len(before) == 50
    model.save(tmp_path)
    loaded = EvidenceRanker.load(tmp_path)
    loaded.validate_index(index.metadata)
    with pytest.raises(ValueError, match="corpus_sha256"):
        loaded.validate_index(dict(index.metadata, corpus_sha256="different"))
    with pytest.raises(ValueError, match="model_revision"):
        loaded.validate_index(dict(index.metadata, model_revision="different"))
    after = loaded.rerank(dataset.queries["q0"], natural_pool)
    assert [hit["doc_id"] for hit in before] == [hit["doc_id"] for hit in after]
    assert [hit["ltr_score"] for hit in before] == pytest.approx([hit["ltr_score"] for hit in after])
    saved = json.loads((tmp_path / "metadata.json").read_text())
    correct_metadata = dict(saved)
    saved["feature_names"] = list(reversed(saved["feature_names"]))
    (tmp_path / "metadata.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="feature schema"):
        EvidenceRanker.load(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps(correct_metadata))
    with (tmp_path / "model.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        EvidenceRanker.load(tmp_path)
