import math

import numpy as np
import pytest

from evidenceforge.data import Dataset
from evidenceforge.retrieval import RetrievalIndex


def dataset():
    return Dataset([
        {"_id": "b", "title": "", "text": "apple"},
        {"_id": "a", "title": "", "text": "apple apple"},
        {"_id": "c", "title": "", "text": "banana"},
    ], {}, {})


def test_bm25_matches_hand_calculated_length_normalization(tmp_path):
    index = RetrievalIndex(dataset(), tmp_path, dense=False).build()
    hits = index.search("apple", method="bm25", k=3)
    assert [hit["doc_id"] for hit in hits] == ["a", "b"]
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_a = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / (4 / 3)))
    assert hits[0]["bm25_score"] == pytest.approx(expected_a)
    assert all(hit["bm25_score"] > 0 for hit in hits)
    assert hits[0]["dense_score"] is None


def test_oov_does_not_return_arbitrary_lexical_documents(tmp_path):
    index = RetrievalIndex(dataset(), tmp_path, dense=False).build()
    assert index.search("unseen", method="bm25", k=100) == []


def test_positive_ties_are_deterministic_by_document_id(tmp_path):
    data = dataset()
    for doc in data.docs:
        doc["text"] = "apple"
    index = RetrievalIndex(data, tmp_path, dense=False).build()
    assert [hit["doc_id"] for hit in index.search("apple", method="bm25", k=100)] == ["a", "b", "c"]


def test_no_silent_dense_fallback(tmp_path):
    index = RetrievalIndex(dataset(), tmp_path, dense=False).build()
    for method in ("dense", "rrf"):
        with pytest.raises(ValueError, match="requires dense=True"):
            index.search("apple", method=method)


@pytest.mark.parametrize("query", ["", "  ", "?!", None])
def test_invalid_queries_rejected(tmp_path, query):
    index = RetrievalIndex(dataset(), tmp_path, dense=False).build()
    with pytest.raises(ValueError, match="Query"):
        index.search(query, method="bm25")


class FakeEncoder:
    calls = []

    def __init__(self, cache_dir, threads):
        self.metadata = {"model_id": "fake-test-encoder", "model_revision": "test", "dimension": 2, "threads": threads}

    def encode(self, texts):
        self.calls.append(texts)
        if len(texts) == 1:
            return np.asarray([[0, 1]], dtype=np.float32)
        return np.asarray([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)


def test_dense_rrf_use_full_ranks_and_do_not_read_labels(tmp_path, monkeypatch):
    monkeypatch.setattr("evidenceforge.retrieval.MiniLMEncoder", FakeEncoder)
    FakeEncoder.calls = []
    data = dataset()
    data.qrels = {"test": {"secret_query": {"b": 1}}}
    index = RetrievalIndex(data, tmp_path).build()
    dense = index.search("apple", method="dense", k=3)
    assert [hit["doc_id"] for hit in dense] == ["c", "b", "a"]
    fused = index.search("apple", method="rrf", k=3)
    # Zero-score lexical document c must not receive an arbitrary BM25 rank vote.
    assert [hit["doc_id"] for hit in fused] == ["a", "b", "c"]
    for hit in fused:
        lexical_vote = 1 / (60 + hit["bm25_rank"]) if hit["bm25_score"] > 0 else 0
        assert hit["rrf_score"] == pytest.approx(lexical_vote + 1 / (60 + hit["dense_rank"]))
    oov = index.search("unseen", method="rrf", k=3)
    assert [hit["doc_id"] for hit in oov] == ["c", "b", "a"]
    calls_before = len(FakeEncoder.calls)
    lexical = index.search("apple", method="bm25", k=3)
    assert len(FakeEncoder.calls) == calls_before
    assert all(hit["dense_score"] is None for hit in lexical)


def test_embedding_cache_reused_only_for_matching_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("evidenceforge.retrieval.MiniLMEncoder", FakeEncoder)
    FakeEncoder.calls = []
    first = RetrievalIndex(dataset(), tmp_path).build()
    second = RetrievalIndex(dataset(), tmp_path).build()
    assert len(FakeEncoder.calls) == 1
    assert first.metadata["embedding_cache_key"] == second.metadata["embedding_cache_key"]
    changed = dataset()
    changed.docs[0]["text"] = "new evidence"
    third = RetrievalIndex(changed, tmp_path).build()
    assert len(FakeEncoder.calls) == 2
    assert first.metadata["embedding_cache_key"] != third.metadata["embedding_cache_key"]


def test_corrupt_embedding_cache_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("evidenceforge.retrieval.MiniLMEncoder", FakeEncoder)
    RetrievalIndex(dataset(), tmp_path).build()
    cache = next(tmp_path.glob("embeddings-*.npy"))
    np.save(cache, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="not L2-normalized"):
        RetrievalIndex(dataset(), tmp_path).build()


def test_existing_model_asset_must_match_pinned_hash(tmp_path, monkeypatch):
    from evidenceforge import neural
    target = tmp_path / "tokenizer.json"
    target.write_bytes(b"bad")
    monkeypatch.setitem(neural.EXPECTED_ASSETS, "tokenizer.json", ("sha256", "0" * 64, 3))
    with pytest.raises(ValueError, match="checksum mismatch"):
        neural._asset(tmp_path, "tokenizer.json")
