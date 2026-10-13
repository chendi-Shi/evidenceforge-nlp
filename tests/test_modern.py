import hashlib
import io

import numpy as np
import pytest

from evidenceforge.modern import DIMENSION, QwenEmbeddingEncoder, _download_verified


def test_low_memory_guard_prevents_model_startup(tmp_path, monkeypatch):
    monkeypatch.setattr("evidenceforge.modern._memory_snapshot", lambda: {"available_physical_bytes": 1024})
    monkeypatch.setattr("evidenceforge.modern.fetch_assets", lambda *args: pytest.fail("Model must not be loaded"))
    encoder = QwenEmbeddingEncoder(tmp_path)
    with pytest.raises(MemoryError, match="1.5 GiB"):
        encoder.start()
    assert encoder.process is None


def test_existing_download_checksum_must_match(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity failure"):
        _download_verified("https://unused.example/file", target, 6, "0" * 64)
    assert _download_verified("https://unused.example/file", target, 6,
                              hashlib.sha256(b"broken").hexdigest()) == target


def test_download_requires_precise_content_range(tmp_path, monkeypatch):
    content = b"verified asset"

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}"}

        def geturl(self):
            return "https://unused.example/file"

    monkeypatch.setattr("evidenceforge.modern.urllib.request.urlopen", lambda *a, **k: Response(content))
    target = tmp_path / "asset"
    _download_verified("https://unused.example/file", target, len(content), hashlib.sha256(content).hexdigest())
    assert target.read_bytes() == content


def test_input_instruction_and_token_truncation_are_explicit(tmp_path):
    encoder = QwenEmbeddingEncoder(tmp_path, max_tokens=32)
    observed = []

    def request(endpoint, payload):
        observed.append((endpoint, payload))
        return {"tokens": list(range(50))}

    encoder._request = request
    tokens, truncated, original = encoder._tokens("A claim", is_query=True)
    assert tokens == list(range(31)) + [151643]
    assert truncated and original == 50
    assert observed[-1][1]["content"].endswith("\nQuery:A claim")
    assert observed[-1][1]["parse_special"] is False
    encoder._tokens("An abstract", is_query=False)
    assert observed[-1][1]["content"] == "An abstract"


def test_eos_variant_reserves_one_token(tmp_path):
    encoder = QwenEmbeddingEncoder(tmp_path, max_tokens=32, append_eos=True)
    encoder._request = lambda *args, **kwargs: {"tokens": list(range(50))}
    tokens, truncated, _ = encoder._tokens("A claim", is_query=True)
    assert len(tokens) == 32 and tokens[-1] == 151643 and truncated


@pytest.mark.parametrize("length,truncated", [(3, False), (127, False), (128, True), (200, True)])
def test_official_eos_survives_truncation_and_matches_total_budget(tmp_path, length, truncated):
    encoder = QwenEmbeddingEncoder(tmp_path, max_tokens=128)
    encoder._request = lambda *args, **kwargs: {"tokens": list(range(length))}
    actual, was_truncated, original_length = encoder._tokens("Scientific input", is_query=False)
    assert actual == list(range(min(length, 127))) + [151643]
    assert len(actual) <= 128
    assert actual.count(151643) == 1
    assert was_truncated is truncated and original_length == length
    assert encoder.metadata["gguf_tokenizer_contract"]["add_eos_token"] is True


def test_no_eos_is_an_explicit_nondefault_ablation(tmp_path):
    encoder = QwenEmbeddingEncoder(tmp_path, max_tokens=32, append_eos=False)
    encoder._request = lambda *args, **kwargs: {"tokens": list(range(40))}
    actual, truncated, _ = encoder._tokens("Scientific input", is_query=False)
    assert actual == list(range(32)) and truncated
    assert "ablation" in encoder.metadata["eos_policy"]


def test_checkpoint_cache_separates_query_and_document(tmp_path, monkeypatch):
    encoder = QwenEmbeddingEncoder(tmp_path, max_tokens=32)
    monkeypatch.setattr(encoder, "start", lambda: encoder)
    calls = []

    def request(endpoint, payload):
        calls.append(endpoint)
        if endpoint == "/tokenize":
            return {"tokens": [1, 2, 3]}
        assert payload["input"] == [1, 2, 3, 151643]
        return {"data": [{"embedding": [1.0] + [0.0] * (DIMENSION - 1)}]}

    encoder._request = request
    first = encoder.encode(["shared text"])
    assert first.shape == (1, DIMENSION)
    assert len(calls) == 2
    second = encoder.encode(["shared text"])
    assert len(calls) == 2
    assert encoder.last_stats["cache_hits"] == 1
    assert np.array_equal(first, second)
    offline = QwenEmbeddingEncoder(tmp_path, max_tokens=32)
    monkeypatch.setattr(offline, "start", lambda: pytest.fail("Cached vectors must not start the model"))
    assert np.array_equal(first, offline.encode(["shared text"]))
    encoder.encode(["shared text"], is_query=True)
    assert len(calls) == 4
    encoder.encode(["shared text"], use_cache=False)
    assert len(calls) == 6


def test_invalid_vectors_rejected_before_caching(tmp_path, monkeypatch):
    encoder = QwenEmbeddingEncoder(tmp_path)
    monkeypatch.setattr(encoder, "start", lambda: encoder)
    encoder._request = lambda endpoint, payload: (
        {"tokens": [1]} if endpoint == "/tokenize" else {"data": [{"embedding": [0.0] * DIMENSION}]}
    )
    with pytest.raises(RuntimeError, match="Invalid Qwen"):
        encoder.encode(["abstract"])
    assert not list((tmp_path / "modern" / "embeddings").glob("*/*.json"))


def test_backend_and_split_caches_are_separate(tmp_path, monkeypatch):
    calls = []
    for backend, namespace in (("cpu", "test_queries"), ("cpu", "train_queries"), ("vulkan", "test_queries")):
        encoder = QwenEmbeddingEncoder(tmp_path, backend=backend)
        monkeypatch.setattr(encoder, "start", lambda: encoder)

        def request(endpoint, payload):
            calls.append(endpoint)
            return ({"tokens": [1]} if endpoint == "/tokenize" else
                    {"data": [{"embedding": [1.0] + [0.0] * (DIMENSION - 1)}]})

        encoder._request = request
        encoder.encode(["same claim"], is_query=True, cache_namespace=namespace)
        assert encoder.last_stats["cache_hits"] == 0
    assert len(calls) == 6


def test_full_experiment_freezes_protocol_and_excludes_train_from_quality(tmp_path, monkeypatch):
    import json
    from evidenceforge.data import Dataset
    from evidenceforge import modern_experiment as experiment

    dataset = Dataset(
        docs=[{"_id": "b", "title": "B", "text": "Abstract B"},
              {"_id": "a", "title": "A", "text": "Abstract A"}],
        queries={"test": "Test claim", "train": "Train claim"},
        qrels={"test": {"test": {"a": 1}}, "train": {"train": {"b": 1}}},
    )
    monkeypatch.setattr(experiment, "EXPECTED_DOCUMENTS", 2)
    monkeypatch.setattr(experiment, "EXPECTED_TEST_QUERIES", 1)
    monkeypatch.setattr(experiment, "EXPECTED_TRAIN_QUERIES", 1)
    monkeypatch.setattr(experiment, "load_dataset", lambda _: dataset)
    cache, output = tmp_path / "cache", tmp_path / "results"
    monkeypatch.setattr(QwenEmbeddingEncoder, "start", lambda self: pytest.fail("Preparation must not load a model"))
    prepared = experiment.run_modern_experiment(tmp_path, cache, output, max_tokens=32,
                                                 encode_extra_train=True, prepare_only=True)
    assert prepared["spec"]["document_count"] == 2
    assert not (output / "report.json").exists()
    monkeypatch.setattr(QwenEmbeddingEncoder, "start", lambda self: self)
    monkeypatch.setattr(QwenEmbeddingEncoder, "_request", lambda self, endpoint, payload: (
        {"tokens": [1, 2]} if endpoint == "/tokenize" else
        {"data": [{"embedding": [1.0] + [0.0] * (DIMENSION - 1)}]}
    ))
    result = experiment.run_modern_experiment(tmp_path, cache, output, max_tokens=32, encode_extra_train=True)
    assert result["result"]["query_count"] == 1
    assert result["result"]["per_query"][0]["query_id"] == "test"
    assert set(result["encoding"]) == {"documents", "test_queries", "train_queries"}
    artifact = result["teacher_vector_artifact"]
    assert artifact["test_query_vectors_included"] is False
    with np.load(cache / artifact["path_relative_to_cache"], allow_pickle=False) as vectors:
        assert vectors["document_ids"].tolist() == ["a", "b"]
        assert vectors["train_query_ids"].tolist() == ["train"]
        assert set(vectors.files) == {"document_ids", "document_vectors", "train_query_ids", "train_query_vectors"}
    protocol_before = (output / "protocol.json").read_bytes()
    resumed = experiment.run_modern_experiment(tmp_path, cache, output, max_tokens=32, encode_extra_train=True)
    assert resumed["encoding"]["documents"]["cache_hits"] == 2
    assert resumed["encoding"]["test_queries"]["cache_hits"] == 1
    assert (output / "protocol.json").read_bytes() == protocol_before
    assert json.loads(protocol_before)["spec"]["extra_train_encoding"]["ordered_query_ids"] == ["train"]
    with pytest.raises(ValueError, match="Frozen Qwen protocol differs"):
        experiment.run_modern_experiment(tmp_path, cache, output, max_tokens=64, encode_extra_train=True)
