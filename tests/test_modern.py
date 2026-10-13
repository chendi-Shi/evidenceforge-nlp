import hashlib
import io

import numpy as np
import pytest

from evidenceforge.modern import DIMENSION, QwenEmbeddingEncoder, _download_verified


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
    assert tokens == list(range(32))
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


def test_checkpoint_cache_separates_query_and_document(tmp_path, monkeypatch):
    encoder = QwenEmbeddingEncoder(tmp_path, max_tokens=32)
    monkeypatch.setattr(encoder, "start", lambda: encoder)
    calls = []

    def request(endpoint, payload):
        calls.append(endpoint)
        if endpoint == "/tokenize":
            return {"tokens": [1, 2, 3]}
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
