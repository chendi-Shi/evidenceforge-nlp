"""CPU MiniLM sentence embeddings via pinned, quantized ONNX weights."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

LOGGER = logging.getLogger(__name__)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Resolved from https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2
# on 2026-10-13. An immutable revision is used for every asset, never `main`.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
MAX_LENGTH = 256
EMBEDDING_DIMENSION = 384
# From the official HF tree API at MODEL_REVISION; ONNX uses the LFS object
# digest and tokenizer.json uses Git's blob digest (header + file contents).
EXPECTED_ASSETS = {
    MODEL_FILE: ("sha256", "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21", 23_046_789),
    "tokenizer.json": ("git-sha1", "cb202bfe2e3c98645018a6d12f182a434c9d3e02", 466_247),
}


def _verify_asset(path: Path, filename: str) -> None:
    algorithm, expected, expected_size = EXPECTED_ASSETS[filename]
    if path.stat().st_size != expected_size:
        raise ValueError(f"Model asset size mismatch for {path}; remove it and retry")
    content = path.read_bytes()
    if algorithm == "sha256":
        actual = hashlib.sha256(content).hexdigest()
    else:
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if actual != expected:
        raise ValueError(f"Model asset checksum mismatch for {path}; remove it and retry")


def _asset(cache_dir: Path, filename: str) -> Path:
    target = cache_dir / filename
    if target.exists() and target.stat().st_size:
        _verify_asset(target, filename)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{filename}"
    LOGGER.info("Downloading pinned model asset %s", filename)
    request = urllib.request.Request(url, headers={"User-Agent": "EvidenceForge/0.1"})
    fd, temp_name = tempfile.mkstemp(prefix=".download-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=90) as response:
            while block := response.read(1024 * 1024):
                output.write(block)
                if output.tell() > EXPECTED_ASSETS[filename][2]:
                    raise ValueError(f"Downloaded asset exceeds expected size: {filename}")
        _verify_asset(Path(temp_name), filename)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


class MiniLMEncoder:
    """No hosted API or PyTorch dependency; downloaded assets run locally."""

    def __init__(self, cache_dir: Path, threads: int = 2, max_length: int = MAX_LENGTH):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Dense search requires onnxruntime and tokenizers; install the project dependencies") from exc
        if threads < 1 or not 2 <= max_length <= 512:
            raise ValueError("threads must be positive and max_length must be in [2, 512]")
        self.cache_dir = Path(cache_dir) / "models" / MODEL_REVISION
        model_path = _asset(self.cache_dir, MODEL_FILE)
        tokenizer_path = _asset(self.cache_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.metadata = {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "model_file": MODEL_FILE, "max_length": max_length,
            "pooling": "attention-mask-mean", "normalize": True,
            "dimension": EMBEDDING_DIMENSION, "backend": "onnxruntime-cpu", "threads": threads,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "model_license": "Apache-2.0",
        }
        (self.cache_dir / "manifest.json").write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch(texts[start:start + batch_size])
            inputs = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
            }
            outputs = self.session.run(None, {key: value for key, value in inputs.items() if key in self.input_names})
            tokens = outputs[0]
            expected_prefix = inputs["input_ids"].shape
            if tokens.ndim != 3 or tokens.shape[:2] != expected_prefix or tokens.shape[2] != EMBEDDING_DIMENSION:
                raise RuntimeError(f"Expected token embeddings from pinned MiniLM, got {tokens.shape}")
            if not np.isfinite(tokens).all():
                raise RuntimeError("MiniLM returned non-finite token embeddings")
            mask = inputs["attention_mask"][..., None].astype(np.float32)
            vectors = (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms < 1e-12):
                raise RuntimeError("MiniLM returned a zero-norm sentence embedding")
            vectors /= norms
            batches.append(vectors.astype(np.float32))
            if len(texts) > batch_size and (start // batch_size) % 25 == 0:
                LOGGER.info("Embedding documents: %s/%s", min(start + batch_size, len(texts)), len(texts))
        return np.concatenate(batches, axis=0)
