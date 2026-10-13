"""Pinned Qwen3 GGUF embeddings on a local CPU, with resumable document cache.

This is an independently measured modern baseline/teacher. It does not replace
the original MiniLM baseline, and candidate reranking is never labelled full
corpus retrieval. No hosted inference endpoint or API key is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

LOGGER = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B-GGUF"
MODEL_REVISION = "370f27d7550e0def9b39c1f16d3fbaa13aa67728"
MODEL_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"
MODEL_BYTES = 639_150_592
MODEL_SHA256 = "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"
RUNTIME_VERSION = "b11580"
RUNTIME_FILE = "llama-b11580-bin-win-cpu-x64.zip"
RUNTIME_BYTES = 19_509_613
RUNTIME_SHA256 = "430efe5b4f10db617ae0f5732f7e9360fa7be4344d1d97688e09bbd3b5b084da"
VULKAN_RUNTIME_FILE = "llama-b11580-bin-win-vulkan-x64.zip"
VULKAN_RUNTIME_BYTES = 32_112_251
VULKAN_RUNTIME_SHA256 = "6e520f0b8bbc6608b53d70065ed80ee90d33acf1636f886d965e0c87f708dff2"
INSTRUCTION = "Given a scientific claim, retrieve scientific abstracts that provide evidence about the claim."
DIMENSION = 1024
EOS_TOKEN_ID = 151643
MIN_START_MEMORY_BYTES = 1536 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, target: Path, size: int, digest: str) -> Path:
    """Resume interrupted downloads, validate upstream SHA256 before use."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size or _sha256(target) != digest:
            raise ValueError(f"Pinned asset integrity failure: {target}")
        return target
    # Keep an actively changing large file outside OneDrive-backed workspaces;
    # publish only the fully verified artifact into the requested cache.
    temporary_root = Path(tempfile.gettempdir()) / "evidenceforge-downloads"
    temporary_root.mkdir(parents=True, exist_ok=True)
    partial = temporary_root / f"{digest}.partial"
    legacy_partial = target.with_name(target.name + ".partial")
    if legacy_partial.exists() and not partial.exists():
        os.replace(legacy_partial, partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Oversized partial download: {partial}")
    if offset < size:
        # Full-body responses are unusually slow on this host. Four bounded
        # one-MiB ranges preserve low memory usage and allow reliable resumption.
        block_size = 1024 * 1024
        offset = offset // block_size * block_size
        download_url = url
        try:
            # Resolve the official host's CDN redirect once instead of paying
            # another HF/GitHub redirect round trip for every one-MiB range.
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
                download_url = response.geturl()
        except OSError:
            pass

        def fetch_range(start):
            end = min(start + block_size, size) - 1
            for attempt in range(3):
                request = urllib.request.Request(download_url, headers={
                    "User-Agent": "EvidenceForge/0.2", "Range": f"bytes={start}-{end}",
                })
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError(f"Invalid bounded Range response for {target.name}")
                        content = response.read(end - start + 2)
                    if len(content) != end - start + 1:
                        raise ValueError(f"Truncated bounded Range response for {target.name}")
                    return content
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (attempt + 1))

        LOGGER.info("Downloading %s from byte %s/%s (four bounded ranges)", target.name, offset, size)
        with partial.open("r+b" if partial.exists() else "wb") as output, ThreadPoolExecutor(max_workers=4) as pool:
            output.truncate(offset)
            output.seek(offset)
            while offset < size:
                starts = list(range(offset, min(offset + 4 * block_size, size), block_size))
                for content in pool.map(fetch_range, starts):
                    output.write(content)
                    offset += len(content)
                output.flush()
                LOGGER.info("%s: %.1f/%.1f MiB", target.name, offset / 2**20, size / 2**20)
    if partial.stat().st_size != size or _sha256(partial) != digest:
        raise ValueError(f"Downloaded asset checksum mismatch: {partial}")
    os.replace(partial, target)
    return target


def fetch_assets(cache_dir: Path, backend: str = "cpu") -> tuple[Path, Path]:
    """Fetch official model and separate official Windows runtimes, all pinned."""
    if backend not in {"cpu", "vulkan"}:
        raise ValueError("backend must be cpu or vulkan")
    cache_dir = Path(cache_dir)
    model = _download_verified(
        f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{MODEL_FILE}",
        cache_dir / "models" / "qwen3" / MODEL_REVISION / MODEL_FILE,
        MODEL_BYTES, MODEL_SHA256,
    )
    filename, size, digest = ((RUNTIME_FILE, RUNTIME_BYTES, RUNTIME_SHA256) if backend == "cpu" else
                              (VULKAN_RUNTIME_FILE, VULKAN_RUNTIME_BYTES, VULKAN_RUNTIME_SHA256))
    archive_path = _download_verified(
        f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}/{filename}",
        cache_dir / "runtime" / filename, size, digest,
    )
    runtime_dir = (cache_dir / "runtime" / (RUNTIME_VERSION if backend == "cpu" else RUNTIME_VERSION + "-vulkan")).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            name = PurePosixPath(item.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or ":" in str(name):
                raise ValueError(f"Unsafe runtime archive path: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Runtime archive contains a symlink")
            target = runtime_dir.joinpath(*name.parts).resolve()
            if not target.is_relative_to(runtime_dir):
                raise ValueError("Runtime extraction escapes cache")
            if name.name != "llama-server.exe" and name.suffix.lower() != ".dll":
                continue
            content = archive.read(item)
            if target.exists() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(content).digest():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    servers = list(runtime_dir.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise ValueError("Expected exactly one llama-server.exe in pinned runtime")
    return model, servers[0]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _memory_snapshot(process=None) -> dict:
    """Actual Windows memory counters; no guessed model-memory requirement."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_pagefile", "available_pagefile",
                "total_virtual", "available_virtual", "available_extended_virtual",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    result = {}
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        result.update(available_physical_bytes=status.available_physical, memory_load_percent=status.load)
    if process is not None:
        class ProcessMemory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "quota_peak_paged", "quota_paged",
                    "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile", "private_usage",
                )
            ]

        memory = ProcessMemory()
        memory.length = ctypes.sizeof(memory)
        handle = wintypes.HANDLE(int(process._handle))
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), ctypes.sizeof(memory)):
            result.update(server_working_set_bytes=memory.working_set,
                          server_peak_working_set_bytes=memory.peak_working_set,
                          server_private_bytes=memory.private_usage)
    return result


class QwenEmbeddingEncoder:
    """Owns one local llama.cpp process and stops only that owned process.

    Text is tokenized by the actual GGUF runtime before explicit right truncation.
    The official tokenizer's TemplateProcessing appends endoftext (151643),
    reflected in this pinned GGUF's add_eos_token=true/add_bos_token=false.
    We request raw tokens, truncate the body while reserving one position, then
    append that EOS explicitly before submitting a pretokenized integer array.
    `append_eos=False` is a non-default ablation, not the official recipe.
    """

    def __init__(self, cache_dir: Path, threads: int = 2, max_tokens: int = 128,
                 instruction: str = INSTRUCTION, append_eos: bool = True, backend: str = "cpu",
                 device: str = "Vulkan0"):
        if threads not in {1, 2}:
            raise ValueError("This CPU experiment permits only one or two threads")
        if not 32 <= max_tokens <= 512:
            raise ValueError("max_tokens must be between 32 and 512")
        if backend not in {"cpu", "vulkan"}:
            raise ValueError("backend must be cpu or vulkan")
        if backend == "vulkan" and not re.fullmatch(r"Vulkan[0-9]+", device):
            raise ValueError("A single explicitly enumerated Vulkan device is required")
        self.cache_dir = Path(cache_dir)
        self.threads, self.max_tokens = threads, max_tokens
        self.backend, self.device = backend, device
        self.instruction, self.append_eos = instruction, append_eos
        self.process = self.log_handle = None
        self.port = None
        self.last_stats = {}
        self.metadata = {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "model_file": MODEL_FILE, "model_sha256": MODEL_SHA256,
            "model_license": "Apache-2.0", "runtime": "llama.cpp",
            "runtime_version": RUNTIME_VERSION,
            "runtime_archive_sha256": RUNTIME_SHA256 if backend == "cpu" else VULKAN_RUNTIME_SHA256,
            "runtime_license": "MIT", "backend": backend, "threads": threads,
            "device": "none" if backend == "cpu" else device,
            "max_tokens": max_tokens, "truncation": "right after instruction formatting, reserving final EOS position when enabled",
            "pooling": "last", "normalization": "L2", "dimension": DIMENSION,
            "query_instruction": instruction, "document_instruction": None,
            "append_eos": append_eos, "eos_token_id": EOS_TOKEN_ID, "bos_added": False,
            "gguf_tokenizer_contract": {"add_eos_token": True, "add_bos_token": False, "eos_token_id": EOS_TOKEN_ID},
            "tokenize_endpoint_add_special": False, "tokenize_endpoint_parse_special": False,
            "eos_policy": ("explicit EOS after body truncation mirrors official automatic TemplateProcessing"
                           if append_eos else "EOS omitted: explicit non-default ablation"),
            "model_release": "2025-06-05", "context_size": 512,
        }

    def start(self):
        if self.process is not None:
            return self
        available = _memory_snapshot().get("available_physical_bytes")
        if available is not None and available < MIN_START_MEMORY_BYTES:
            raise MemoryError("Qwen startup requires at least 1.5 GiB free physical memory; resume after other scheduled work finishes")
        model_path, executable = fetch_assets(self.cache_dir, self.backend)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        log_dir = self.cache_dir / "modern" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"server-{self.port}-t{self.threads}.log"
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        command = [str(executable), "--model", str(model_path.resolve()),
                   "--embedding", "--pooling", "last", "--ctx-size", "512",
                   "--batch-size", "512", "--ubatch-size", "512", "--parallel", "1",
                   "--threads", str(self.threads), "--threads-batch", str(self.threads),
                   "--host", "127.0.0.1", "--port", str(self.port), "--no-warmup"]
        command += (["--device", "none", "--n-gpu-layers", "0", "--no-op-offload"] if self.backend == "cpu"
                    else ["--device", self.device, "--n-gpu-layers", "all"])
        # A public experiment report needs reproducible flags, not the user's
        # private absolute workspace or home-directory names.
        self.metadata["server_command"] = [executable.name, "--model", MODEL_FILE, *command[3:]]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(command, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                        creationflags=flags, cwd=executable.parent)
        deadline = time.monotonic() + 120
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama.cpp exited; see {self.log_path}")
                try:
                    if self._request("/health", timeout=2).get("status") == "ok":
                        return self
                except (OSError, urllib.error.URLError):
                    pass
                except RuntimeError as exc:
                    if "HTTP 503" not in str(exc):
                        raise
                time.sleep(0.25)
            raise TimeoutError(f"Model startup exceeded 120 seconds; see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def _request(self, endpoint: str, payload=None, timeout: float = 120):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.base_url + endpoint, data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"llama.cpp {endpoint} HTTP {exc.code}: {message}") from exc

    def _tokens(self, text: str, is_query: bool) -> tuple[list[int], bool, int]:
        formatted = f"Instruct: {self.instruction}\nQuery:{text}" if is_query else text
        tokens = self._request("/tokenize", {"content": formatted, "add_special": False,
                                              "parse_special": False})["tokens"]
        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Tokenizer must return a flat nonnegative integer token list")
        if not tokens:
            raise ValueError("Cannot embed empty token sequence")
        original_length = len(tokens)
        reserve = 1 if self.append_eos else 0
        tokens = tokens[:self.max_tokens - reserve]
        if self.append_eos:
            tokens = tokens + [EOS_TOKEN_ID]
        return tokens, original_length > self.max_tokens - reserve, original_length

    def encode(self, texts: list[str], batch_size: int = 1, *, is_query: bool = False,
               use_cache: bool = True, cache_namespace: str = "shared"):
        """Return [N,1024] vectors; write each completed item as a checkpoint.

        Inference remains sequential to respect the two-GB free-memory budget.
        batch_size is accepted for encoder-protocol compatibility; every item is
        checkpointed independently and it never raises request concurrency.
        `use_cache=False` measures actual encoding throughput, not cache latency.
        """
        import numpy as np

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", cache_namespace):
            raise ValueError("cache_namespace must be a simple lowercase identifier")
        signature = {k: v for k, v in self.metadata.items() if k not in {"server_command", "threads"}}
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        # The full digest stays in metadata. A checked short directory avoids
        # Windows MAX_PATH failures in long workspaces and pytest temp roots.
        item_dir = self.cache_dir / "modern" / "embeddings" / fingerprint[:24] / cache_namespace
        item_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = item_dir / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Embedding cache signature mismatch")
        output, cached, truncated, token_count, latencies = [], 0, 0, 0, []
        recorded_seconds, record_threads = 0.0, {}
        started = time.perf_counter()
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding texts must be non-empty strings")
            key = hashlib.sha256(json.dumps([is_query, text], ensure_ascii=False).encode()).hexdigest()
            cache_path = item_dir / f"{key}.json"
            if use_cache and cache_path.exists():
                record = json.loads(cache_path.read_text(encoding="utf-8"))
                cached += 1
            else:
                # Fully cached teacher outputs can be consumed during student
                # training without loading a second model into scarce RAM.
                self.start()
                if self.process is not None:
                    available = _memory_snapshot().get("available_physical_bytes")
                    if available is not None and available < 256 * 1024**2:
                        self.close()
                        raise MemoryError("Stopped owned Qwen process after free physical memory fell below 256 MiB; completed checkpoints remain reusable")
                item_started = time.perf_counter()
                tokens, was_truncated, original_length = self._tokens(text, is_query)
                response = self._request("/v1/embeddings", {"input": tokens, "encoding_format": "float"})
                vector = np.asarray(response["data"][0]["embedding"], dtype=np.float32)
                if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-9:
                    raise RuntimeError("Invalid Qwen embedding output")
                vector /= np.linalg.norm(vector)
                record = {"vector": vector.tolist(), "tokens": len(tokens), "original_tokens": original_length,
                          "truncated": was_truncated, "seconds": time.perf_counter() - item_started,
                          "threads": self.threads}
                temp_path = cache_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(record), encoding="utf-8")
                os.replace(temp_path, cache_path)
                latencies.append(record["seconds"])
            vector = np.asarray(record["vector"], dtype=np.float32)
            if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                raise ValueError(f"Corrupt Qwen embedding checkpoint: {cache_path}")
            output.append(vector)
            truncated += int(record["truncated"])
            token_count += int(record["tokens"])
            recorded_seconds += float(record["seconds"])
            thread_key = str(record.get("threads", "unknown"))
            record_threads[thread_key] = record_threads.get(thread_key, 0) + 1
        self.last_stats = {"items": len(texts), "cache_hits": cached, "truncated": truncated,
                           "tokens": token_count, "elapsed_seconds": time.perf_counter() - started,
                           "uncached_item_seconds": latencies, "embedding_fingerprint": fingerprint,
                           "recorded_original_inference_seconds": recorded_seconds,
                           "recorded_threads_item_counts": record_threads, "cache_namespace": cache_namespace}
        manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        return np.stack(output) if output else np.empty((0, DIMENSION), dtype=np.float32)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def benchmark(data_dir: Path, cache_dir: Path, output_path: Path, max_tokens: int = 128,
              include_vulkan: bool = False, vulkan_device: str = "Vulkan0"):
    """Measure 16 label-free hash-selected corpus documents at 1 and 2 threads."""
    from .data import load_dataset

    if max_tokens < 64:
        raise ValueError("The published semantic smoke test needs at least a 64-token budget")
    dataset = load_dataset(data_dir)
    selected = sorted(dataset.docs, key=lambda d: hashlib.sha256(("qwen-cpu-benchmark-v1:" + d["_id"]).encode()).hexdigest())[:16]
    texts = [doc["title"] + "\n" + doc["text"] for doc in selected]
    report = {"scope": "Local throughput microbenchmark; not a retrieval quality result",
              "selection": "16 corpus IDs with smallest SHA256(qwen-cpu-benchmark-v1:ID), no labels",
              "document_ids": [doc["_id"] for doc in selected], "corpus_sha256": dataset.corpus_hash,
              "memory_before": _memory_snapshot(), "runs": []}
    configurations = [("cpu", 1), ("cpu", 2)] + ([("vulkan", 2)] if include_vulkan else [])
    reference = None
    for backend, threads in configurations:
        startup_started = time.perf_counter()
        with QwenEmbeddingEncoder(cache_dir, threads=threads, max_tokens=max_tokens,
                                  backend=backend, device=vulkan_device) as encoder:
            startup_seconds = time.perf_counter() - startup_started
            # The published four-text example checks input/token/pooling
            # semantics independently of SciFact labels or retrieval scores.
            # Send the already formatted queries as raw text so this small
            # check uses the official web-search instruction, not ours.
            example_vectors = encoder.encode([
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity",
                "The capital of China is Beijing.",
                "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
            ], use_cache=False, cache_namespace="official_example")
            import numpy as np
            example_scores = example_vectors[:2] @ example_vectors[2:].T
            expected = np.asarray([[0.7645568251609802, 0.14142508804798126],
                                   [0.13549736142158508, 0.5999549627304077]])
            example_error = float(np.abs(example_scores - expected).max())
            if example_error > 0.08:
                raise RuntimeError(f"Qwen official example differs by {example_error:.3f}; inspect tokenization/pooling before full evaluation")
            encoder.encode(["An example scientific abstract about cellular mechanisms."], use_cache=False)
            vectors = encoder.encode(texts, use_cache=False)
            run = dict(encoder.last_stats, threads=threads, metadata=encoder.metadata,
                       output_shape=list(vectors.shape), memory=_memory_snapshot(encoder.process),
                       startup_and_integrity_check_seconds=startup_seconds,
                       official_example={"scores": example_scores.tolist(), "maximum_absolute_error": example_error,
                                         "full_precision_reference": expected.tolist(), "tolerance": 0.08,
                                         "source": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage"})
            if reference is None:
                reference = vectors.copy()
            else:
                run["cpu_reference_agreement"] = {"minimum_cosine": float(np.sum(vectors * reference, axis=1).min()),
                                                  "maximum_absolute_difference": float(np.abs(vectors - reference).max())}
            run["documents_per_second"] = 16 / run["elapsed_seconds"]
            run["estimated_5183_document_seconds"] = 5183 / run["documents_per_second"]
            report["runs"].append(run)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            LOGGER.info("Qwen backend=%s threads=%s: %.3f docs/sec", backend, threads, run["documents_per_second"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "benchmark"])
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/modern/cpu_benchmark.json"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="cpu", help="Runtime to fetch")
    parser.add_argument("--include-vulkan", action="store_true", help="Add Vulkan benchmark after both CPU arms")
    parser.add_argument("--vulkan-device", default="Vulkan0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "fetch":
        print("\n".join(map(str, fetch_assets(args.cache_dir, args.backend))))
    else:
        print(json.dumps(benchmark(args.data_dir, args.cache_dir, args.output, args.max_tokens,
                                   args.include_vulkan, args.vulkan_device), indent=2))


if __name__ == "__main__":
    main()
