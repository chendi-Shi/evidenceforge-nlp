"""Offline integration smoke against real evaluated SciFact/LoRA artifacts.

Run after the complete adaptation experiment. This is a small correctness check,
not a load test. No model or dataset downloads, training, or corpus encoding are
allowed; the FastAPI lifespan still loads the real model and builds the index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
from time import perf_counter
from unittest.mock import patch
import urllib.request


# The first three numeric public test IDs, fixed before inspecting their ranks.
QUERY_IDS = ("1", "3", "5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_digest(value) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def read_reference(path: Path, document_ids: set[str]) -> dict:
    selected = {qid: [] for qid in QUERY_IDS}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        require(len(fields) == 6, f"Invalid TREC row {line_number}")
        qid, _, doc_id, rank, score, method = fields
        if qid in selected:
            require(method == "retrieved", f"Unexpected TREC method: {method}")
            require(doc_id in document_ids, f"Unknown document: {doc_id}")
            require(math.isfinite(float(score)), "Nonfinite reference score")
            selected[qid].append((int(rank), doc_id, float(score)))
    for qid, rows in selected.items():
        rows.sort()
        require([row[0] for row in rows] == list(range(1, 51)), f"Incomplete TREC ranks: {qid}")
        require(len({row[1] for row in rows}) == 50, f"Repeated TREC documents: {qid}")
    return selected


def verify_evaluated_cache(dataset, report: dict, cache_dir: Path, cache_path: Path) -> int:
    """Prove the service cache equals the hash-checked evaluation corpus chunks."""
    import numpy as np

    query_ids = sorted(qid for qid, rels in dataset.qrels["test"].items() if any(v > 0 for v in rels.values()))
    texts = [doc["title"] + "\n" + doc["text"] for doc in dataset.docs]
    texts.extend(dataset.queries[qid] for qid in query_ids)
    chunk_size = 64
    identity = canonical_digest({"encoder": report["encoder"], "texts": texts, "chunk_size": chunk_size})
    directory = cache_dir / "adaptation_vectors" / identity
    manifest = read_json(directory / "manifest.json")
    require(manifest == {"identity": identity, "encoder": report["encoder"],
                         "texts": len(texts), "chunk_size": chunk_size}, "Evaluation cache manifest mismatch")
    values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(dataset.docs), report["encoder"]["retrieved"]["dimension"])
    require(values.shape == expected_shape and values.dtype == np.float32, "Invalid service cache shape/dtype")
    require(bool(np.isfinite(values).all()), "Nonfinite service vectors")
    require(bool(np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4)), "Service vectors are not normalized")
    chunks_checked = 0
    for start in range(0, len(dataset.docs), chunk_size):
        path = directory / f"{start:07d}.npz"
        require(sha256(path) == path.with_suffix(".sha256").read_text().strip(), f"Corrupt chunk: {path.name}")
        count = min(chunk_size, len(dataset.docs) - start)
        with np.load(path, allow_pickle=False) as chunk:
            require(np.array_equal(values[start:start + count], chunk["retrieved"][:count]),
                    f"Service cache differs from evaluated vectors at offset {start}")
        chunks_checked += 1
    del values
    return chunks_checked


def run(args, result: dict) -> None:
    from evidenceforge.data import load_dataset
    from evidenceforge.model_assets import FLOAT_ASSETS, verify_float_asset
    from evidenceforge.neural import MODEL_REVISION
    from evidenceforge.retrieval import RetrievalIndex, embedding_cache_key

    started = perf_counter()
    base_dir = args.cache_dir / "models" / f"{MODEL_REVISION}-float"
    for name in FLOAT_ASSETS:
        require((base_dir / name).is_file(), f"Missing local base asset {name}; downloads are disabled")
    base_hashes = {name: verify_float_asset(base_dir / name, name) for name in FLOAT_ASSETS}
    adapter = read_json(args.adapter_dir / "metadata.json")
    adapter_hash = sha256(args.adapter_dir / "adapter.safetensors")
    require(adapter["arm"] == "retrieved", "This smoke requires the predeclared retrieved adapter")
    require(adapter_hash == adapter["adapter_sha256"], "Adapter byte hash mismatch")
    require(adapter["base_encoder"]["base_model_sha256"] == base_hashes["model.safetensors"],
            "Adapter base model does not match verified local assets")
    report = read_json(args.results_dir / "report.json")
    require(set(report["datasets"]) == {"scifact", "nfcorpus"}, "Wait for both full adaptation evaluations")
    protocol_path = args.results_dir / "protocol.json"
    protocol = read_json(protocol_path)
    frozen = read_json(args.results_dir / "frozen_adapters.json")
    require(report["protocol_sha256"] == frozen["protocol_sha256"] == sha256(protocol_path),
            "Protocol byte hash mismatch")
    require(report["spec_sha256"] == protocol["spec_sha256"] == canonical_digest(protocol["spec"]),
            "Protocol spec hash mismatch")
    require(frozen["adapters"]["retrieved"] == adapter_hash, "Adapter is not the frozen evaluated artifact")
    expected_encoder = report["encoder"]["retrieved"]
    require(expected_encoder["adapter_sha256"] == adapter_hash, "Report adapter identity mismatch")
    dataset = load_dataset(args.data_dir)
    require(len(dataset.docs) == 5183 and len(dataset.qrels["test"]) == 300, "Expected full BEIR SciFact")
    corpus_hash = dataset.corpus_hash
    require(corpus_hash == report["datasets"]["scifact"]["corpus_sha256"]
            == protocol["spec"]["source_corpus_sha256"], "SciFact corpus identity mismatch")
    require(canonical_digest(dataset.queries) == protocol["spec"]["source_queries_sha256"],
            "SciFact query identity mismatch")
    require(canonical_digest(dataset.qrels["test"]) == protocol["spec"]["source_test_qrels_sha256"],
            "SciFact test identity mismatch")
    require(all(qid in dataset.qrels["test"] for qid in QUERY_IDS), "Fixed public test queries are missing")
    cache_key = embedding_cache_key(corpus_hash, expected_encoder)
    cache_path = args.cache_dir / f"embeddings-{cache_key}.npy"
    require(cache_path.is_file(), "Matching evaluated corpus cache missing; refusing corpus re-encoding")
    chunks_checked = verify_evaluated_cache(dataset, report, args.cache_dir, cache_path)
    cache_hash = sha256(cache_path)
    run_path = args.results_dir / "scifact" / "runs" / "retrieved.trec"
    run_hash = sha256(run_path)
    require(run_hash == report["datasets"]["scifact"]["methods"]["retrieved"]["run_sha256"],
            "Evaluated TREC byte hash mismatch")
    reference = read_reference(run_path, {doc["_id"] for doc in dataset.docs})
    result.update({"adapter_sha256": adapter_hash, "corpus_sha256": corpus_hash,
                   "base_model_sha256": base_hashes["model.safetensors"], "trec_sha256": run_hash,
                   "embedding_cache_key": cache_key, "embedding_cache_sha256": cache_hash,
                   "evaluation_chunks_checked": chunks_checked})
    result["checks"]["verified_local_assets_and_evaluated_corpus_cache"] = True
    result["timing_seconds"]["preflight"] = perf_counter() - started

    from evidenceforge.adaptation import AdaptedEncoder
    from evidenceforge.api import create_app
    from fastapi.testclient import TestClient

    original_encode = AdaptedEncoder.encode
    original_build = RetrievalIndex.build
    allowed_queries = {dataset.queries[qid] for qid in QUERY_IDS}
    encoded_queries = []

    def guarded_encode(self, texts, batch_size=8):
        require(len(texts) == 1 and texts[0] in allowed_queries,
                "Smoke refuses any corpus encoding or non-predeclared query encoding")
        encoded_queries.append(texts[0])
        return original_encode(self, texts, batch_size=batch_size)

    def guarded_build(self):
        encoder = self.encoder_override
        require(encoder is not None, "Expected real adapted encoder from the API lifespan")
        require({k: v for k, v in encoder.metadata.items() if k != "threads"}
                == {k: v for k, v in expected_encoder.items() if k != "threads"},
                "Loaded API encoder differs from evaluated encoder")
        actual_key = embedding_cache_key(self.dataset.corpus_hash, encoder.metadata)
        require(actual_key == cache_key and cache_path.is_file() and sha256(cache_path) == cache_hash,
                "Evaluated cache changed before API startup")
        return original_build(self)

    def deny_network(*unused_args, **unused_kwargs):
        raise RuntimeError("Network access is disabled during this offline integration smoke")

    # Windows event-loop wakeup sockets may use loopback TCP internally.
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def require_loopback(address):
        require(isinstance(address, tuple) and isinstance(address[0], str),
                "Only loopback sockets needed by TestClient are allowed")
        require(address[0] == "localhost" or ipaddress.ip_address(address[0]).is_loopback,
                "External network connections are disabled during the smoke")

    def local_connect(self, address):
        require_loopback(address)
        return original_connect(self, address)

    def local_connect_ex(self, address):
        require_loopback(address)
        return original_connect_ex(self, address)

    with (patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
          patch.object(urllib.request, "urlopen", deny_network),
          patch.object(socket.socket, "connect", local_connect),
          patch.object(socket.socket, "connect_ex", local_connect_ex),
          patch.object(AdaptedEncoder, "encode", guarded_encode),
          patch.object(RetrievalIndex, "build", guarded_build)):
        app = create_app(data_dir=args.data_dir, cache_dir=args.cache_dir,
                         adapter_dir=args.adapter_dir, threads=args.threads)
        startup_started = perf_counter()
        with TestClient(app) as client:
            result["timing_seconds"]["api_lifespan_startup"] = perf_counter() - startup_started
            response = client.get("/health")
            require(response.status_code == 200 and response.json()
                    == {"ready": True, "dense": True, "ranker": False}, "Unexpected health state")
            result["checks"]["health_ready_dense_without_v1_ranker"] = True
            response = client.get("/metadata")
            require(response.status_code == 200, "Metadata endpoint failed")
            metadata = response.json()["index"]
            require(all(metadata.get(key) == value for key, value in expected_encoder.items() if key != "threads"),
                    "API metadata model identity mismatch")
            require(metadata["embedding_cache_key"] == cache_key and metadata["corpus_sha256"] == corpus_hash
                    and metadata["documents"] == 5183, "API metadata corpus/cache mismatch")
            result["checks"]["metadata_identity_matches_evaluation"] = True
            for qid in QUERY_IDS:
                query_started = perf_counter()
                response = client.post("/search", json={"query": dataset.queries[qid], "method": "dense", "top_k": 5})
                require(response.status_code == 200, f"Dense query {qid} failed: {response.status_code}")
                payload = response.json()
                actual = [hit["doc_id"] for hit in payload["results"]]
                expected = [row[1] for row in reference[qid][:5]]
                scores = [row[2] for row in reference[qid][:6]]
                result["queries"].append({"query_id": qid, "expected_top5": expected, "actual_top5": actual,
                                          "exact_order_match": actual == expected,
                                          "reference_min_adjacent_gap_top6": min(a - b for a, b in zip(scores, scores[1:])),
                                          "request_wall_seconds": perf_counter() - query_started,
                                          "api_search_latency_ms": payload["latency_ms"]})
                require(actual == expected, f"Dense top-5 differs from evaluated TREC for query {qid}; inspect scores")
            result["checks"]["three_dense_top5_equal_evaluated_ranks"] = True
            response = client.post("/search", json={"query": dataset.queries[QUERY_IDS[0]], "method": "rrf", "top_k": 5})
            require(response.status_code == 200 and len(response.json()["results"]) == 5
                    and all(math.isfinite(hit["rrf_score"]) for hit in response.json()["results"]), "RRF smoke failed")
            result["checks"]["rrf_success"] = True
            response = client.post("/search", json={"query": dataset.queries[QUERY_IDS[0]], "method": "ltr"})
            require(response.status_code == 409, "Adapted API must reject unloaded v1 LTR")
            result["checks"]["ltr_without_compatible_ranker_returns_409"] = True
            for query in ("", "   "):
                response = client.post("/search", json={"query": query, "method": "dense"})
                require(response.status_code == 422, "Empty/whitespace query must return 422")
            result["checks"]["empty_and_whitespace_queries_return_422"] = True
        require(not app.state.ready, "API lifespan did not clear ready state")
    require(len(encoded_queries) == 4, "Unexpected number of actual query encodings")
    require(sha256(cache_path) == cache_hash, "Smoke modified the evaluated corpus cache")
    result["checks"]["lifespan_shutdown"] = True
    result["checks"]["four_real_query_encodings_no_corpus_reencoding"] = True
    result["checks"]["corpus_cache_bytes_unchanged"] = True
    result["checks"]["external_network_connections_blocked"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/adaptation"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("results/adaptation/training/retrieved"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/adaptation/service_smoke.json"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be between 1 and 64")
    result = {"schema_version": 1, "passed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "query_ids": list(QUERY_IDS), "checks": {}, "queries": [], "timing_seconds": {},
              "limitations": ["Four sequential requests encode real queries; this is not a load or concurrency test.",
                              "Timing covers this invocation only, including separately reported local preflight and API startup.",
                              "Single-query inference can differ slightly from batched evaluation; exact top-5 order is required, with no tie tolerance.",
                              "External network and non-query encoding guards reject work; they do not replace model, retrieval, or API outputs.",
                              "Loopback sockets remain available for Windows TestClient event-loop internals; no real HTTP server is started."]}
    started = perf_counter()
    try:
        run(args, result)
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["timing_seconds"]["total_invocation"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
