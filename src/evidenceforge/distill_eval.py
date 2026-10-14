"""Explicit held-out SCIDOCS loading and model-free distillation evaluation.

Importing this module neither downloads data nor reads held-out labels. The
caller must freeze training/selection before invoking the SCIDOCS loader.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import urllib.request
import zipfile

import numpy as np

from .data import Dataset, load_dataset
from .evaluation import METRIC_NAMES, macro_metrics, paired_bootstrap, query_metrics

SCIDOCS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
SCIDOCS_MD5 = "38121350fc3a4d2f48850f6aff52e4a9"
SCIDOCS_DOCUMENTS = 25657
SCIDOCS_QUERIES = 1000
MAX_ARCHIVE_BYTES = 150_000_000
MAX_EXPANDED_BYTES = 300_000_000
DATA_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
BEIR_SOURCE = "https://github.com/beir-cellar/beir#available-datasets"
CARD_SOURCE = "https://huggingface.co/datasets/BeIR/scidocs/blob/main/README.md"
LICENSE_SOURCE = "https://github.com/allenai/scidocs/blob/master/LICENSE"
RANKING_POLICY = "all supplied documents, excluding identical query/document IDs by default"


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_scidocs(archive: zipfile.ZipFile, destination: Path) -> None:
    """Validate the complete member table before writing into fresh staging."""
    members = archive.infolist()
    files = {f"scidocs/{name}" for name in DATA_FILES}
    directories = {"scidocs/", "scidocs/qrels/"}
    names = [member.filename for member in members]
    if len(names) != len(set(names)) or {m.filename for m in members if not m.is_dir()} != files:
        raise ValueError("Unsafe or unexpected SCIDOCS archive members")
    if sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
        raise ValueError("SCIDOCS archive exceeds expanded size limit")
    destination = destination.resolve()
    for member in members:
        name, mode = member.filename, stat.S_IFMT(member.external_attr >> 16)
        path = PurePosixPath(name)
        if ("\\" in name or ":" in name or path.is_absolute() or ".." in path.parts
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                or (member.is_dir() and name not in directories)
                or (mode == stat.S_IFDIR and not member.is_dir())
                or member.flag_bits & 1
                or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError(f"Unsafe SCIDOCS archive member: {name}")
        if not destination.joinpath(*path.parts).resolve().is_relative_to(destination):
            raise ValueError(f"Unsafe SCIDOCS archive path: {name}")
    written = 0
    for member in members:
        path = destination.joinpath(*PurePosixPath(member.filename).parts)
        if member.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, path.open("xb") as output:
            count = 0
            while block := source.read(64 * 1024):
                count += len(block)
                written += len(block)
                if count > member.file_size or written > MAX_EXPANDED_BYTES:
                    raise ValueError("SCIDOCS archive exceeds expanded size limit")
                output.write(block)
            if count != member.file_size:
                raise ValueError("SCIDOCS member size mismatch")


def _dataset_summary(dataset: Dataset) -> dict:
    if (len(dataset.docs) != SCIDOCS_DOCUMENTS or len(dataset.queries) != SCIDOCS_QUERIES
            or set(dataset.qrels) != {"test"} or set(dataset.qrels["test"]) != set(dataset.queries)
            or any(not any(score > 0 for score in rels.values()) for rels in dataset.qrels["test"].values())):
        raise ValueError("SCIDOCS counts or test-query coverage differ from the complete BEIR release")
    return {
        "corpus_sha256": dataset.corpus_hash,
        "queries_sha256": _digest(dataset.queries), "test_qrels_sha256": _digest(dataset.qrels["test"]),
        "documents": len(dataset.docs), "queries": len(dataset.queries),
        "split_query_counts": {"test": len(dataset.qrels["test"])},
        "test_qrel_count": sum(len(rels) for rels in dataset.qrels["test"].values()),
        "relevance_levels": sorted({score for rels in dataset.qrels["test"].values() for score in rels.values()}),
    }


def _provenance() -> dict:
    return {
        "schema_version": 1, "dataset": "BEIR/SCIDOCS", "url": SCIDOCS_URL,
        "archive_md5": SCIDOCS_MD5, "checksum_source": BEIR_SOURCE, "count_source": CARD_SOURCE,
        "upstream_license": "CC-BY-4.0", "license_source": LICENSE_SOURCE,
        "beir_card_license_label": "CC-BY-SA-4.0", "beir_card_source": CARD_SOURCE,
        "license_note": "Original SciDocs LICENSE and the BEIR card's generic license label are recorded separately; the project code license does not relicense data.",
        "evaluation_note": "BEIR citation-prediction retrieval from scientific titles; not the complete original SciDocs suite. Held-out test only: no fitting or selection on its labels.",
    }


def _load_installed_scidocs(data_dir: Path) -> Dataset:
    expected = {*DATA_FILES, "manifest.json"}
    actual = set()
    for path in data_dir.rglob("*"):
        relative = path.relative_to(data_dir).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(data_dir):
            raise ValueError("Unsafe installed SCIDOCS path")
        if path.is_file():
            actual.add(relative)
        elif not path.is_dir() or relative != "qrels":
            raise ValueError("Unexpected installed SCIDOCS directory")
    if actual != expected:
        raise ValueError("Missing or unexpected SCIDOCS files/splits")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if (any(manifest.get(key) != value for key, value in _provenance().items())
            or not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("archive_sha256", "")))
            or set(manifest.get("file_sha256", {})) != set(DATA_FILES)
            or set(manifest.get("file_bytes", {})) != set(DATA_FILES)):
        raise ValueError("SCIDOCS manifest provenance mismatch")
    for name in DATA_FILES:
        path = data_dir / name
        if _hash_file(path) != manifest["file_sha256"][name] or path.stat().st_size != manifest["file_bytes"][name]:
            raise ValueError(f"SCIDOCS file checksum/size mismatch: {name}")
    dataset = load_dataset(data_dir)
    if any(manifest.get(key) != value for key, value in _dataset_summary(dataset).items()):
        raise ValueError("SCIDOCS manifest content mismatch")
    return dataset


def load_or_download_scidocs(data_dir: Path) -> Dataset:
    """Explicitly unseal SCIDOCS after freezing training; reuse verified files offline.

    The official published MD5 authenticates the initial archive identity. Its
    observed SHA256 and file SHA256 values are recorded, not claimed to be
    independently published checksums. Reuse verifies files against that local
    manifest; callers should freeze the manifest hash in their experiment.
    """
    data_dir = Path(data_dir)
    if data_dir.is_symlink():
        raise ValueError("Refusing a symlink as the SCIDOCS destination")
    data_dir = data_dir.resolve()
    if data_dir.exists():
        if not data_dir.is_dir():
            raise FileExistsError("Refusing to replace an existing SCIDOCS path")
        if (data_dir / "manifest.json").exists():
            return _load_installed_scidocs(data_dir)
        if any(data_dir.iterdir()):
            raise FileExistsError("Refusing to replace a non-empty SCIDOCS directory without a valid manifest")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scidocs-", dir=data_dir.parent) as name:
        staging = Path(name)
        archive_path = staging / "scidocs.zip"
        request = urllib.request.Request(SCIDOCS_URL, headers={"User-Agent": "EvidenceForge/0.3"})
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as output:
            downloaded = 0
            while block := response.read(64 * 1024):
                downloaded += len(block)
                if downloaded > MAX_ARCHIVE_BYTES:
                    raise ValueError("SCIDOCS archive exceeds compressed size limit")
                output.write(block)
        if _hash_file(archive_path, "md5") != SCIDOCS_MD5:
            raise ValueError("SCIDOCS archive MD5 mismatch; no files installed")
        with zipfile.ZipFile(archive_path) as archive:
            _extract_scidocs(archive, staging / "extracted")
        source = staging / "extracted" / "scidocs"
        dataset = load_dataset(source)
        manifest = {
            **_provenance(), **_dataset_summary(dataset),
            "archive_sha256": _hash_file(archive_path), "archive_bytes": downloaded,
            "file_sha256": {name: _hash_file(source / name) for name in DATA_FILES},
            "file_bytes": {name: (source / name).stat().st_size for name in DATA_FILES},
        }
        _write_json(source / "manifest.json", manifest)
        if data_dir.exists():
            if any(data_dir.iterdir()):
                raise FileExistsError("Refusing to replace a non-empty SCIDOCS directory")
            data_dir.rmdir()
        os.replace(source, data_dir)
    dataset.metadata = manifest
    return dataset


def _ids(values: Sequence[str], kind: str) -> list[str]:
    ids = list(values)
    if (not ids or any(not isinstance(value, str) or not value or re.search(r"\s", value) for value in ids)
            or len(ids) != len(set(ids))):
        raise ValueError(f"{kind} IDs must be nonempty, unique, whitespace-free strings")
    return ids


def rank_embeddings(query_ids, query_vectors, doc_ids, doc_vectors, qrels, run_path: Path, tag: str,
                    *, ignore_identical_ids: bool = True) -> dict:
    """Exact cosine ranking of all supplied documents, saving explicit top-50 ranks.

    Inputs must already be unit-normalized; rows align with their supplied IDs.
    All qrels queries must be present, and every qrel document must be in the
    supplied corpus. The caller defines whether this corpus is a full benchmark
    or a fixed sampled validation pool. No labels influence ranking.
    """
    query_ids, doc_ids = _ids(query_ids, "Query"), _ids(doc_ids, "Document")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("TREC tag must be a simple nonempty identifier")
    query_vectors, doc_vectors = np.asarray(query_vectors), np.asarray(doc_vectors)
    if (query_vectors.ndim != 2 or doc_vectors.ndim != 2
            or query_vectors.shape[0] != len(query_ids) or doc_vectors.shape[0] != len(doc_ids)
            or query_vectors.shape[1] != doc_vectors.shape[1] or query_vectors.shape[1] < 1):
        raise ValueError("Embedding/ID dimensions do not align")
    for vectors in (query_vectors, doc_vectors):
        if (vectors.dtype.kind != "f" or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4, rtol=0)):
            raise ValueError("Embeddings must be finite unit-normalized floating-point vectors")
    documents = set(doc_ids)
    if set(qrels) != set(query_ids):
        raise ValueError("Qrels must cover exactly the supplied query IDs")
    for qid, rels in qrels.items():
        if (not isinstance(rels, Mapping) or not rels
                or any(doc not in documents or isinstance(rel, bool) or not isinstance(rel, int) or rel < 0
                       for doc, rel in rels.items()) or not any(rel > 0 for rel in rels.values())):
            raise ValueError(f"Invalid qrels or missing relevant document for query {qid}")
    run_path = Path(run_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_path.with_suffix(run_path.suffix + ".tmp")
    ids_array = np.asarray(doc_ids)
    rows, depths = [], []
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            # Canonical query order also makes artifacts invariant to input row order.
            for position in sorted(range(len(query_ids)), key=lambda index: query_ids[index]):
                qid = query_ids[position]
                scores = doc_vectors @ query_vectors[position]
                order = np.lexsort((ids_array, -scores))
                if ignore_identical_ids:
                    order = order[ids_array[order] != qid]
                order = order[:50]
                depths.append(len(order))
                ranking = ids_array[order].tolist()
                rows.append({"query_id": qid, **query_metrics(ranking, qrels[qid])})
                for rank, index in enumerate(order, 1):
                    handle.write(f"{qid} Q0 {doc_ids[index]} {rank} {float(scores[index]):.17g} {tag}\n")
        os.replace(temporary, run_path)
    finally:
        temporary.unlink(missing_ok=True)
    per_query_path = run_path.with_suffix(".per_query.json")
    _write_json(per_query_path, rows)
    return {
        "metrics": macro_metrics(rows), "per_query": rows, "query_count": len(query_ids),
        "corpus_size": len(doc_ids), "ranking_scope": "all supplied documents",
        "saved_ranking_depth": 50, "actual_depth_min": min(depths), "actual_depth_max": max(depths),
        "tie_break": "document ID lexicographic ascending", "ignore_identical_ids": ignore_identical_ids,
        "query_ids_sha256": _digest(sorted(query_ids)), "document_ids_sha256": _digest(sorted(doc_ids)),
        "qrels_sha256": _digest(qrels), "score_function": "normalized dot product (cosine)",
        "run_sha256": _hash_file(run_path), "per_query_sha256": _hash_file(per_query_path),
        "run_filename": run_path.name, "per_query_filename": per_query_path.name,
        "scope_note": "Counts and hashes identify the supplied evaluation set; these fields alone do not certify that it is the full original benchmark.",
    }


def _checked_rows(result: Mapping) -> dict[str, dict]:
    rows = result["per_query"]
    query_ids = _ids([row["query_id"] for row in rows], "Per-query")
    if result["query_count"] != len(rows) or result["query_ids_sha256"] != _digest(sorted(query_ids)):
        raise ValueError("Inconsistent per-query result identity/count")
    for row in rows:
        for metric in METRIC_NAMES:
            value = row[metric]
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 + 1e-12:
                raise ValueError("Invalid per-query metric")
    metrics = macro_metrics(rows)
    if any(not math.isclose(result["metrics"][name], metrics[name], rel_tol=0, abs_tol=1e-10) for name in METRIC_NAMES):
        raise ValueError("Aggregate metrics disagree with per-query rows")
    return {row["query_id"]: row for row in rows}


def summarize_seeds(runs: Mapping[str, Mapping[int | str, Mapping]], *, treatment: str = "kd_anchor",
                    reference: str = "kd", metric: str = "ndcg@10", samples: int = 2000,
                    bootstrap_seed: int = 42) -> dict:
    """Keep training-seed variation separate from each run's paired-query CI.

    A frozen baseline may have one entry. Primary treatment/reference must have
    exactly the same seed keys. Every run must evaluate the same query/doc IDs,
    judgments and self-match policy. This does not pool seed x query observations.
    """
    if not runs or treatment not in runs or reference not in runs or treatment == reference or metric not in METRIC_NAMES:
        raise ValueError("Need distinct primary arms and a supported metric")
    normalized, identities, checked, summary = {}, set(), {}, {}
    for arm, entries in runs.items():
        if not entries:
            raise ValueError(f"No seed runs for {arm}")
        values = {str(seed): result for seed, result in entries.items()}
        if len(values) != len(entries):
            raise ValueError("Duplicate normalized seed keys")
        normalized[arm], checked[arm] = values, {}
        for seed, result in values.items():
            checked[arm][seed] = _checked_rows(result)
            identities.add(_digest({key: result[key] for key in (
                "query_ids_sha256", "document_ids_sha256", "qrels_sha256", "corpus_size", "query_count",
                "ignore_identical_ids", "score_function", "tie_break", "saved_ranking_depth",
            )}))
        summary[arm] = {
            "seed_count": len(values), "per_seed_metrics": {seed: result["metrics"] for seed, result in values.items()},
            "metrics": {name: {
                "mean": float(np.mean([result["metrics"][name] for result in values.values()])),
                "sample_std": float(np.std([result["metrics"][name] for result in values.values()], ddof=1)) if len(values) > 1 else None,
            } for name in METRIC_NAMES},
        }
    if len(identities) != 1:
        raise ValueError("Seed runs have mismatched evaluation identities or ranking policies")
    if set(normalized[treatment]) != set(normalized[reference]):
        raise ValueError("Primary arms must have exactly matching training seeds")
    comparisons = {}
    for seed in sorted(normalized[treatment]):
        a, b = checked[treatment][seed], checked[reference][seed]
        if set(a) != set(b):
            raise ValueError("Paired runs have mismatched query IDs")
        ids = sorted(a)
        interval = paired_bootstrap([a[qid][metric] for qid in ids], [b[qid][metric] for qid in ids],
                                    samples=samples, seed=bootstrap_seed)
        comparisons[seed] = {
            **interval, "training_seed": seed, "query_count": len(ids),
            "uncertainty_unit": "paired query, conditional on these two fitted models",
        }
    deltas = [row["delta"] for row in comparisons.values()]
    return {
        "arms": summary,
        "primary": {"treatment": treatment, "reference": reference, "metric": metric,
                    "per_seed": comparisons, "mean_delta": float(np.mean(deltas)),
                    "sample_std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None},
        "uncertainty_note": "Sample SD across training seeds is descriptive, not a confidence interval. Each bootstrap resamples paired queries within one seed; it does not estimate training randomness, cluster related queries, correct multiple comparisons, or treat seed x query pairs as independent observations.",
    }
