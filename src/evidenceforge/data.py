"""Validated BEIR-format data loading and the reproducible SciFact download."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
DATASET_SOURCE = "https://github.com/beir-cellar/beir"


@dataclass
class Dataset:
    docs: list[dict]
    queries: dict[str, str]
    qrels: dict[str, dict[str, dict[str, int]]]
    metadata: dict = field(default_factory=dict)

    @property
    def corpus_hash(self) -> str:
        """Content hash independent of source JSONL formatting or document order."""
        digest = hashlib.sha256()
        for doc in sorted(self.docs, key=lambda item: str(item["_id"])):
            value = {key: str(doc.get(key, "")) for key in ("_id", "title", "text")}
            digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield value


def load_dataset(data_dir: Path) -> Dataset:
    """Read corpus.jsonl, queries.jsonl and qrels/*.tsv; reject broken joins."""
    data_dir = Path(data_dir)
    docs, doc_ids, queries, qrels = [], set(), {}, {}
    for item in _jsonl(data_dir / "corpus.jsonl"):
        if "_id" not in item or "text" not in item:
            raise ValueError("Each corpus row needs _id and text")
        doc_id = str(item["_id"])
        if not doc_id or doc_id in doc_ids:
            raise ValueError(f"Empty or duplicate corpus ID: {doc_id!r}")
        if not isinstance(item["text"], str) or not isinstance(item.get("title", ""), str):
            raise ValueError(f"Non-string document text/title: {doc_id}")
        doc_ids.add(doc_id)
        docs.append({"_id": doc_id, "title": item.get("title", ""), "text": item["text"]})
    if not docs:
        raise ValueError("Corpus is empty")
    for item in _jsonl(data_dir / "queries.jsonl"):
        if "_id" not in item or not isinstance(item.get("text"), str):
            raise ValueError("Each query needs _id and string text")
        query_id = str(item["_id"])
        if not query_id or query_id in queries:
            raise ValueError(f"Empty or duplicate query ID: {query_id!r}")
        queries[query_id] = item["text"]
    for path in sorted((data_dir / "qrels").glob("*.tsv")):
        split = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Invalid qrels header: {path}")
            for row in reader:
                query_id, doc_id = row["query-id"], row["corpus-id"]
                if query_id not in queries or doc_id not in doc_ids:
                    raise ValueError(f"Qrel references missing query/document: {query_id}/{doc_id}")
                score = int(row["score"])
                if score < 0:
                    raise ValueError(f"Negative relevance score: {query_id}/{doc_id}")
                judgments = split.setdefault(query_id, {})
                if doc_id in judgments:
                    raise ValueError(f"Duplicate qrel: {query_id}/{doc_id}")
                judgments[doc_id] = score
        qrels[path.stem] = split
    if not qrels:
        raise ValueError(f"No qrels/*.tsv found under {data_dir}")
    metadata_path = data_dir / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return Dataset(sorted(docs, key=lambda doc: doc["_id"]), queries, qrels, metadata)


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract after validating every member; reject traversal, links and zip bombs."""
    destination = Path(destination).resolve()
    members = archive.infolist()
    if sum(item.file_size for item in members) > 100_000_000:
        raise ValueError("SciFact archive exceeds the 100 MB uncompressed limit")
    for member in members:
        normalized = member.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        mode = member.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or ":" in normalized or mode & 0o170000 == 0o120000:
            raise ValueError(f"Unsafe archive member: {member.filename}")
        resolved = destination.joinpath(*path.parts).resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError(f"Archive member escapes destination: {member.filename}")
    for member in members:
        path = destination.joinpath(*PurePosixPath(member.filename.replace("\\", "/")).parts)
        if member.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, path.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_scifact(data_dir: Path) -> Path:
    """Download official BEIR SciFact, verify its published MD5, install atomically.

    Raw data keeps its upstream licenses: CC BY 4.0 for claims/evidence;
    ODC-By 1.0 for abstracts. Neither is replaced by this project's code license.
    Existing, valid local datasets work without network access.
    """
    data_dir = Path(data_dir).resolve()
    if (data_dir / "corpus.jsonl").exists():
        load_dataset(data_dir)
        return data_dir
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scifact-", dir=data_dir.parent) as staging_name:
        staging = Path(staging_name)
        zip_path = staging / "scifact.zip"
        request = urllib.request.Request(SCIFACT_URL, headers={"User-Agent": "EvidenceForge/0.1"})
        LOGGER.info("Downloading official SciFact archive")
        with urllib.request.urlopen(request, timeout=90) as response, zip_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual_md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
        if actual_md5 != SCIFACT_MD5:
            raise ValueError(f"SciFact checksum mismatch: expected {SCIFACT_MD5}, got {actual_md5}")
        with zipfile.ZipFile(zip_path) as archive:
            _safe_extract(archive, staging / "extracted")
        source = staging / "extracted" / "scifact"
        dataset = load_dataset(source)
        manifest = {
            "dataset": "BEIR/SciFact", "url": SCIFACT_URL, "archive_md5": actual_md5,
            "corpus_sha256": dataset.corpus_hash, "documents": len(dataset.docs),
            "queries": len(dataset.queries),
            "split_query_counts": {split: len(rows) for split, rows in dataset.qrels.items()},
            "license_source": "https://github.com/allenai/scifact/blob/master/LICENSE.md",
            "claims_license": "CC-BY-4.0", "abstracts_license": "ODC-By-1.0",
        }
        (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if data_dir.exists():
            if any(data_dir.iterdir()):
                raise FileExistsError(f"Refusing to replace non-empty dataset directory: {data_dir}")
            data_dir.rmdir()
        os.replace(source, data_dir)
    return data_dir
