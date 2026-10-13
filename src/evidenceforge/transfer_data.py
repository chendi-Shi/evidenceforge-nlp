"""Pinned, offline-verifiable BEIR data for held-out transfer evaluation.

NFCorpus is a medical IR task with nontechnical queries. It shifts the query
distribution relative to SciFact, but their biomedical topics can overlap.
Loading its official train/dev splits does not authorize their use for fitting:
the transfer experiment must keep target-domain labels out of model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from .data import Dataset, _safe_extract, load_dataset

LOGGER = logging.getLogger(__name__)
BEIR_SOURCE = "https://github.com/beir-cellar/beir#available-datasets"
NFCORPUS_SOURCE = "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/"
MAX_ARCHIVE_BYTES = 30_000_000


@dataclass(frozen=True)
class ExternalSource:
    name: str
    dataset: str
    url: str
    md5: str
    sha256: str
    documents: int
    queries: int
    split_query_counts: dict[str, int]
    file_sha256: dict[str, str]


# The MD5 is published by BEIR. SHA256 values below pin the exact archive and
# extracted files independently verified against that published MD5.
SOURCES = {
    "nfcorpus": ExternalSource(
        name="nfcorpus",
        dataset="BEIR/NFCorpus",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        md5="a89dba18a62ef92f7d323ec890a0d38d",
        sha256="efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        documents=3633,
        queries=3237,
        split_query_counts={"train": 2590, "dev": 324, "test": 323},
        file_sha256={
            "corpus.jsonl": "10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f",
            "queries.jsonl": "d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a",
            "qrels/train.tsv": "6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a",
            "qrels/dev.tsv": "b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8",
            "qrels/test.tsv": "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
        },
    ),
}


def _source(name: str) -> ExternalSource:
    try:
        return SOURCES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported external dataset {name!r}; choose from {sorted(SOURCES)}") from exc


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_files(source: ExternalSource, data_dir: Path) -> None:
    actual_qrels = {path.relative_to(data_dir).as_posix() for path in (data_dir / "qrels").glob("*.tsv")}
    expected_qrels = {name for name in source.file_sha256 if name.startswith("qrels/")}
    if actual_qrels != expected_qrels:
        raise ValueError("External dataset qrel splits differ from the pinned release")
    for relative, expected in source.file_sha256.items():
        path = data_dir / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(data_dir.resolve()):
            raise ValueError(f"Missing or unsafe external dataset file: {relative}")
        if _hash_file(path) != expected:
            raise ValueError(f"External dataset file checksum mismatch: {relative}")


def _manifest(source: ExternalSource, dataset: Dataset) -> dict:
    counts = {split: len(rows) for split, rows in dataset.qrels.items()}
    if (len(dataset.docs), len(dataset.queries), counts) != (
        source.documents, source.queries, source.split_query_counts,
    ):
        raise ValueError("External dataset counts differ from the pinned release")
    return {
        "schema_version": 1,
        "dataset": source.dataset,
        "name": source.name,
        "url": source.url,
        "archive_md5": source.md5,
        "archive_sha256": source.sha256,
        "checksum_source": BEIR_SOURCE,
        "file_sha256": source.file_sha256,
        "corpus_sha256": dataset.corpus_hash,
        "documents": len(dataset.docs),
        "queries": len(dataset.queries),
        "split_query_counts": counts,
        "split_qrel_counts": {
            split: sum(len(judgments) for judgments in rows.values())
            for split, rows in dataset.qrels.items()
        },
        "relevance_levels": {
            split: sorted({score for judgments in rows.values() for score in judgments.values()})
            for split, rows in dataset.qrels.items()
        },
        "license_source": NFCORPUS_SOURCE,
        "license": "Upstream academic-use terms; no SPDX license asserted",
        "license_note": (
            "The authors permit academic use. Other uses of included NutritionFacts.org data "
            "require consulting its Terms of Service and contacting Dr. Michael Greger. "
            "This project's source-code license does not relicense the data."
        ),
        "citation": (
            "Boteva, Gholipour, Sokolov, Riezler (2016). A Full-Text Learning to Rank Dataset "
            "for Medical Information Retrieval. ECIR."
        ),
        "evaluation_note": (
            "BEIR adaptation, not the original 9,964-document full-text release. "
            "For transfer evaluation use official test; do not fit or tune on target labels. "
            "Relevance is not a medical truth or claim-support label."
        ),
    }


def load_external(name: str, data_dir: Path) -> Dataset:
    """Load a pinned installed dataset offline, validating every data file and manifest.

    ``data_dir`` is the dataset directory itself, e.g. ``data/nfcorpus``.
    Graded relevance values are retained. No network request is made.
    """
    source = _source(name)
    data_dir = Path(data_dir)
    _verify_files(source, data_dir)
    dataset = load_dataset(data_dir)
    expected = _manifest(source, dataset)
    if any(dataset.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("External dataset manifest missing or inconsistent with the pinned release")
    return dataset


def download_external(name: str, data_dir: Path) -> Dataset:
    """Download and atomically install a checksum-pinned release; reuse valid data offline.

    Invalid existing data is never silently overwritten. Archives are capped at
    30 MB compressed / 100 MB expanded, validated before extraction, and cannot
    install links, traversal paths, duplicate members, or unlisted files.
    """
    source = _source(name)
    data_dir = Path(data_dir).resolve()
    if data_dir.exists():
        if (data_dir / "corpus.jsonl").exists():
            return load_external(name, data_dir)
        if not data_dir.is_dir() or any(data_dir.iterdir()):
            raise FileExistsError(f"Refusing to replace non-empty dataset directory: {data_dir}")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{source.name}-", dir=data_dir.parent) as staging_name:
        staging = Path(staging_name)
        zip_path = staging / "dataset.zip"
        request = urllib.request.Request(source.url, headers={"User-Agent": "EvidenceForge/0.1"})
        LOGGER.info("Downloading official %s archive", source.dataset)
        with urllib.request.urlopen(request, timeout=90) as response, zip_path.open("wb") as output:
            downloaded = 0
            while block := response.read(64 * 1024):
                downloaded += len(block)
                if downloaded > MAX_ARCHIVE_BYTES:
                    raise ValueError("External archive exceeds the compressed size limit")
                output.write(block)
        if _hash_file(zip_path, "md5") != source.md5 or _hash_file(zip_path) != source.sha256:
            raise ValueError("External archive checksum mismatch; no files installed")
        with zipfile.ZipFile(zip_path) as archive:
            expected_files = {f"{source.name}/{relative}" for relative in source.file_sha256}
            members = [member.filename for member in archive.infolist() if not member.is_dir()]
            if len(members) != len(set(members)) or set(members) != expected_files:
                raise ValueError("Unsafe or unexpected archive members")
            allowed_dirs = {f"{source.name}/", f"{source.name}/qrels/"}
            if any(member.is_dir() and member.filename not in allowed_dirs for member in archive.infolist()):
                raise ValueError("Unsafe or unexpected archive directories")
            _safe_extract(archive, staging / "extracted")
        extracted = staging / "extracted" / source.name
        _verify_files(source, extracted)
        dataset = load_dataset(extracted)
        manifest = _manifest(source, dataset)
        (extracted / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if data_dir.exists():
            if any(data_dir.iterdir()):
                raise FileExistsError(f"Refusing to replace non-empty dataset directory: {data_dir}")
            data_dir.rmdir()
        os.replace(extracted, data_dir)
    dataset.metadata = manifest
    return dataset
