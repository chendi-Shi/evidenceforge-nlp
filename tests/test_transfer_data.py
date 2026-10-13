from dataclasses import replace
import hashlib
import io
import json
import zipfile

import pytest

from evidenceforge import transfer_data


@pytest.fixture
def tiny_release(monkeypatch):
    files = {
        "corpus.jsonl": json.dumps({"_id": "d1", "title": "Study", "text": "Medical evidence"}),
        "queries.jsonl": json.dumps({"_id": "q1", "text": "health question"}),
        "qrels/test.tsv": "query-id\tcorpus-id\tscore\nq1\td1\t2\n",
    }

    def install(extra_member=None, duplicate=False, symlink=False):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for filename, content in files.items():
                member = zipfile.ZipInfo(f"nfcorpus/{filename}")
                if symlink and filename == "corpus.jsonl":
                    member.create_system = 3
                    member.external_attr = 0o120777 << 16
                archive.writestr(member, content)
            if extra_member:
                archive.writestr(extra_member, "unsafe")
            if duplicate:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("nfcorpus/corpus.jsonl", files["corpus.jsonl"])
        payload = buffer.getvalue()
        source = replace(
            transfer_data.SOURCES["nfcorpus"],
            md5=hashlib.md5(payload).hexdigest(),
            sha256=hashlib.sha256(payload).hexdigest(),
            documents=1,
            queries=1,
            split_query_counts={"test": 1},
            file_sha256={name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
        )
        monkeypatch.setitem(transfer_data.SOURCES, "nfcorpus", source)
        monkeypatch.setattr(transfer_data.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
        return payload

    install()
    return install


def test_download_preserves_graded_qrels_and_reuses_pinned_data_offline(tmp_path, monkeypatch, tiny_release):
    target = tmp_path / "nfcorpus"
    dataset = transfer_data.download_external("nfcorpus", target)
    assert dataset.qrels == {"test": {"q1": {"d1": 2}}}
    assert dataset.metadata["relevance_levels"] == {"test": [2]}
    assert dataset.metadata["license_source"] == transfer_data.NFCORPUS_SOURCE

    def network_forbidden(*args, **kwargs):
        raise AssertionError("An installed dataset must work offline")

    monkeypatch.setattr(transfer_data.urllib.request, "urlopen", network_forbidden)
    assert transfer_data.load_external("nfcorpus", target).corpus_hash == dataset.corpus_hash
    assert transfer_data.download_external("nfcorpus", target).metadata == dataset.metadata


@pytest.mark.parametrize("algorithm", ["md5", "sha256"])
def test_wrong_archive_checksum_cannot_install(tmp_path, monkeypatch, tiny_release, algorithm):
    source = transfer_data.SOURCES["nfcorpus"]
    monkeypatch.setitem(transfer_data.SOURCES, "nfcorpus", replace(source, **{algorithm: "0" * 64}))
    target = tmp_path / "nfcorpus"
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        transfer_data.download_external("nfcorpus", target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["query", "qrel", "manifest", "extra_split"])
def test_cached_corruption_is_rejected_without_repair(tmp_path, tiny_release, change):
    transfer_data.download_external("nfcorpus", tmp_path)
    if change == "query":
        (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"modified"}', encoding="utf-8")
    elif change == "qrel":
        (tmp_path / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    elif change == "manifest":
        metadata = json.loads((tmp_path / "manifest.json").read_text())
        metadata["dataset"] = "another release"
        (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    else:
        (tmp_path / "qrels/train.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    with pytest.raises(ValueError, match="checksum|manifest|splits"):
        transfer_data.download_external("nfcorpus", tmp_path)


@pytest.mark.parametrize("kwargs", [
    {"extra_member": "../escape.txt"},
    {"extra_member": "nfcorpus/../escape.txt"},
    {"duplicate": True},
    {"symlink": True},
])
def test_unsafe_archive_is_rejected_before_installing(tmp_path, tiny_release, kwargs):
    tiny_release(**kwargs)
    with pytest.raises(ValueError, match="Unsafe"):
        transfer_data.download_external("nfcorpus", tmp_path / "nfcorpus")
    assert list(tmp_path.iterdir()) == []


def test_download_size_limit_and_existing_directory_protection(tmp_path, monkeypatch, tiny_release):
    monkeypatch.setattr(transfer_data, "MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(ValueError, match="size limit"):
        transfer_data.download_external("nfcorpus", tmp_path / "nfcorpus")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("user data")
    with pytest.raises(FileExistsError, match="Refusing"):
        transfer_data.download_external("nfcorpus", tmp_path)
    assert unrelated.read_text() == "user data"


def test_unknown_dataset_fails_before_network(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        transfer_data.download_external("made-up", tmp_path)
