import io
import json
from pathlib import Path
import zipfile

import pytest

from evidenceforge.data import Dataset, _safe_extract, load_dataset


def write_dataset(path: Path, docs=None, qrels="q1\td1\t1\n"):
    path.mkdir(parents=True, exist_ok=True)
    docs = docs if docs is not None else [{"_id": "d1", "title": "Title", "text": "Evidence"}]
    (path / "corpus.jsonl").write_text("\n".join(json.dumps(doc) for doc in docs), encoding="utf-8")
    (path / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "claim"}), encoding="utf-8")
    (path / "qrels").mkdir(exist_ok=True)
    (path / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\n" + qrels, encoding="utf-8")


def test_load_validates_join_and_preserves_split(tmp_path):
    write_dataset(tmp_path)
    dataset = load_dataset(tmp_path)
    assert dataset.docs[0]["title"] == "Title"
    assert dataset.queries == {"q1": "claim"}
    assert dataset.qrels == {"train": {"q1": {"d1": 1}}}


@pytest.mark.parametrize("qrels", ["q2\td1\t1\n", "q1\td2\t1\n"])
def test_rejects_qrels_with_missing_ids(tmp_path, qrels):
    write_dataset(tmp_path, qrels=qrels)
    with pytest.raises(ValueError, match="missing query/document"):
        load_dataset(tmp_path)


def test_duplicate_ids_are_not_silently_overwritten(tmp_path):
    write_dataset(tmp_path, docs=[{"_id": "d1", "text": "a"}, {"_id": "d1", "text": "b"}])
    with pytest.raises(ValueError, match="duplicate corpus ID"):
        load_dataset(tmp_path)


def test_corpus_hash_is_order_independent_but_content_sensitive():
    first = {"_id": "a", "title": "A", "text": "Evidence"}
    second = {"_id": "b", "title": "B", "text": "More evidence"}
    assert Dataset([first, second], {}, {}).corpus_hash == Dataset([second, first], {}, {}).corpus_hash
    changed = dict(first, text="Changed evidence")
    assert Dataset([first, second], {}, {}).corpus_hash != Dataset([changed, second], {}, {}).corpus_hash


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/outside.txt", "..\\escape.txt"])
def test_zip_slip_rejected_before_extracting_any_file(tmp_path, member):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("safe.txt", "first file must not be extracted either")
        archive.writestr(member, "danger")
    data.seek(0)
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="Unsafe"):
        _safe_extract(archive, tmp_path)
    assert not list(tmp_path.iterdir())


def test_zip_symlink_rejected(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (0o120777 << 16)
        archive.writestr(member, "../outside")
    data.seek(0)
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="Unsafe"):
        _safe_extract(archive, tmp_path)
