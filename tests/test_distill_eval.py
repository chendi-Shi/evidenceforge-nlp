from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from evidenceforge import distill_eval


@pytest.fixture
def tiny_release(monkeypatch):
    files = {
        "corpus.jsonl": "\n".join(json.dumps({"_id": key, "title": key, "text": f"Text {key}"})
                                    for key in ("a", "b", "c")),
        "queries.jsonl": json.dumps({"_id": "q", "text": "Scientific title"}),
        "qrels/test.tsv": "query-id\tcorpus-id\tscore\nq\tb\t2\nq\tc\t1\n",
    }

    def install(*, extra=None, duplicate=False, symlink=False, missing=False):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for relative, value in files.items():
                if missing and relative == "queries.jsonl":
                    continue
                info = zipfile.ZipInfo(f"scidocs/{relative}")
                if symlink and relative == "corpus.jsonl":
                    info.create_system = 3
                    info.external_attr = 0o120777 << 16
                archive.writestr(info, value)
            if extra:
                archive.writestr(extra, "unexpected")
            if duplicate:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("scidocs/corpus.jsonl", files["corpus.jsonl"])
        payload = stream.getvalue()
        monkeypatch.setattr(distill_eval, "SCIDOCS_MD5", hashlib.md5(payload).hexdigest())
        monkeypatch.setattr(distill_eval, "SCIDOCS_DOCUMENTS", 3)
        monkeypatch.setattr(distill_eval, "SCIDOCS_QUERIES", 1)
        monkeypatch.setattr(distill_eval.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
        return payload

    install()
    return install


def test_mock_download_hashes_counts_license_and_offline_reuse(tmp_path, monkeypatch, tiny_release):
    payload = tiny_release()
    target = tmp_path / "scidocs"
    dataset = distill_eval.load_or_download_scidocs(target)
    metadata = dataset.metadata
    assert metadata["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["archive_md5"] == hashlib.md5(payload).hexdigest()
    assert metadata["documents"] == 3 and metadata["split_query_counts"] == {"test": 1}
    assert metadata["relevance_levels"] == [1, 2]
    assert metadata["upstream_license"] == "CC-BY-4.0"
    assert metadata["beir_card_license_label"] == "CC-BY-SA-4.0"
    for name in distill_eval.DATA_FILES:
        assert metadata["file_sha256"][name] == hashlib.sha256((target / name).read_bytes()).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("Installed dataset must load without network")

    monkeypatch.setattr(distill_eval.urllib.request, "urlopen", forbidden)
    assert distill_eval.load_or_download_scidocs(target).metadata == metadata


def test_bad_archive_checksum_installs_nothing(tmp_path, monkeypatch, tiny_release):
    monkeypatch.setattr(distill_eval, "SCIDOCS_MD5", "0" * 32)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        distill_eval.load_or_download_scidocs(tmp_path / "scidocs")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kwargs", [
    {"extra": "../escape"}, {"extra": "scidocs/../escape"}, {"extra": "scidocs/qrels/train.tsv"},
    {"extra": "scidocs/extra/"}, {"duplicate": True}, {"symlink": True}, {"missing": True},
])
def test_unsafe_or_incomplete_archive_rejected_before_install(tmp_path, tiny_release, kwargs):
    tiny_release(**kwargs)
    with pytest.raises(ValueError, match="Unsafe"):
        distill_eval.load_or_download_scidocs(tmp_path / "scidocs")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE_BYTES", "MAX_EXPANDED_BYTES"])
def test_compressed_and_expanded_limits(tmp_path, monkeypatch, tiny_release, limit):
    monkeypatch.setattr(distill_eval, limit, 10)
    with pytest.raises(ValueError, match="size limit"):
        distill_eval.load_or_download_scidocs(tmp_path / "scidocs")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["content", "manifest", "extra_split"])
def test_reuse_rejects_corrupt_files_or_manifest(tmp_path, tiny_release, mutation):
    distill_eval.load_or_download_scidocs(tmp_path)
    if mutation == "content":
        path = tmp_path / "queries.jsonl"
        path.write_text(path.read_text().replace("Scientific", "Different"))
    elif mutation == "manifest":
        path = tmp_path / "manifest.json"
        value = json.loads(path.read_text())
        value["corpus_sha256"] = "changed"
        path.write_text(json.dumps(value))
    else:
        (tmp_path / "qrels/train.tsv").write_text("query-id\tcorpus-id\tscore\nq\ta\t1\n")
    with pytest.raises(ValueError, match="checksum|manifest|files/splits"):
        distill_eval.load_or_download_scidocs(tmp_path)


def test_existing_unknown_data_never_overwritten_and_counts_checked(tmp_path, monkeypatch, tiny_release):
    keep = tmp_path / "keep.txt"
    keep.write_text("owned by user")
    with pytest.raises(FileExistsError, match="non-empty"):
        distill_eval.load_or_download_scidocs(tmp_path)
    assert keep.read_text() == "owned by user"
    monkeypatch.setattr(distill_eval, "SCIDOCS_DOCUMENTS", 4)
    with pytest.raises(ValueError, match="counts"):
        distill_eval.load_or_download_scidocs(tmp_path / "scidocs")
    assert not (tmp_path / "scidocs").exists()


def rank_tiny(path, *, reverse=False):
    # Input rows are deliberately noncanonical and a/c tie exactly.
    qids = ["q2", "q1"]
    qvec = np.asarray([[0., 1.], [1., 0.]], dtype=np.float32)
    dids = ["c", "b", "a"]
    dvec = np.asarray([[1., 0.], [0., 1.], [1., 0.]], dtype=np.float32)
    qrels = {"q1": {"a": 2, "b": 1}, "q2": {"b": 1}}
    if reverse:
        qids, qvec, dids, dvec = qids[::-1], qvec[::-1], dids[::-1], dvec[::-1]
    return distill_eval.rank_embeddings(qids, qvec, dids, dvec, qrels, path, "tiny")


def test_exact_ranking_has_canonical_ties_and_independent_graded_metrics(tmp_path):
    path = tmp_path / "run.trec"
    result = rank_tiny(path)
    lines = path.read_text().splitlines()
    assert [line.split()[2] for line in lines[:3]] == ["a", "c", "b"]
    expected_q1 = (2 + 1 / np.log2(4)) / (2 + 1 / np.log2(3))
    assert result["metrics"]["ndcg@10"] == pytest.approx((expected_q1 + 1) / 2)
    assert result["metrics"]["recall@10"] == 1
    assert result["actual_depth_min"] == 3
    assert result["per_query"] == json.loads(path.with_suffix(".per_query.json").read_text())
    reversed_result = rank_tiny(tmp_path / "reversed.trec", reverse=True)
    assert result["run_sha256"] == reversed_result["run_sha256"]
    assert result["per_query_sha256"] == reversed_result["per_query_sha256"]


def test_top50_and_self_match_policy(tmp_path):
    doc_ids = [f"d{i:03d}" for i in range(60)]
    vectors = np.ones((60, 1), dtype=np.float32)
    path = tmp_path / "self.trec"
    result = distill_eval.rank_embeddings(["d000"], vectors[:1], doc_ids, vectors,
                                          {"d000": {"d050": 1}}, path, "t")
    rows = path.read_text().splitlines()
    assert len(rows) == 50 and rows[0].split()[2] == "d001" and rows[-1].split()[2] == "d050"
    assert result["metrics"]["recall@50"] == 1
    other = distill_eval.rank_embeddings(["d000"], vectors[:1], doc_ids, vectors,
                                         {"d000": {"d050": 1}}, path, "t", ignore_identical_ids=False)
    assert other["metrics"]["recall@50"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "whitespace", "nan", "norm", "shape", "qrels", "grade", "tag"])
def test_invalid_rank_inputs_rejected_without_artifacts(tmp_path, mutation):
    kwargs = dict(query_ids=["q"], query_vectors=np.asarray([[1., 0.]]), doc_ids=["a", "b"],
                  doc_vectors=np.asarray([[1., 0.], [0., 1.]]), qrels={"q": {"a": 1}},
                  run_path=tmp_path / "run.trec", tag="t")
    if mutation == "duplicate":
        kwargs["doc_ids"] = ["a", "a"]
    elif mutation == "whitespace":
        kwargs["query_ids"] = ["bad id"]
    elif mutation == "nan":
        kwargs["doc_vectors"][0, 0] = np.nan
    elif mutation == "norm":
        kwargs["doc_vectors"][0, 0] = 2
    elif mutation == "shape":
        kwargs["query_vectors"] = np.ones((1, 3))
    elif mutation == "qrels":
        kwargs["qrels"] = {"q": {"missing": 1}}
    elif mutation == "grade":
        kwargs["qrels"] = {"q": {"a": .5}}
    else:
        kwargs["tag"] = "bad tag"
    with pytest.raises(ValueError):
        distill_eval.rank_embeddings(**kwargs)
    assert not list(tmp_path.iterdir())


def summary_inputs(tmp_path):
    result = rank_tiny(tmp_path / "run.trec")
    return {"base": {"frozen": deepcopy(result)},
            "kd": {seed: deepcopy(result) for seed in (42, 43, 44)},
            "kd_anchor": {seed: deepcopy(result) for seed in (42, 43, 44)}}


def test_seed_summary_keeps_training_sd_and_query_ci_separate(tmp_path):
    runs = summary_inputs(tmp_path)
    runs["kd_anchor"][42]["per_query"].reverse()
    result = distill_eval.summarize_seeds(runs, samples=20)
    assert result["arms"]["kd"]["seed_count"] == 3
    assert result["arms"]["base"]["metrics"]["ndcg@10"]["sample_std"] is None
    assert result["primary"]["mean_delta"] == result["primary"]["sample_std_delta"] == 0
    assert set(result["primary"]["per_seed"]) == {"42", "43", "44"}
    assert all(row["query_count"] == 2 and row["ci95_low"] == row["ci95_high"] == 0
               for row in result["primary"]["per_seed"].values())


@pytest.mark.parametrize("mutation", ["seed", "identity", "duplicate", "aggregate"])
def test_seed_summary_rejects_misaligned_or_corrupt_results(tmp_path, mutation):
    runs = summary_inputs(tmp_path)
    if mutation == "seed":
        del runs["kd_anchor"][44]
    elif mutation == "identity":
        runs["kd"][42]["qrels_sha256"] = "other labels"
    elif mutation == "duplicate":
        rows = runs["kd"][42]["per_query"]
        rows[1] = rows[0]
    else:
        runs["kd"][42]["metrics"]["ndcg@10"] = 0
    with pytest.raises(ValueError):
        distill_eval.summarize_seeds(runs, samples=20)


def test_import_and_pure_ranking_do_not_unseal_target_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Neither network nor held-out Dataset loading belongs in ranking")

    monkeypatch.setattr(distill_eval.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(distill_eval, "load_dataset", forbidden)
    monkeypatch.chdir(tmp_path)
    assert rank_tiny(tmp_path / "run.trec")["query_count"] == 2
    assert not (tmp_path / "data/scidocs").exists()
