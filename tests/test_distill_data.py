import hashlib
import json

import pytest

from evidenceforge import distill_data
from evidenceforge.distill_data import (
    canonical_text, document_id, normalize_query, prepare_triplets, query_id,
)


def write_source(tmp_path, rows, name="source.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_rows(output):
    return {
        split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        for split in ("train", "validation")
    }


def ordinary_rows(count=30):
    return [{"query": f"Question {i}?", "positive": f"Relevant text {i}", "negative": f"Unrelated text {i}"}
            for i in range(count)]


def test_normalization_groups_case_and_whitespace_without_erasing_numbers_or_punctuation():
    assert normalize_query("  Ｒates\n\tIN  2020? ") == "rates in 2020?"
    assert query_id("Rates IN 2020?") == query_id("  rates in 2020?  ")
    assert query_id("rates in 2020?") != query_id("rates in 2021?")
    assert query_id("rates in 2020?") != query_id("rates in 2020")
    assert canonical_text(" ＵＳ\t A\nB ") == "US A B"
    assert document_id("\tUS A B  ") == document_id("ＵＳ\nA\tB")
    assert document_id("US A B") != document_id("us A B")


def test_full_scan_excludes_later_known_positives_and_preserves_all_labels(tmp_path):
    rows = [
        {"query": "Alpha?", "positive": "A", "negative": "B"},
        {"query": " ALPHA? ", "positive": "B", "negative": "N1"},
        {"query": "alpha?", "positive": "A", "negative": " \tA\n"},
        {"query": "alpha?", "positive": "A", "negative": "N2"},
        {"query": "Beta?", "positive": " Shared\ntext ", "negative": "NB"},
        {"query": "Gamma?", "positive": "Shared text", "negative": "NG"},
        # Even a malformed negative must not discard a valid positive label.
        {"query": "Delta?", "positive": "D", "negative": None},
        {"query": " DELTA?", "positive": "E", "negative": "D"},
        {"query": "delta?", "positive": "E", "negative": "ND"},
        {"query": "Epsilon?", "positive": "E", "negative": "NE"},
    ]
    source = write_source(tmp_path, rows)
    output = tmp_path / "prepared"
    manifest = prepare_triplets(source, output, n_train=4, n_validation=1)
    splits = read_rows(output)
    selected = {row["query_id"]: row for values in splits.values() for row in values}
    assert len(selected) == 5
    assert set(row["query_id"] for row in splits["train"]).isdisjoint(
        row["query_id"] for row in splits["validation"])
    assert selected[query_id("alpha?")]["source_row_index"] in {1, 3}
    assert selected[query_id("alpha?")]["known_positive_ids"] == sorted([document_id("A"), document_id("B")])
    assert selected[query_id("delta?")]["known_positive_ids"] == sorted([document_id("D"), document_id("E")])
    assert selected[query_id("delta?")]["source_row_index"] == 8
    documents = {}
    for row in selected.values():
        assert row["negative_id"] not in row["known_positive_ids"]
        assert row["positive_id"] in row["known_positive_ids"]
        assert query_id(row["query"]) == row["query_id"]
        for kind in ("positive", "negative"):
            text, doc_id = row[kind], row[f"{kind}_id"]
            assert "doc_" + hashlib.sha256(text.encode("utf-8")).hexdigest() == doc_id
            assert documents.setdefault(doc_id, text) == text
    audit = manifest["audit"]
    assert audit["source_rows"] == 10
    assert audit["unique_normalized_queries"] == 5
    assert audit["invalid_triplet_rows"] == 1
    assert audit["identical_positive_negative_rows"] == 1
    assert audit["negative_is_other_known_positive_rows"] == 2
    assert audit["queries_with_multiple_known_positives"] == 2
    assert audit["eligible_triplet_rows"] == 6
    assert audit["eligible_unique_queries"] == 5
    assert audit["selected_unique_document_texts"] == len(documents)
    assert audit["selected_deduplicated_document_occurrences"] == 10 - len(documents)
    assert manifest["protocol"]["selection_uses_heldout_data"] is False
    assert manifest["protocol"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    for filename, artifact in manifest["artifacts"].items():
        assert artifact["sha256"] == hashlib.sha256((output / filename).read_bytes()).hexdigest()


def test_sampling_scans_every_query_and_is_reproducible_under_source_reordering(tmp_path):
    rows = ordinary_rows(200)
    # Multiple valid rows for one query test deterministic representative choice.
    rows.extend([
        {"query": "QUESTION 20?", "positive": "second positive", "negative": "second negative"},
        {"query": "Question 80?", "positive": "Relevant text 80", "negative": "third negative"},
    ])
    first = write_source(tmp_path, rows)
    second = write_source(tmp_path, list(reversed(rows)), "reordered.jsonl")
    a, b = tmp_path / "a", tmp_path / "b"
    m1 = prepare_triplets(first, a, n_train=24, n_validation=6, seed=17)
    m2 = prepare_triplets(second, b, n_train=24, n_validation=6, seed=17)
    out1, out2 = read_rows(a), read_rows(b)
    for split in out1:
        assert [{key: value for key, value in row.items() if key != "source_row_index"} for row in out1[split]] == [
            {key: value for key, value in row.items() if key != "source_row_index"} for row in out2[split]]
    assert m1["selected_query_ids_sha256"] == m2["selected_query_ids_sha256"]
    assert m1["audit"]["source_rows"] == len(rows)
    assert max(row["source_row_index"] for values in out1.values() for row in values) > 100
    third = tmp_path / "different-seed"
    m3 = prepare_triplets(first, third, n_train=24, n_validation=6, seed=18)
    assert m3["selected_query_ids_sha256"] != m1["selected_query_ids_sha256"]


def test_duplicate_rows_do_not_increase_query_sample_weight(tmp_path):
    rows = ordinary_rows(6)
    one = write_source(tmp_path, rows)
    repeated = write_source(tmp_path, rows + [rows[0]] * 100, "repeated.jsonl")
    a = prepare_triplets(one, tmp_path / "a", n_train=2, n_validation=1)
    b = prepare_triplets(repeated, tmp_path / "b", n_train=2, n_validation=1)
    assert a["selected_query_ids_sha256"] == b["selected_query_ids_sha256"]
    assert b["audit"]["repeated_normalized_query_occurrences"] == 100


def test_same_query_variants_never_satisfy_a_larger_split_request(tmp_path):
    source = write_source(tmp_path, [
        {"query": "One?", "positive": "P1", "negative": "N1"},
        {"query": " one?\t", "positive": "P2", "negative": "N2"},
    ])
    with pytest.raises(ValueError, match="Only 1 eligible unique queries"):
        prepare_triplets(source, tmp_path / "out", n_train=1, n_validation=1)
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_cached_preparation_verifies_artifacts_and_rejects_protocol_changes(tmp_path, monkeypatch):
    source = write_source(tmp_path, ordinary_rows())
    output = tmp_path / "out"
    original = prepare_triplets(source, output, n_train=3, n_validation=2)
    monkeypatch.setattr(distill_data, "_iter_triplets", lambda _: pytest.fail("Verified cached data should not rescan"))
    assert prepare_triplets(source, output, n_train=3, n_validation=2) == original
    with pytest.raises(ValueError, match="different protocol"):
        prepare_triplets(source, output, n_train=3, n_validation=2, seed=7)
    (output / "train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failure"):
        prepare_triplets(source, output, n_train=3, n_validation=2)


def test_source_integrity_and_expected_row_count_are_enforced(tmp_path):
    source = write_source(tmp_path, ordinary_rows(4))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Source SHA256"):
        prepare_triplets(source, output, n_train=1, n_validation=1, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="Source row count"):
        prepare_triplets(source, output, n_train=1, n_validation=1, expected_rows=5)
    assert not (output / "manifest.json").exists()


def test_output_cannot_overwrite_source_before_any_file_is_written(tmp_path):
    source = write_source(tmp_path, ordinary_rows(4), "validation.jsonl")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite its source"):
        prepare_triplets(source, tmp_path, n_train=1, n_validation=1)
    assert source.read_bytes() == before
    assert not (tmp_path / "train.jsonl").exists()


def test_source_mutation_between_passes_is_rejected(tmp_path, monkeypatch):
    source = write_source(tmp_path, ordinary_rows(4))
    original = distill_data._iter_triplets
    calls = 0

    def changing_iterator(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ordinary_rows(1)[0]) + "\n")
        yield from original(path)

    monkeypatch.setattr(distill_data, "_iter_triplets", changing_iterator)
    with pytest.raises(ValueError, match="Source changed"):
        prepare_triplets(source, tmp_path / "out", n_train=1, n_validation=1)
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_pinned_entry_reuses_the_agreed_download_path_without_network(tmp_path, monkeypatch):
    fixture = write_source(tmp_path, ordinary_rows(4))
    seen = []

    def fake_download(url, target, size, digest):
        seen.append((url, target, size, digest))
        return fixture

    monkeypatch.setattr(distill_data, "_download_verified", fake_download)
    monkeypatch.setattr(distill_data, "SOURCE_ROWS", 4)
    monkeypatch.setattr(distill_data, "SOURCE_SHA256", hashlib.sha256(fixture.read_bytes()).hexdigest())
    output = tmp_path / "prepared"
    manifest = distill_data.prepare_msmarco_distill(tmp_path, output, n_train=2, n_validation=1)
    assert seen[0][1] == tmp_path / "source" / "msmarco-bm25-triplet.parquet"
    assert distill_data.DATASET_REVISION in seen[0][0]
    metadata = manifest["protocol"]["source_metadata"]
    assert metadata["license"]["url"] == "https://microsoft.github.io/msmarco/#terms-and-conditions"
    assert "non-commercial research" in metadata["license"]["name"]


def test_parquet_streaming_matches_jsonl_fixture(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = ordinary_rows(10)
    source = write_source(tmp_path, rows)
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=3)
    a = prepare_triplets(source, tmp_path / "a", n_train=3, n_validation=2)
    b = prepare_triplets(parquet, tmp_path / "b", n_train=3, n_validation=2)
    assert read_rows(tmp_path / "a") == read_rows(tmp_path / "b")
    assert a["audit"] == b["audit"]


@pytest.mark.parametrize("n_train,n_validation,seed", [(0, 1, 42), (1, 0, 42), (1.5, 1, 42), (True, 1, 42)])
def test_invalid_sample_parameters_rejected_before_reading(tmp_path, n_train, n_validation, seed):
    with pytest.raises(ValueError):
        prepare_triplets(tmp_path / "missing.jsonl", tmp_path / "out",
                         n_train=n_train, n_validation=n_validation, seed=seed)
