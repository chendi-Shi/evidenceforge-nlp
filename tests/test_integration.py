"""Real data/index/LambdaMART/artifact/API/CLI paths, with only neural encoding mocked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

from fastapi.testclient import TestClient
import numpy as np
import pytest

from evidenceforge.api import create_app
from evidenceforge.cli import main as cli_main
from evidenceforge.data import load_dataset
from evidenceforge.experiment import run_experiment
from evidenceforge.ranker import EvidenceRanker
from evidenceforge.retrieval import RetrievalIndex


class OfflineEncoder:
    """Deterministic normalized hashing vectors; never benchmarked as a real neural model."""
    def __init__(self, cache_dir, threads=2):
        self.metadata = {"model_id": "offline-test-fixture", "model_revision": "fixture-v1", "model_sha256": "fixture-model",
                         "tokenizer_sha256": "fixture-tokenizer", "dimension": 32, "max_length": 256,
                         "backend": "offline-test", "threads": threads}

    def encode(self, texts):
        vectors = np.zeros((len(texts), 32), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\b\w+\b", text.lower()):
                position = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % 32
                vectors[row, position] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def _audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_results.py"
    spec = importlib.util.spec_from_file_location("verify_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def offline_experiment(tmp_path_factory):
    base = tmp_path_factory.mktemp("full-offline-experiment")
    data_dir, cache_dir, output_dir = base / "data", base / "cache", base / "results"
    (data_dir / "qrels").mkdir(parents=True)
    topics = ["asthma", "diabetes", "anemia", "arthritis", "cancer", "cholera", "dementia", "eczema", "epilepsy", "influenza", "malaria", "measles"]
    docs = [{"_id": f"doc{i:02d}", "title": f"Controlled study of {topic}",
             "text": f"Investigators measured {topic} treatment response in a randomized study. The {topic} results concern patient outcomes."}
            for i, topic in enumerate(topics)]
    queries = [{"_id": f"train{i:02d}", "text": f"Evidence about {topic} treatment response"} for i, topic in enumerate(topics)]
    queries += [{"_id": f"test{i:02d}", "text": f"What does a controlled study say about {topic}?"} for i, topic in enumerate(topics[:4])]
    for name, rows in (("corpus", docs), ("queries", queries)):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for split, count in (("train", 12), ("test", 4)):
        (data_dir / "qrels" / f"{split}.tsv").write_text(
            "query-id\tcorpus-id\tscore\n" + "".join(f"{split}{i:02d}\tdoc{i:02d}\t1\n" for i in range(count)), encoding="utf-8")

    def forbid_network(*args, **kwargs):
        raise AssertionError("Integration test attempted network access")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("evidenceforge.retrieval.MiniLMEncoder", OfflineEncoder)
        patch.setattr("urllib.request.urlopen", forbid_network)
        report = run_experiment(data_dir, cache_dir, output_dir, threads=1)
        index = RetrievalIndex(load_dataset(data_dir), cache_dir, dense=True, threads=1).build()
        ranker = EvidenceRanker.load(output_dir / "model")
        yield {"data": data_dir, "cache": cache_dir, "output": output_dir, "report": report, "index": index, "ranker": ranker}


def test_offline_experiment_artifacts_and_independent_metric_audit(offline_experiment):
    bundle = offline_experiment
    report = bundle["report"]
    assert report["corpus_size"] == 12 and report["test_query_count"] == 4
    assert report["ranker_training"]["split"]["shared_positive_documents"] == 0
    assert report["ranker_training"]["validation_positive_injection_count"] == 0
    assert report["ranker_training"]["test_positive_injection_count"] == 0
    assert report["frozen_models"]["model_sha256"] == report["ranker_training"]["model_sha256"]
    assert report["ranker_training"]["split"]["split_sha256"] == report["ablation_training"]["split"]["split_sha256"]
    for name in ("protocol.json", "frozen_models.json", "report.json", "per_query.jsonl", "errors.json"):
        assert (bundle["output"] / name).exists()
    audit = _audit_module().verify_results(bundle["data"], bundle["output"])
    assert audit["ok"], audit["discrepancies"]
    assert set(audit["metrics"]) == {"bm25", "dense", "rrf", "learned", "learned_no_dense"}


def test_real_saved_model_api_and_cli_rankings_agree(offline_experiment, monkeypatch, capsys):
    bundle = offline_experiment
    query = "What does a controlled study say about asthma?"
    ranker, index = bundle["ranker"], bundle["index"]
    ranker.validate_index(index.metadata)
    expected = ranker.rerank(query, index.search(query, method="rrf", k=50))[:5]
    expected_ids = [hit["doc_id"] for hit in expected]
    with TestClient(create_app(index=index, ranker=ranker, dense=True)) as client:
        result = client.post("/search", json={"query": query, "method": "ltr", "top_k": 5})
        assert result.status_code == 200
        assert [hit["doc_id"] for hit in result.json()["results"]] == expected_ids
        assert client.post("/search", json={"query": "!!!", "method": "bm25"}).status_code == 422
    # This path reconstructs the index and loads/checks the actual on-disk model.
    with TestClient(create_app(data_dir=bundle["data"], cache_dir=bundle["cache"], model_dir=bundle["output"] / "model", threads=1)) as client:
        result = client.post("/search", json={"query": query, "method": "ltr", "top_k": 5})
        assert result.status_code == 200
        assert [hit["doc_id"] for hit in result.json()["results"]] == expected_ids
    monkeypatch.setattr("sys.argv", ["evidenceforge", "--data-dir", str(bundle["data"]), "--cache-dir", str(bundle["cache"]), "--threads", "1",
                                    "search", query, "--method", "ltr", "--top-k", "5", "--model-dir", str(bundle["output"] / "model")])
    cli_main()
    assert [hit["doc_id"] for hit in json.loads(capsys.readouterr().out)["results"]] == expected_ids


def test_trec_audit_detects_tampered_resume_numbers(offline_experiment, tmp_path):
    bundle = offline_experiment
    altered = tmp_path / "altered"
    shutil.copytree(bundle["output"], altered)
    report = json.loads((altered / "report.json").read_text())
    report["metrics"]["learned"]["ndcg@10"] += 0.125
    (altered / "report.json").write_text(json.dumps(report))
    audit = _audit_module().verify_results(bundle["data"], altered)
    assert not audit["ok"]
    assert any(item["field"] == "metrics.learned.ndcg@10" for item in audit["discrepancies"])


def test_trec_parser_uses_explicit_ranks_for_ties_and_rejects_duplicates(tmp_path):
    module, path = _audit_module(), tmp_path / "ranking.trec"
    path.write_text("q Q0 z 2 0.5 method\nq Q0 a 1 0.5 method\n")
    assert module.read_trec(path, {"q"}, {"a", "z"}) == {"q": ["a", "z"]}
    path.write_text("q Q0 a 1 0.5 method\nq Q0 a 2 0.5 method\n")
    with pytest.raises(ValueError, match="Duplicate document"):
        module.read_trec(path, {"q"}, {"a", "z"})
