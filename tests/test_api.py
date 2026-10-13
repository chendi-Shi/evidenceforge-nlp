from fastapi.testclient import TestClient

from evidenceforge.api import create_app


class FakeIndex:
    metadata = {"corpus_sha256": "offline-fixture"}

    def search(self, query, method="rrf", k=50):
        return [{"doc_id": "d1", "title": "Evidence", "text": query}][:k]


def test_search_validates_inputs_and_reports_missing_capability():
    with TestClient(create_app(index=FakeIndex(), dense=False)) as client:
        assert client.get("/health").json()["ready"]
        assert client.post("/search", json={"query": "   "}).status_code == 422
        assert client.post("/search", json={"query": "!!!"}).status_code == 422
        assert client.post("/search", json={"query": "a", "top_k": 21}).status_code == 422
        assert client.post("/search", json={"query": "a", "method": "rrf"}).status_code == 409
        result = client.post("/search", json={"query": "a", "method": "bm25"})
        assert result.status_code == 200
        assert result.json()["results"][0]["doc_id"] == "d1"
        assert "Relevance" in result.json()["notice"]


def test_ltr_requires_model(tmp_path):
    with TestClient(create_app(index=FakeIndex(), model_dir=tmp_path / "absent")) as client:
        assert client.post("/search", json={"query": "a", "method": "ltr"}).status_code == 409
