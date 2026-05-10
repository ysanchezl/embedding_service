from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.services.embedding_service import EmbeddingServiceError

client = TestClient(app)


def test_embed_returns_metadata(monkeypatch):
    def fake_embed_texts(texts):
        assert texts == ["first chunk", "second chunk"]
        return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    monkeypatch.setattr(routes, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(routes.settings, "model_name", "test-model")

    response = client.post(
        "/embed",
        json={"texts": ["first chunk", "second chunk"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "count": 2,
        "dimensions": 3,
        "model": "test-model",
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }


def test_embed_rejects_empty_text_list():
    response = client.post("/embed", json={"texts": []})

    assert response.status_code == 422


def test_embed_rejects_empty_chunk():
    response = client.post("/embed", json={"texts": [""]})

    assert response.status_code == 422


def test_embed_returns_503_when_model_fails(monkeypatch):
    def fake_embed_texts(_texts):
        raise EmbeddingServiceError("Embeddings could not be generated")

    monkeypatch.setattr(routes, "embed_texts", fake_embed_texts)

    response = client.post("/embed", json={"texts": ["first chunk"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "Embeddings could not be generated"}
