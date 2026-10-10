from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.services import jobs
from app.services.embedding_service import EmbeddingServiceError

client = TestClient(app)


class FakeJobStore:
    def __init__(self, job=None, full=False):
        self.job = job
        self.full = full
        self.submitted = None

    def submit(self, documents):
        if self.full:
            raise jobs.JobQueueFull()
        self.submitted = documents
        return self.job

    def get(self, job_id):
        return self.job if self.job and self.job.id == job_id else None


def make_job(**overrides):
    values = dict(id="job-1", status="pending", done=0, total=0, created_at=0.0)
    values.update(overrides)
    return jobs.Job(**values)


def use_store(monkeypatch, store):
    monkeypatch.setattr(routes, "job_store", store)


# --- POST /embed/jobs ---------------------------------------------------


def test_create_job_returns_202_with_job_id(monkeypatch):
    store = FakeJobStore(make_job())
    use_store(monkeypatch, store)

    response = client.post("/embed/jobs", json={"documents": ["uno", "dos"]})

    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1", "status": "pending"}
    assert store.submitted == ["uno", "dos"]


def test_create_job_rejects_empty_list_and_blank_documents():
    assert client.post("/embed/jobs", json={"documents": []}).status_code == 422
    assert client.post("/embed/jobs", json={"documents": [""]}).status_code == 422
    assert client.post("/embed/jobs", json={"documents": ["   "]}).status_code == 422


def test_create_job_rejects_too_many_documents_with_413(monkeypatch):
    use_store(monkeypatch, FakeJobStore(make_job()))
    monkeypatch.setattr(routes.settings, "max_documents", 2)

    response = client.post("/embed/jobs", json={"documents": ["a", "b", "c"]})

    assert response.status_code == 413


def test_create_job_rejects_oversized_document_with_413(monkeypatch):
    use_store(monkeypatch, FakeJobStore(make_job()))
    monkeypatch.setattr(routes.settings, "max_document_chars", 5)

    response = client.post("/embed/jobs", json={"documents": ["123456"]})

    assert response.status_code == 413


def test_create_job_rejects_oversized_total_with_413(monkeypatch):
    use_store(monkeypatch, FakeJobStore(make_job()))
    monkeypatch.setattr(routes.settings, "max_total_chars", 8)

    response = client.post("/embed/jobs", json={"documents": ["12345", "12345"]})

    assert response.status_code == 413


def test_create_job_returns_429_when_queue_is_full(monkeypatch):
    use_store(monkeypatch, FakeJobStore(full=True))

    response = client.post("/embed/jobs", json={"documents": ["a"]})

    assert response.status_code == 429


# --- GET /embed/jobs/{id} -----------------------------------------------


def test_get_job_unknown_returns_404(monkeypatch):
    use_store(monkeypatch, FakeJobStore())

    assert client.get("/embed/jobs/nope").status_code == 404


def test_get_job_running_shows_progress_without_results(monkeypatch):
    use_store(monkeypatch, FakeJobStore(make_job(status="running", done=2, total=5)))

    body = client.get("/embed/jobs/job-1").json()

    assert body["status"] == "running"
    assert body["progress"] == {"done": 2, "total": 5}
    assert body["chunks"] is None


def test_get_job_done_returns_chunks_and_metadata(monkeypatch):
    chunks = [
        {"document_index": 0, "chunk_index": 0, "text": "a", "embedding": [0.1, 0.2, 0.3]},
        {"document_index": 0, "chunk_index": 1, "text": "b", "embedding": [0.4, 0.5, 0.6]},
    ]
    use_store(monkeypatch, FakeJobStore(make_job(status="done", done=2, total=2, chunks=chunks)))
    monkeypatch.setattr(routes.settings, "model_name", "test-model")

    body = client.get("/embed/jobs/job-1").json()

    assert body["status"] == "done"
    assert body["model"] == "test-model"
    assert body["count"] == 2
    assert body["dimensions"] == 3
    assert body["chunks"] == chunks


def test_get_job_failed_returns_error(monkeypatch):
    use_store(monkeypatch, FakeJobStore(make_job(status="failed", error="Embedding model could not be loaded")))

    body = client.get("/embed/jobs/job-1").json()

    assert body["status"] == "failed"
    assert body["error"] == "Embedding model could not be loaded"


# --- POST /embed/query --------------------------------------------------


def test_embed_query_uses_query_prefix_and_returns_metadata(monkeypatch):
    def fake_embed_texts(texts, is_query):
        assert texts == ["¿qué es esto?"]
        assert is_query is True
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(routes, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(routes.settings, "model_name", "test-model")

    response = client.post("/embed/query", json={"texts": ["¿qué es esto?"]})

    assert response.status_code == 200
    assert response.json() == {
        "count": 1,
        "dimensions": 3,
        "model": "test-model",
        "embeddings": [[0.1, 0.2, 0.3]],
    }


def test_embed_query_rejects_empty_list_and_empty_text():
    assert client.post("/embed/query", json={"texts": []}).status_code == 422
    assert client.post("/embed/query", json={"texts": [""]}).status_code == 422


def test_embed_query_returns_503_when_model_fails(monkeypatch):
    def fake_embed_texts(_texts, is_query):
        raise EmbeddingServiceError("Embeddings could not be generated")

    monkeypatch.setattr(routes, "embed_texts", fake_embed_texts)

    response = client.post("/embed/query", json={"texts": ["hola"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "Embeddings could not be generated"}


def test_old_embed_endpoint_is_gone():
    assert client.post("/embed", json={"texts": ["hola"]}).status_code in (404, 405)
