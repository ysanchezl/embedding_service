import pytest
from fastapi.testclient import TestClient

from app.core.config import get_config
from app.core.errors import EmbeddingServiceError
from app.main import app
from app.services import jobs
from app.services.embedding_service import get_embedding_service
from app.services.jobs import get_job_store
from app.services.model_registry import ModelRegistry, get_registry
from tests.fakes import FakeSentenceTransformer


class FakeJobStore:
    def __init__(self, job=None, full=False):
        self.job = job
        self.full = full
        self.submitted = None

    def submit(self, documents, model):
        if self.full:
            raise jobs.JobQueueFull()
        self.submitted = (documents, model)
        return jobs.Job(id="job-1", model=model, created_at=0.0)

    def get(self, job_id):
        return self.job if self.job and self.job.id == job_id else None


class FakeEmbeddingService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def embed_texts(self, texts, model=None, is_query=False):
        self.calls.append({"texts": texts, "model": model, "is_query": is_query})
        if self.error:
            raise self.error
        return [[0.1, 0.2, 0.3] for _ in texts]


def make_job(**overrides):
    values = dict(id="job-1", model="alpha", status="pending", done=0, total=0, created_at=0.0)
    values.update(overrides)
    return jobs.Job(**values)


@pytest.fixture
def api(downloaded):
    """TestClient wired with a two-model config and fakes; returns (client, overrides)."""
    config = downloaded(
        models={"alpha": {"name": "org/alpha", "metadata": {"note": "x"}}, "beta": {"name": "org/beta"}},
        service={"limits": {"max_documents": 2, "max_document_chars": 10, "max_total_chars": 15, "max_query_texts": 2}},
    )
    registry = ModelRegistry(config, loader=FakeSentenceTransformer)
    state = {"store": FakeJobStore(), "service": FakeEmbeddingService()}

    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_job_store] = lambda: state["store"]
    app.dependency_overrides[get_embedding_service] = lambda: state["service"]
    yield TestClient(app), state, registry
    app.dependency_overrides.clear()


# --- POST /embed/jobs ---------------------------------------------------


def test_create_job_uses_default_model(api):
    client, state, _ = api

    response = client.post("/embed/jobs", json={"documents": ["uno", "dos"]})

    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1", "status": "pending", "model": "alpha"}
    assert state["store"].submitted == (["uno", "dos"], "alpha")


def test_create_job_with_explicit_model(api):
    client, state, _ = api

    response = client.post("/embed/jobs", json={"documents": ["uno"], "model": "beta"})

    assert response.status_code == 202
    assert response.json()["model"] == "beta"
    assert state["store"].submitted == (["uno"], "beta")


def test_create_job_unknown_model_is_422_and_lists_available(api):
    client, state, _ = api

    response = client.post("/embed/jobs", json={"documents": ["uno"], "model": "nope"})

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]
    assert "alpha, beta" in response.json()["detail"]
    assert state["store"].submitted is None


def test_create_job_rejects_empty_list_and_blank_documents(api):
    client, _, _ = api

    assert client.post("/embed/jobs", json={"documents": []}).status_code == 422
    assert client.post("/embed/jobs", json={"documents": [""]}).status_code == 422
    assert client.post("/embed/jobs", json={"documents": ["   "]}).status_code == 422


def test_create_job_limits_come_from_config_and_return_413(api):
    client, state, _ = api

    assert client.post("/embed/jobs", json={"documents": ["a", "b", "c"]}).status_code == 413  # > 2 documents
    assert client.post("/embed/jobs", json={"documents": ["x" * 11]}).status_code == 413  # > 10 chars
    assert client.post("/embed/jobs", json={"documents": ["x" * 9, "y" * 9]}).status_code == 413  # > 15 total
    assert state["store"].submitted is None


def test_create_job_returns_429_when_queue_is_full(api):
    client, state, _ = api
    state["store"].full = True

    assert client.post("/embed/jobs", json={"documents": ["a"]}).status_code == 429


# --- GET /embed/jobs/{id} -----------------------------------------------


def test_get_job_unknown_returns_404(api):
    client, _, _ = api

    assert client.get("/embed/jobs/nope").status_code == 404


def test_get_job_running_shows_progress_without_results(api):
    client, state, _ = api
    state["store"].job = make_job(status="running", done=2, total=5)

    body = client.get("/embed/jobs/job-1").json()

    assert body["status"] == "running"
    assert body["model"] == "alpha"
    assert body["progress"] == {"done": 2, "total": 5}
    assert body["chunks"] is None


def test_get_job_done_returns_chunks_and_metadata(api):
    client, state, _ = api
    chunks = [
        {"document_index": 0, "chunk_index": 0, "text": "a", "embedding": [0.1, 0.2, 0.3]},
        {"document_index": 0, "chunk_index": 1, "text": "b", "embedding": [0.4, 0.5, 0.6]},
    ]
    state["store"].job = make_job(status="done", model="beta", done=2, total=2, chunks=chunks)

    body = client.get("/embed/jobs/job-1").json()

    assert body["status"] == "done"
    assert body["model"] == "beta"
    assert body["count"] == 2
    assert body["dimensions"] == 3
    assert body["chunks"] == chunks


def test_get_job_failed_returns_error(api):
    client, state, _ = api
    state["store"].job = make_job(status="failed", error="Model 'alpha' could not be loaded")

    body = client.get("/embed/jobs/job-1").json()

    assert body["status"] == "failed"
    assert body["error"] == "Model 'alpha' could not be loaded"


# --- POST /embed/query --------------------------------------------------


def test_query_uses_query_mode_and_default_model(api):
    client, state, _ = api

    response = client.post("/embed/query", json={"texts": ["¿qué es esto?"]})

    assert response.status_code == 200
    assert response.json() == {
        "count": 1,
        "dimensions": 3,
        "model": "alpha",
        "embeddings": [[0.1, 0.2, 0.3]],
    }
    assert state["service"].calls == [{"texts": ["¿qué es esto?"], "model": "alpha", "is_query": True}]


def test_query_with_explicit_model(api):
    client, state, _ = api

    response = client.post("/embed/query", json={"texts": ["hola"], "model": "beta"})

    assert response.json()["model"] == "beta"
    assert state["service"].calls[0]["model"] == "beta"


def test_query_unknown_model_is_422(api):
    client, _, _ = api

    response = client.post("/embed/query", json={"texts": ["hola"], "model": "nope"})

    assert response.status_code == 422
    assert "Available: alpha, beta" in response.json()["detail"]


def test_query_rejects_empty_list_and_empty_text(api):
    client, _, _ = api

    assert client.post("/embed/query", json={"texts": []}).status_code == 422
    assert client.post("/embed/query", json={"texts": [""]}).status_code == 422


def test_query_limit_comes_from_config(api):
    client, _, _ = api

    assert client.post("/embed/query", json={"texts": ["a", "b", "c"]}).status_code == 413


def test_query_returns_503_when_model_fails(api):
    client, state, _ = api
    state["service"].error = EmbeddingServiceError("Model 'alpha' is not available")

    response = client.post("/embed/query", json={"texts": ["hola"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "Model 'alpha' is not available"}


# --- GET /models --------------------------------------------------------


def test_models_lists_configured_models_and_default(api):
    client, _, registry = api
    registry.get("alpha")

    body = client.get("/models").json()

    assert body["default"] == "alpha"
    assert [m["key"] for m in body["models"]] == ["alpha", "beta"]
    assert body["models"][0] == {
        "key": "alpha",
        "name": "org/alpha",
        "default": True,
        "downloaded": True,
        "loaded": True,
        "dimensions": 1,
        "metadata": {"note": "x"},
    }
    assert body["models"][1]["loaded"] is False


def test_old_embed_endpoint_is_gone(api):
    client, _, _ = api

    assert client.post("/embed", json={"texts": ["hola"]}).status_code in (404, 405)
