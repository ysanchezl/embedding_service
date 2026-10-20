"""Every step of the flow leaves a line in the log, and no line ever contains user text or credentials."""
import logging
import threading

import httpx
import pytest

from app.core.errors import EmbeddingServiceError
from app.services.jobs import JobQueueFull

# Reused fixtures and helpers from the module tests.
from tests.test_embedding_service import setup, words  # noqa: F401
from tests.test_jobs import stores, wait_for  # noqa: F401
from tests.test_model_registry import api_registry
from tests.test_openai_backend import KEY, Server, build  # noqa: F401
from tests.test_routes import api  # noqa: F401


def lines_of(caplog, logger_name):
    return [r for r in caplog.records if r.name == logger_name]


# --- API backend ----------------------------------------------------------


def test_every_api_call_logs_its_size_attempt_status_and_duration(build, caplog):  # noqa: F811
    backend, _ = build(Server())

    with caplog.at_level(logging.INFO):
        backend.encode(["secret text one", "secret text two", "secret text three"])

    lines = [r.getMessage() for r in lines_of(caplog, "app.providers.openai_compatible")]
    assert len(lines) == 2  # batch_size is 2: two calls
    assert "Model 'm'" in lines[0] and "inputs=2" in lines[0] and "attempt=1/3" in lines[0] and "HTTP 200" in lines[0]
    assert "inputs=1" in lines[1]
    assert " ms" in lines[0]


def test_backend_logs_never_contain_the_texts_or_the_key(build, caplog):  # noqa: F811
    backend, _ = build(Server(httpx.Response(429), httpx.Response(401)))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbeddingServiceError):
            backend.encode(["texto confidencial"])

    assert "texto confidencial" not in caplog.text
    assert KEY not in caplog.text


def test_a_retried_call_logs_each_attempt(build, caplog):  # noqa: F811
    backend, _ = build(Server(httpx.Response(429, headers={"Retry-After": "1"})))

    with caplog.at_level(logging.INFO):
        backend.encode(["a"])

    lines = [r.getMessage() for r in lines_of(caplog, "app.providers.openai_compatible")]
    assert any("attempt=1/3" in line and "HTTP 429" in line for line in lines)
    assert any("attempt=2/3" in line and "HTTP 200" in line for line in lines)


def test_a_timeout_logs_the_error_type_and_the_attempt(build, caplog):  # noqa: F811
    backend, _ = build(Server(httpx.ReadTimeout("slow")))

    with caplog.at_level(logging.INFO):
        backend.encode(["a"])

    lines = [r.getMessage() for r in lines_of(caplog, "app.providers.openai_compatible")]
    assert any("ReadTimeout" in line and "attempt=1/3" in line for line in lines)


# --- embedding service ----------------------------------------------------


def test_embed_texts_logs_model_size_and_duration(setup, caplog):  # noqa: F811
    service, _ = setup()

    with caplog.at_level(logging.INFO):
        service.embed_texts(["hola", "adios"], is_query=True)

    [line] = [r.getMessage() for r in lines_of(caplog, "app.services.embedding_service")]
    assert "embed_texts" in line and "model=alpha" in line and "query=True" in line and "inputs=2" in line
    assert " ms" in line


def test_embed_documents_logs_the_chunking_result_and_the_total(setup, caplog):  # noqa: F811
    service, _ = setup()

    with caplog.at_level(logging.INFO):
        service.embed_documents([words(40), "corto"])

    lines = [r.getMessage() for r in lines_of(caplog, "app.services.embedding_service")]
    assert any("documents=2" in l and "chunks=4" in l and "model=alpha" in l for l in lines)
    assert any("embed_documents" in l and "done" in l and "chunks=4" in l and " ms" in l for l in lines)


def test_service_logs_never_contain_the_texts(setup, caplog):  # noqa: F811
    service, _ = setup()

    with caplog.at_level(logging.DEBUG):
        service.embed_texts(["informacion personal del paciente"])
        service.embed_documents(["otro documento privado"])

    assert "paciente" not in caplog.text and "privado" not in caplog.text


# --- jobs -----------------------------------------------------------------


def test_job_lifecycle_is_logged_from_creation_to_completion(stores, caplog):  # noqa: F811
    store = stores(lambda documents, model, on_progress: [{"embedding": [1.0]}] * 3)

    with caplog.at_level(logging.INFO):
        job = store.submit(["uno", "dos"], "alpha")
        wait_for(store, job.id, "done")

    lines = [r.getMessage() for r in lines_of(caplog, "app.services.jobs")]
    assert any(f"Job {job.id} created" in l and "model=alpha" in l and "documents=2" in l and "chars=6" in l for l in lines)
    assert any(f"Job {job.id} started" in l for l in lines)
    assert any(f"Job {job.id} done" in l and "chunks=3" in l for l in lines)


def test_a_failed_job_logs_the_reason_as_an_error(stores, caplog):  # noqa: F811
    def runner(documents, model, on_progress):
        raise EmbeddingServiceError("Model 'alpha' is unavailable (ConnectError)")

    store = stores(runner)

    with caplog.at_level(logging.INFO):
        job = store.submit(["a"], "alpha")
        wait_for(store, job.id, "failed")

    failures = [r for r in lines_of(caplog, "app.services.jobs") if r.levelno == logging.ERROR]
    assert failures and f"Job {job.id} failed" in failures[0].getMessage()
    assert "unavailable (ConnectError)" in failures[0].getMessage()


def test_a_rejected_job_is_logged_as_a_warning(stores, caplog):  # noqa: F811
    release = threading.Event()

    def runner(documents, model, on_progress):
        release.wait(5)
        return []

    store = stores(runner, max_pending=1)
    store.submit(["a"], "alpha")

    with caplog.at_level(logging.INFO):
        with pytest.raises(JobQueueFull):
            store.submit(["b"], "alpha")

    release.set()
    warnings = [r for r in lines_of(caplog, "app.services.jobs") if r.levelno == logging.WARNING]
    assert any("queue is full" in r.getMessage() for r in warnings)


def test_job_logs_never_contain_the_documents(stores, caplog):  # noqa: F811
    store = stores(lambda documents, model, on_progress: [])

    with caplog.at_level(logging.DEBUG):
        job = store.submit(["contenido confidencial del documento"], "alpha")
        wait_for(store, job.id, "done")

    assert "confidencial" not in caplog.text


# --- registry -------------------------------------------------------------


def test_a_model_is_logged_when_it_becomes_ready(downloaded, caplog):
    registry, _ = api_registry(downloaded, keys={"REMOTE_KEY": "k"})

    with caplog.at_level(logging.INFO):
        registry.get("local")
        registry.get("remote")
        registry.get("local")  # cached: no new line

    ready = [r.getMessage() for r in caplog.records if "ready" in r.getMessage()]
    assert len(ready) == 2
    assert any("'local'" in l and "sentence-transformers" in l for l in ready)
    assert any("'remote'" in l and "openai" in l for l in ready)


# --- routes ---------------------------------------------------------------


def test_a_503_is_logged_with_its_reason(api, caplog):  # noqa: F811
    client, state, _ = api
    state["service"].error = EmbeddingServiceError("Model 'alpha' is unavailable (ConnectError)")

    with caplog.at_level(logging.INFO):
        client.post("/embed/query", json={"texts": ["hola"]})

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("unavailable (ConnectError)" in r.getMessage() for r in errors)


def test_an_unknown_model_is_logged_as_a_warning(api, caplog):  # noqa: F811
    client, _, _ = api

    with caplog.at_level(logging.INFO):
        client.post("/embed/query", json={"texts": ["hola"], "model": "nope"})

    assert any(r.levelno == logging.WARNING and "nope" in r.getMessage() for r in caplog.records)


def test_rejected_job_requests_log_why(api, caplog):  # noqa: F811
    client, state, _ = api

    with caplog.at_level(logging.INFO):
        client.post("/embed/jobs", json={"documents": ["a", "b", "c"]})  # more than max_documents
        state["store"].full = True
        client.post("/embed/jobs", json={"documents": ["a"]})  # queue is full

    messages = [r.getMessage() for r in lines_of(caplog, "app.api.routes")]
    assert any("Too many documents" in m for m in messages)
    assert any("queue" in m.lower() for m in messages)
