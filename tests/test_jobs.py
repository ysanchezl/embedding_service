import threading
import time

import pytest

from app.core.errors import EmbeddingServiceError
from app.services.jobs import JobQueueFull, JobStore


def wait_for(store, job_id, status, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job.status == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {status}, is {store.get(job_id).status}")


@pytest.fixture
def stores():
    created = []

    def make(runner, max_pending=10, ttl_seconds=3600):
        store = JobStore(runner, max_pending=max_pending, ttl_seconds=ttl_seconds)
        created.append(store)
        return store

    yield make
    for store in created:
        store.shutdown()


def test_job_goes_pending_running_done_with_progress(stores):
    started = threading.Event()
    release = threading.Event()

    def runner(documents, model, on_progress):
        started.set()
        release.wait(5)
        on_progress(1, 2)
        on_progress(2, 2)
        return [{"document_index": 0, "chunk_index": 0, "text": "a", "embedding": [1.0]}]

    store = stores(runner)
    job = store.submit(["a"], "alpha")
    assert job.status in ("pending", "running")

    started.wait(5)
    assert store.get(job.id).status == "running"

    release.set()
    done = wait_for(store, job.id, "done")

    assert (done.done, done.total) == (2, 2)
    assert done.chunks[0]["text"] == "a"
    assert done.finished_at is not None


def test_runner_receives_documents_and_the_requested_model(stores):
    received = {}

    def runner(documents, model, on_progress):
        received.update(documents=documents, model=model)
        return []

    store = stores(runner)
    job = store.submit(["uno", "dos"], "beta")
    wait_for(store, job.id, "done")

    assert received == {"documents": ["uno", "dos"], "model": "beta"}
    assert store.get(job.id).model == "beta"


def test_job_fails_with_service_error_message(stores):
    def runner(documents, model, on_progress):
        raise EmbeddingServiceError("Model 'alpha' could not be loaded")

    store = stores(runner)
    job = store.submit(["a"], "alpha")

    failed = wait_for(store, job.id, "failed")
    assert failed.error == "Model 'alpha' could not be loaded"


def test_unexpected_error_is_hidden_from_client_but_job_fails(stores):
    def runner(documents, model, on_progress):
        raise RuntimeError("secret internal detail")

    store = stores(runner)
    job = store.submit(["a"], "alpha")

    failed = wait_for(store, job.id, "failed")
    assert failed.error == "Internal error"


def test_queue_full_rejects_new_jobs(stores):
    release = threading.Event()

    def runner(documents, model, on_progress):
        release.wait(5)
        return []

    store = stores(runner, max_pending=2)
    store.submit(["a"], "alpha")
    store.submit(["b"], "alpha")

    with pytest.raises(JobQueueFull):
        store.submit(["c"], "alpha")

    release.set()


def test_finished_jobs_expire_after_ttl(stores):
    store = stores(lambda documents, model, on_progress: [], ttl_seconds=0)
    job = store.submit(["a"], "alpha")
    wait_for(store, job.id, "done")

    time.sleep(0.01)
    store.submit(["b"], "alpha")  # purges expired jobs

    assert store.get(job.id) is None


def test_get_unknown_job_returns_none(stores):
    store = stores(lambda documents, model, on_progress: [])

    assert store.get("nope") is None
