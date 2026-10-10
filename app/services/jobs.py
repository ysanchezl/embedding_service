import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

from app.core.settings import get_settings
from app.services import embedding_service
from app.services.embedding_service import EmbeddingServiceError

logger = logging.getLogger(__name__)

# runner(documents, on_progress) -> chunks
Runner = Callable[[list[str], Callable[[int, int], None]], list[dict]]


class JobQueueFull(Exception):
    """Raised when too many jobs are pending or running."""


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | running | done | failed
    done: int = 0
    total: int = 0
    created_at: float = 0.0
    finished_at: float | None = None
    chunks: list[dict] | None = None
    error: str | None = None


class JobStore:
    """In-memory job queue with a single worker (the model is CPU/GPU bound).

    Jobs are lost on restart and finished jobs are dropped after ttl_seconds.
    """

    def __init__(self, runner: Runner, max_pending: int, ttl_seconds: int):
        self._runner = runner
        self._max_pending = max_pending
        self._ttl_seconds = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed-job")

    def submit(self, documents: list[str]) -> Job:
        with self._lock:
            self._purge_expired()
            active = sum(1 for j in self._jobs.values() if j.status in ("pending", "running"))
            if active >= self._max_pending:
                raise JobQueueFull()

            job = Job(id=uuid.uuid4().hex, created_at=time.time())
            self._jobs[job.id] = job
            snapshot = replace(job)

        self._executor.submit(self._run, job.id, documents)
        return snapshot

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and now - job.finished_at >= self._ttl_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for name, value in fields.items():
                    setattr(job, name, value)

    def _run(self, job_id: str, documents: list[str]) -> None:
        self._update(job_id, status="running")

        def on_progress(done: int, total: int) -> None:
            self._update(job_id, done=done, total=total)

        try:
            chunks = self._runner(documents, on_progress)
        except EmbeddingServiceError as exc:
            self._update(job_id, status="failed", error=str(exc), finished_at=time.time())
        except Exception:
            logger.exception("Job %s failed unexpectedly", job_id)
            self._update(job_id, status="failed", error="Internal error", finished_at=time.time())
        else:
            self._update(job_id, status="done", chunks=chunks, finished_at=time.time())


_settings = get_settings()

job_store = JobStore(
    # Looked up at call time so the real function can be swapped in tests.
    runner=lambda documents, on_progress: embedding_service.embed_documents(documents, on_progress),
    max_pending=_settings.max_pending_jobs,
    ttl_seconds=_settings.job_ttl_seconds,
)
