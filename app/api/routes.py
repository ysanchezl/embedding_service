from fastapi import APIRouter, HTTPException

from app.core.settings import get_settings
from app.schemas.embedding import (
    ErrorResponse,
    JobCreated,
    JobRequest,
    JobStatus,
    QueryRequest,
    QueryResponse,
)
from app.services.embedding_service import EmbeddingServiceError, embed_texts
from app.services.jobs import JobQueueFull, job_store

router = APIRouter()
settings = get_settings()


@router.post(
    "/embed/jobs",
    status_code=202,
    response_model=JobCreated,
    responses={413: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
def create_job(req: JobRequest):
    documents = req.documents

    if len(documents) > settings.max_documents:
        raise HTTPException(413, f"Too many documents (max {settings.max_documents})")
    if any(len(d) > settings.max_document_chars for d in documents):
        raise HTTPException(413, f"A document exceeds {settings.max_document_chars} characters")
    if sum(len(d) for d in documents) > settings.max_total_chars:
        raise HTTPException(413, f"Request exceeds {settings.max_total_chars} characters in total")

    try:
        job = job_store.submit(documents)
    except JobQueueFull as exc:
        raise HTTPException(429, "Too many pending jobs, try again later") from exc

    return {"job_id": job.id, "status": job.status}


@router.get(
    "/embed/jobs/{job_id}",
    response_model=JobStatus,
    responses={404: {"model": ErrorResponse}},
)
def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")

    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": {"done": job.done, "total": job.total},
        "error": job.error,
    }
    if job.status == "done":
        chunks = job.chunks or []
        response.update(
            model=settings.model_name,
            count=len(chunks),
            dimensions=len(chunks[0]["embedding"]) if chunks else 0,
            chunks=chunks,
        )
    return response


@router.post(
    "/embed/query",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}},
)
def embed_query(req: QueryRequest):
    try:
        vectors = embed_texts(req.texts, is_query=True)
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "count": len(vectors),
        "dimensions": len(vectors[0]) if vectors else 0,
        "model": settings.model_name,
        "embeddings": vectors,
    }
