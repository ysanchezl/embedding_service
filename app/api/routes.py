from fastapi import APIRouter, Depends, HTTPException

from app.core.config import AppConfig, get_config
from app.schemas.embedding import (
    ErrorResponse,
    JobCreated,
    JobRequest,
    JobStatus,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.jobs import JobQueueFull, JobStore, get_job_store
from app.services.model_registry import ModelRegistry, get_registry

router = APIRouter()

UNKNOWN_MODEL = {422: {"model": ErrorResponse, "description": "Invalid payload or unknown model alias"}}


@router.get("/models", response_model=ModelsResponse)
def list_models(config: AppConfig = Depends(get_config), registry: ModelRegistry = Depends(get_registry)):
    return {"default": config.service.default_model, "models": registry.describe()}


@router.post(
    "/embed/jobs",
    status_code=202,
    response_model=JobCreated,
    responses={413: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, **UNKNOWN_MODEL},
)
def create_job(
    req: JobRequest,
    config: AppConfig = Depends(get_config),
    registry: ModelRegistry = Depends(get_registry),
    store: JobStore = Depends(get_job_store),
):
    limits = config.service.limits
    documents = req.documents

    if len(documents) > limits.max_documents:
        raise HTTPException(413, f"Too many documents (max {limits.max_documents})")
    if any(len(d) > limits.max_document_chars for d in documents):
        raise HTTPException(413, f"A document exceeds {limits.max_document_chars} characters")
    if sum(len(d) for d in documents) > limits.max_total_chars:
        raise HTTPException(413, f"Request exceeds {limits.max_total_chars} characters in total")

    model = registry.resolve(req.model)  # unknown alias -> 422 (see main.py)

    try:
        job = store.submit(documents, model)
    except JobQueueFull as exc:
        raise HTTPException(429, "Too many pending jobs, try again later") from exc

    return {"job_id": job.id, "status": job.status, "model": job.model}


@router.get(
    "/embed/jobs/{job_id}",
    response_model=JobStatus,
    responses={404: {"model": ErrorResponse}},
)
def get_job(job_id: str, store: JobStore = Depends(get_job_store)):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")

    response = {
        "job_id": job.id,
        "status": job.status,
        "model": job.model,
        "progress": {"done": job.done, "total": job.total},
        "error": job.error,
    }
    if job.status == "done":
        chunks = job.chunks or []
        response.update(
            count=len(chunks),
            dimensions=len(chunks[0]["embedding"]) if chunks else 0,
            chunks=chunks,
        )
    return response


@router.post(
    "/embed/query",
    response_model=QueryResponse,
    responses={413: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, **UNKNOWN_MODEL},
)
def embed_query(
    req: QueryRequest,
    config: AppConfig = Depends(get_config),
    registry: ModelRegistry = Depends(get_registry),
    service: EmbeddingService = Depends(get_embedding_service),
):
    if len(req.texts) > config.service.limits.max_query_texts:
        raise HTTPException(413, f"Too many texts (max {config.service.limits.max_query_texts})")

    model = registry.resolve(req.model)
    vectors = service.embed_texts(req.texts, model=model, is_query=True)

    return {
        "count": len(vectors),
        "dimensions": len(vectors[0]) if vectors else 0,
        "model": model,
        "embeddings": vectors,
    }
