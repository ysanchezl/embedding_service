from fastapi import APIRouter, HTTPException
from app.schemas.embedding import EmbeddingRequest, EmbeddingResponse, ErrorResponse
from app.core.settings import get_settings
from app.services.embedding_service import EmbeddingServiceError, embed_texts

router = APIRouter()
settings = get_settings()

@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    responses={503: {"model": ErrorResponse}},
)
def embed(req: EmbeddingRequest):
    try:
        vectors = embed_texts(req.texts)
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "count": len(vectors),
        "dimensions": len(vectors[0]) if vectors else 0,
        "model": settings.model_name,
        "embeddings": vectors,
    }
