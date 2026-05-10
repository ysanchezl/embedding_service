from typing import Annotated, List

from pydantic import BaseModel, Field

ChunkText = Annotated[str, Field(min_length=1)]

class EmbeddingRequest(BaseModel):
    texts: List[ChunkText] = Field(..., min_length=1, max_length=64)

class EmbeddingResponse(BaseModel):
    count: int
    dimensions: int
    model: str
    embeddings: List[List[float]]

class ErrorResponse(BaseModel):
    detail: str
