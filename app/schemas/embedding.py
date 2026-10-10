from typing import Annotated, List

from pydantic import BaseModel, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class JobRequest(BaseModel):
    # Size limits are enforced in the route (413), driven by settings.
    documents: List[Text] = Field(..., min_length=1)


class QueryRequest(BaseModel):
    texts: List[Text] = Field(..., min_length=1, max_length=64)


class QueryResponse(BaseModel):
    count: int
    dimensions: int
    model: str
    embeddings: List[List[float]]


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobProgress(BaseModel):
    done: int
    total: int


class ChunkResult(BaseModel):
    document_index: int
    chunk_index: int
    text: str
    embedding: List[float]


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: JobProgress
    model: str | None = None
    dimensions: int | None = None
    count: int | None = None
    chunks: List[ChunkResult] | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    detail: str
