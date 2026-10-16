from typing import Annotated, Any, List

from pydantic import BaseModel, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelKey = Annotated[
    str | None,
    Field(default=None, description="Model alias from config/models.json. Omit to use the default model."),
]


class JobRequest(BaseModel):
    # Size limits are enforced in the route (413), driven by config/service.json.
    documents: List[Text] = Field(..., min_length=1)
    model: ModelKey = None


class QueryRequest(BaseModel):
    texts: List[Text] = Field(..., min_length=1)
    model: ModelKey = None


class QueryResponse(BaseModel):
    count: int
    dimensions: int
    model: str
    embeddings: List[List[float]]


class JobCreated(BaseModel):
    job_id: str
    status: str
    model: str


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
    model: str
    progress: JobProgress
    dimensions: int | None = None
    count: int | None = None
    chunks: List[ChunkResult] | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    key: str
    name: str
    provider: str
    default: bool
    downloaded: bool | None  # null for API models: nothing to download
    loaded: bool | None  # null for API models: nothing kept in memory
    dimensions: int | None
    metadata: dict[str, Any]


class ModelsResponse(BaseModel):
    default: str
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    detail: str
