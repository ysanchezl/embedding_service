from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_config, get_env
from app.core.errors import EmbeddingServiceError, UnknownModelError
from app.core.security import check_auth_config
from app.services.jobs import get_job_store
from app.services.model_registry import get_registry


@asynccontextmanager
async def lifespan(_app: FastAPI):
    check_auth_config(get_env())  # fail fast if no API keys are configured
    get_config()  # fail fast on invalid config/models.json or config/service.json
    registry = get_registry()
    registry.check_startup()  # fail fast if the default API model has no key
    store = get_job_store()
    yield
    store.shutdown()
    registry.close()


app = FastAPI(title="Embedding Service", lifespan=lifespan)

app.include_router(router)


@app.exception_handler(UnknownModelError)
async def unknown_model_handler(_request: Request, exc: UnknownModelError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EmbeddingServiceError)
async def embedding_error_handler(_request: Request, exc: EmbeddingServiceError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/")
def root():
    return {"status": "ok"}
