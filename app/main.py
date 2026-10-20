import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_config, get_env
from app.core.errors import EmbeddingServiceError, UnknownModelError
from app.core.logging_config import setup_logging, startup_summary
from app.core.request_logging import RequestLoggingMiddleware
from app.core.security import check_auth_config
from app.services.jobs import get_job_store
from app.services.model_registry import get_registry

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    env = get_env()
    setup_logging(env.log_level)
    check_auth_config(env)  # fail fast if no API keys are configured
    config = get_config()  # fail fast on invalid config/models.json or config/service.json
    registry = get_registry()
    registry.check_startup()  # fail fast if the default API model has no key
    store = get_job_store()
    logger.info("Service starting | %s", startup_summary(config, env))
    yield
    logger.info("Service stopping")
    store.shutdown()
    registry.close()


app = FastAPI(title="Embedding Service", lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(router)


@app.exception_handler(UnknownModelError)
async def unknown_model_handler(_request: Request, exc: UnknownModelError):
    logger.warning("Request rejected (422): %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EmbeddingServiceError)
async def embedding_error_handler(_request: Request, exc: EmbeddingServiceError):
    logger.error("Request failed (503): %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/")
def root():
    return {"status": "ok"}
