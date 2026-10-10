from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.routes import router
from app.core.settings import get_settings
from app.services.jobs import job_store

settings = get_settings()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    job_store.shutdown()


app = FastAPI(title="Embedding Service", lifespan=lifespan)

app.include_router(router)

@app.get("/")
def root():
    return {"status": "ok"}
