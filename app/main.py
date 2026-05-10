from fastapi import FastAPI
from app.api.routes import router
from app.core.settings import get_settings

settings = get_settings()

app = FastAPI(title="BGE Embedding Service")

app.include_router(router)

@app.get("/")
def root():
    return {"status": "ok"}
