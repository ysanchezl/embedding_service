FROM python:3.12-slim

WORKDIR /app

# By default the image only needs what API models use (small, fast to build, low memory).
# Build with --build-arg LOCAL_MODELS=true to also run local models (installs CPU torch).
ARG LOCAL_MODELS=false
COPY requirements.txt requirements-local.txt ./
RUN if [ "$LOCAL_MODELS" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-local.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY app ./app
COPY scripts ./scripts
COPY config ./config

# Local models are NOT baked into the image: they live in MODEL_DIR, which should be a
# persistent volume. The default model is downloaded on first start if it is a local one.
# To change models, mount your own config over /app/config.
ENV MODEL_DIR=/models
VOLUME /models

# Logs go straight to stdout so the platform (Render...) shows them as they happen.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# --no-access-log: the app logs every request itself (with a request id and duration).
CMD ["sh", "-c", "python -m scripts.download_model && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log"]
