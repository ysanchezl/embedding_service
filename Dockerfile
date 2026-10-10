FROM python:3.12-slim

WORKDIR /app

# Prefer the CPU build of torch (much smaller than the CUDA one).
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY config ./config

# Models are NOT baked into the image: they live in MODEL_DIR, which should be a
# persistent volume. The default model is downloaded on first start and reused afterwards.
# To change models, mount your own config over /app/config.
ENV MODEL_DIR=/models
VOLUME /models

EXPOSE 8000

CMD ["sh", "-c", "python -m scripts.download_model && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
