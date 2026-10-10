FROM python:3.12-slim

WORKDIR /app

# Prefer the CPU build of torch (much smaller than the CUDA one).
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# The model is NOT baked into the image: it lives in MODEL_DIR, which should be a
# persistent volume. It is downloaded on first start and reused afterwards.
ENV MODEL_DIR=/models
VOLUME /models

EXPOSE 8000

CMD ["sh", "-c", "python -m scripts.download_model && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
