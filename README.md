# Embedding Service

FastAPI service that chunks large documents by tokens and returns one embedding vector per chunk. Works with any
`sentence-transformers` model: switching models only requires editing `.env`.

The reference model is `intfloat/multilingual-e5-small` (~470 MB, good Spanish, runs well on CPU).

## Features

- Asynchronous jobs for large documents: submit, then poll for progress and result.
- Token-based chunking (paragraph, then sentence, then word boundaries) with overlap, sized from the model itself.
- Batched encoding, with progress reporting.
- Synchronous endpoint for search queries.
- Model downloaded once to a local folder and loaded offline afterwards.
- Request limits (`413`) and a bounded job queue (`429`).
- Real errors are logged; clients only get controlled error messages.

## Requirements

- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit it
python -m scripts.download_model
```

`download_model` stores the model in `MODEL_DIR` (default `./models`, not versioned). It skips the download if the
model is already there; use `--force` to download again. `HF_TOKEN` is only needed for private or gated models.

## Run Locally

```bash
uvicorn app.main:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`

## Configuration

All settings are environment variables (or `.env`). Defaults are model-neutral; nothing model-specific is in the code.

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | *(required)* | Hugging Face model id, e.g. `intfloat/multilingual-e5-small`. |
| `MODEL_DIR` | `./models` | Folder where models are stored. Each model goes in its own subfolder. |
| `QUERY_PREFIX` | empty | Text prepended to queries. E5 models need `"query: "`. Quote values with trailing spaces. |
| `PASSAGE_PREFIX` | empty | Text prepended to document chunks. E5 models need `"passage: "`. |
| `NORMALIZE` | `true` | Whether embeddings are L2-normalized. |
| `BATCH_SIZE` | `32` | Chunks encoded per batch. |
| `CHUNK_TOKENS` | model's `max_seq_length` | Max tokens per chunk (prefix and special tokens are subtracted). |
| `CHUNK_OVERLAP_RATIO` | `0.12` | Fraction of a chunk repeated at the start of the next one. |
| `MAX_DOCUMENTS` | `100` | Documents per job request. |
| `MAX_DOCUMENT_CHARS` | `2000000` | Characters per document. |
| `MAX_TOTAL_CHARS` | `10000000` | Characters per request. |
| `MAX_PENDING_JOBS` | `20` | Pending + running jobs before returning `429`. |
| `JOB_TTL_SECONDS` | `3600` | How long finished jobs are kept. |
| `TRUST_REMOTE_CODE` | `false` | Allow custom model code from the Hub. Some models need `true`. |
| `HF_TOKEN` | none | Hugging Face token, used only to download the model. |

### Changing the model

1. Edit `MODEL_NAME`, the two prefixes (empty if the model does not use them) and `TRUST_REMOTE_CODE` in `.env`.
2. Run `python -m scripts.download_model`.
3. Restart the service.

The vector size comes from the model, so update any vector column (e.g. `vector(384)`) and re-index existing data.

## API

### Create a job

```http
POST /embed/jobs
```

```json
{ "documents": ["Full text of document 1...", "Full text of document 2..."] }
```

Returns `202`:

```json
{ "job_id": "5a7a5e62...", "status": "pending" }
```

Errors: `422` empty list or blank documents, `413` limits exceeded, `429` queue full.

### Get a job

```http
GET /embed/jobs/{job_id}
```

While running:

```json
{ "job_id": "5a7a5e62...", "status": "running", "progress": { "done": 8, "total": 21 }, "chunks": null }
```

When finished (`status` is `done`):

```json
{
  "job_id": "5a7a5e62...",
  "status": "done",
  "progress": { "done": 21, "total": 21 },
  "model": "intfloat/multilingual-e5-small",
  "dimensions": 384,
  "count": 21,
  "chunks": [
    { "document_index": 0, "chunk_index": 0, "text": "...", "embedding": [0.012, -0.045] }
  ]
}
```

`status` is one of `pending`, `running`, `done`, `failed` (with `error`). Unknown or expired jobs return `404`.

Jobs live in memory with a single worker: they are lost if the service restarts, and finished jobs expire after
`JOB_TTL_SECONDS`.

### Embed search queries

```http
POST /embed/query
```

```json
{ "texts": ["¿qué riesgos tiene la IA en medicina?"] }
```

Synchronous. Applies `QUERY_PREFIX`. Between 1 and 64 non-empty texts. Errors: `422`, `503` (model unavailable).

### Health check

```http
GET /
```

## Example

```bash
curl -X POST http://127.0.0.1:8000/embed/jobs \
  -H "Content-Type: application/json" \
  -d '{"documents":["Texto largo del documento..."]}'

curl http://127.0.0.1:8000/embed/jobs/<job_id>
```

## Tests

```bash
pytest
```

The tests use fakes and never load the real model.

## Docker

```bash
docker build -t embedding-service .
docker run -p 8000:8000 -v "$(pwd)/.env:/app/.env:ro" -v embedding-models:/models embedding-service
```

- The image does **not** contain the model nor `.env`. The model lives in the `/models` volume: it is downloaded on
  first start (needs network) and reused afterwards.
- `.env` is mounted as a file (not `--env-file`, which would keep the quotes around the prefixes).
- `MODEL_DIR` is forced to `/models` inside the container and wins over the value in `.env`.
- To avoid any download at runtime, run `python -m scripts.download_model` once into the volume beforehand.
