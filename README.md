# Embedding Service

FastAPI service that chunks large documents by tokens and returns one embedding vector per chunk. It supports **several
`sentence-transformers` models at once**: each request can pick one by alias, and adding a model only requires editing a
JSON file. Request and response formats are the same for every model.

## Features

- Models and their settings in `config/models.json`; service settings in `config/service.json`.
- Optional `model` field in requests; a configurable default model.
- Asynchronous jobs for large documents: submit, then poll for progress and result.
- Token-based chunking (paragraph, then sentence, then word boundaries) with overlap, sized from each model.
- Batched encoding, with progress reporting.
- Synchronous endpoint for search queries.
- Models downloaded once to a local folder and loaded offline afterwards; the least recently used ones are unloaded
  when `max_loaded_models` is reached.
- Request limits (`413`) and a bounded job queue (`429`).
- Real errors are logged; clients only get controlled error messages.

## Requirements

- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env              # only needed for HF_TOKEN / path overrides
python -m scripts.download_model  # downloads the default model
```

## Run Locally

```bash
uvicorn app.main:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`. The service refuses to start if the config files are invalid.

## Configuration

### `config/models.json`

One entry per model. The key is the alias used in requests. Only `name` is required; everything else is optional and
model-neutral by default, so a model that needs nothing special is just `{ "name": "org/model" }`.

```json
{
  "e5-small": {
    "name": "intfloat/multilingual-e5-small",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "normalize": true,
    "trust_remote_code": false,
    "batch_size": 32,
    "chunk_tokens": null,
    "chunk_overlap_ratio": 0.12,
    "metadata": { "description": "any free-form data you want to expose in GET /models" }
  },
  "bge-m3": {
    "name": "BAAI/bge-m3",
    "chunk_tokens": 512
  }
}
```

| Field | Default | Description |
| --- | --- | --- |
| `name` | *(required)* | Hugging Face repo id. |
| `query_prefix` | `""` | Prepended to queries. E5 models need `"query: "`. |
| `passage_prefix` | `""` | Prepended to document chunks. E5 models need `"passage: "`. |
| `normalize` | `true` | L2-normalize the vectors. |
| `trust_remote_code` | `false` | Allow custom model code from the Hub (some models need it). |
| `batch_size` | `32` | Chunks encoded per batch. |
| `chunk_tokens` | `null` | Max tokens per chunk. `null` uses the model's `max_seq_length`; the prefix and special tokens are subtracted. |
| `chunk_overlap_ratio` | `0.12` | Fraction of a chunk repeated at the start of the next one. |
| `metadata` | `{}` | Free-form data, returned by `GET /models`. |

Unknown fields are rejected, so a typo (`querry_prefix`) fails at startup instead of being silently ignored.

### `config/service.json`

```json
{
  "default_model": "e5-small",
  "model_dir": "./models",
  "max_loaded_models": 2,
  "limits": { "max_documents": 100, "max_document_chars": 2000000, "max_total_chars": 10000000, "max_query_texts": 64 },
  "jobs": { "max_pending": 20, "ttl_seconds": 3600 }
}
```

| Field | Description |
| --- | --- |
| `default_model` | Alias used when a request has no `model`. Must exist in `models.json`. |
| `model_dir` | Where models are stored (not versioned). Each Hugging Face repo gets its own subfolder. |
| `max_loaded_models` | Models kept in memory; the least recently used is dropped beyond this. |
| `limits.*` | Request limits, `413` when exceeded. |
| `jobs.max_pending` | Pending + running jobs before `429`. |
| `jobs.ttl_seconds` | How long finished jobs are kept. |

### `.env`

Secrets and deploy paths only. See `.env.example`.

| Variable | Description |
| --- | --- |
| `HF_TOKEN` | Hugging Face token, used only to download private or gated models. |
| `CONFIG_DIR` | Folder with the two JSON files (default `config`). |
| `MODEL_DIR` | Overrides `model_dir` (used by the Docker image). |

### Adding a model

1. Add its block to `config/models.json` (prefixes only if the model needs them).
2. `python -m scripts.download_model --model <alias>` (or `--all`).
3. Restart the service. It is now available as `"model": "<alias>"` and listed in `GET /models`.

Vectors from different models have different sizes and are not comparable: index and query each collection with the same
model, and update any vector column size (e.g. `vector(384)`) when you change it.

## API

All endpoints that embed accept an optional `"model": "<alias>"`. An unknown alias returns `422` and lists the valid ones.

### List models

```http
GET /models
```

```json
{
  "default": "e5-small",
  "models": [
    { "key": "e5-small", "name": "intfloat/multilingual-e5-small", "default": true,
      "downloaded": true, "loaded": true, "dimensions": 384, "metadata": {} }
  ]
}
```

`dimensions` is only known once the model has been loaded.

### Create a job

```http
POST /embed/jobs
```

```json
{ "documents": ["Full text of document 1...", "Full text of document 2..."], "model": "bge-m3" }
```

Returns `202`:

```json
{ "job_id": "5a7a5e62...", "status": "pending", "model": "bge-m3" }
```

Errors: `422` empty list, blank documents or unknown model; `413` limits exceeded; `429` queue full.

### Get a job

```http
GET /embed/jobs/{job_id}
```

While running:

```json
{ "job_id": "5a7a5e62...", "status": "running", "model": "bge-m3", "progress": { "done": 8, "total": 21 }, "chunks": null }
```

When finished (`status` is `done`):

```json
{
  "job_id": "5a7a5e62...",
  "status": "done",
  "model": "bge-m3",
  "progress": { "done": 21, "total": 21 },
  "dimensions": 1024,
  "count": 21,
  "chunks": [
    { "document_index": 0, "chunk_index": 0, "text": "...", "embedding": [0.012, -0.045] }
  ]
}
```

`status` is one of `pending`, `running`, `done`, `failed` (with `error`). Unknown or expired jobs return `404`.

Jobs live in memory and run one at a time: they are lost if the service restarts, and finished jobs expire after
`jobs.ttl_seconds`.

### Embed search queries

```http
POST /embed/query
```

```json
{ "texts": ["¿qué riesgos tiene la IA en medicina?"], "model": "bge-m3" }
```

Synchronous. Applies the model's `query_prefix`. Errors: `422`, `413` (more than `limits.max_query_texts`), `503` (the
model is not downloaded or cannot be loaded).

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

The tests use fakes and never load a real model.

## Docker

```bash
docker build -t embedding-service .
docker run -p 8000:8000 -v embedding-models:/models embedding-service
```

- The image contains the default `config/` but **not** the models nor `.env`. Models live in the `/models` volume: the
  default model is downloaded on first start (needs network) and reused afterwards.
- To use your own models, mount your config: `-v "$(pwd)/config:/app/config:ro"`. To download a model that is not the
  default, run `docker run --rm -v embedding-models:/models -v "$(pwd)/config:/app/config:ro" embedding-service python -m scripts.download_model --all`.
- For a private model, pass the token: `-e HF_TOKEN=...`.
- `MODEL_DIR` is forced to `/models` inside the container.
