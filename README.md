# Embedding Service

FastAPI service that chunks large documents by tokens and returns one embedding vector per chunk. It supports **several
models at once, local or through an API**: each request can pick one by alias, and adding a model (including a new
API endpoint) only requires editing a JSON file. Request and response formats are the same for every model.

| Provider | What it is | Memory | Needs |
| --- | --- | --- | --- |
| `openai` | Any server speaking the OpenAI `/embeddings` protocol: OpenAI, **Groq**, Mistral, Ollama, vLLM... | ~60 MB | an API key |
| `sentence-transformers` (default) | A Hugging Face model running in the service | ~1 GB | `requirements-local.txt` and a download |

The default model is `openrouter-lfm-free` (a free OpenRouter model, via API); `openrouter-bge-m3` (BAAI bge-m3, paid)
is configured as the alternative for when the OpenRouter account has credits. Texts sent to an API model **leave your infrastructure**: check the
provider's data terms before sending sensitive data.

## Features

- Models and their settings in `config/models.json`; service settings in `config/service.json`.
- Two providers behind one interface: local models and OpenAI-compatible APIs. Adding another endpoint is config only.
- API-only deployments never import torch, so they run in ~60 MB of RAM.
- Optional `model` field in requests; a configurable default model.
- Asynchronous jobs for large documents: submit, then poll for progress and result.
- Token-based chunking (paragraph, then sentence, then word boundaries) with overlap, sized from each model.
- Batched encoding, with progress reporting.
- Synchronous endpoint for search queries.
- Local models downloaded once to a folder and loaded offline afterwards; the least recently used ones are unloaded
  when `max_loaded_models` is reached.
- API calls with timeouts and retries (exponential backoff, `Retry-After` honored on `429`).
- Request limits (`413`) and a bounded job queue (`429`).
- API-key authentication for clients (`Authorization: Bearer <key>`); the service will not start without keys.
- Real errors are logged; clients only get controlled error messages.

## Requirements

- Python 3.10+

## Installation

```bash
pip install -r requirements.txt          # API models only (small)
# or, to also run local models (installs torch):
pip install -r requirements-local.txt

cp .env.example .env                     # then set OPENROUTER_API_KEY (or the key your default model needs)
python -m scripts.download_model         # downloads the default model if it is local; no-op for API models
```

The service **refuses to start** if the default model is an API model whose key is not set, or if `SERVICE_API_KEYS` is
empty (see Authentication).

## Run Locally

```bash
uvicorn app.main:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`. The service refuses to start if the config files are invalid.

## Configuration

### `config/models.json`

One entry per model. The key is the alias used in requests. `provider` is `"sentence-transformers"` (default, local) or
`"openai"` (API). Fields that do not belong to the model's provider are rejected.

**Local model** (only `name` is required; a model that needs nothing special is just `{ "name": "org/model" }`):

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

**API model** (`"provider": "openai"`), any server with `POST {base_url}/embeddings`:

```json
"groq-nomic": {
  "provider": "openai",
  "name": "nomic-embed-text-v1_5",
  "base_url": "https://api.groq.com/openai/v1",
  "api_key_env": "GROQ_API_KEY",
  "query_prefix": "search_query: ",
  "passage_prefix": "search_document: ",
  "chunk_tokens": 512
}
```

| Field | Default | Description |
| --- | --- | --- |
| `base_url` | *(required)* | API root, e.g. `https://api.openai.com/v1`. The service calls `{base_url}/embeddings`. |
| `api_key_env` | *(required)* | Name of the environment variable that holds the key. **The key itself never goes in the JSON.** |
| `name` | *(required)* | Model name the API expects. |
| `chunk_tokens` | *(required)* | Max tokens per chunk. There is no local tokenizer to derive it from. |
| `chars_per_token` | `3` | Chunk sizes are estimated from characters (no tokenizer); lower it to be more conservative. |
| `dimensions` | none | Sent to the API when set (for models that can shorten vectors) and shown in `/models`. |
| `timeout_seconds` | `30` | Per request. |
| `max_retries` | `3` | For timeouts, connection errors, `429` and `5xx`. Auth and `4xx` errors are not retried. |
| `query_prefix`, `passage_prefix`, `normalize`, `batch_size`, `chunk_overlap_ratio`, `metadata` | as above | Same meaning. `batch_size` is inputs per request. |

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
| `max_loaded_models` | Local models kept in memory; the least recently used is dropped beyond this. API models do not count. |
| `limits.*` | Request limits, `413` when exceeded. |
| `jobs.max_pending` | Pending + running jobs before `429`. |
| `jobs.ttl_seconds` | How long finished jobs are kept. |

### `.env`

Secrets and deploy paths only. See `.env.example`.

| Variable | Description |
| --- | --- |
| `SERVICE_API_KEYS` | Keys clients must send. **Required** unless `DISABLE_AUTH=true`. |
| `OPENROUTER_API_KEY`, `GROQ_API_KEY`, ... | One variable per API model, named by its `api_key_env`. Read from the environment or from `.env`. |
| `HF_TOKEN` | Hugging Face token, used only to download private or gated local models. |
| `CONFIG_DIR` | Folder with the two JSON files (default `config`). |
| `MODEL_DIR` | Overrides `model_dir` (used by the Docker image). |

### Adding a model

**An API endpoint** (OpenAI, Groq, Mistral, Ollama, vLLM...):

1. Add a block with `"provider": "openai"` to `config/models.json`, for example:
   ```json
   "openai-small": {
     "provider": "openai",
     "name": "text-embedding-3-small",
     "base_url": "https://api.openai.com/v1",
     "api_key_env": "OPENAI_API_KEY",
     "chunk_tokens": 512
   }
   ```
2. Set `OPENAI_API_KEY` in the environment or `.env`.
3. Restart. To make it the default, set `default_model` in `config/service.json`.

**A local model:**

1. Add its block to `config/models.json` (prefixes only if the model needs them) and install `requirements-local.txt`.
2. `python -m scripts.download_model --model <alias>` (or `--all`).
3. Restart the service.

Vectors from different models have different sizes and are not comparable: index and query each collection with the same
model, and update any vector column size (e.g. `vector(384)`) when you change it.

## Authentication

Every endpoint except `GET /` (health check) requires an API key:

```http
Authorization: Bearer <key>
```

- Valid keys are listed in `SERVICE_API_KEYS`, comma-separated. Give each client its own key, and rotate by adding the
  new one before removing the old one.
- Generate a key: `python -c "import secrets; print(secrets.token_urlsafe(32))"` (minimum 16 characters).
- A missing or wrong key returns `401`. Keys are compared in constant time and never logged.
- The service **will not start without keys**, so it cannot be deployed open by mistake. For local development only,
  `DISABLE_AUTH=true` turns authentication off (and logs a warning).
- Use HTTPS in production (Render provides it): the key travels in a header.
- In Swagger (`/docs`) press **Authorize** and paste the key.

## API

**Integrating another application with this service?** See the [client integration guide](docs/uso-cliente.md) (in
Spanish): auth, exact request/response formats, limits, errors and best practices, in one document.

All endpoints below need the `Authorization` header (except the health check).

All endpoints that embed accept an optional `"model": "<alias>"`. An unknown alias returns `422` and lists the valid ones.

### List models

```http
GET /models
```

```json
{
  "default": "e5-small",
  "models": [
    { "key": "groq-nomic", "name": "nomic-embed-text-v1_5", "provider": "openai", "default": true,
      "downloaded": null, "loaded": null, "dimensions": null, "metadata": {} },
    { "key": "e5-small", "name": "intfloat/multilingual-e5-small", "provider": "sentence-transformers",
      "default": false, "downloaded": true, "loaded": false, "dimensions": null, "metadata": {} }
  ]
}
```

For API models `downloaded` and `loaded` are `null` (nothing to download or keep in memory) and `dimensions` is only
shown if configured. For local models `dimensions` is known once the model has been loaded.

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

Errors: `401` missing or wrong key; `422` empty list, blank documents or unknown model; `413` limits exceeded; `429` queue full.

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

Synchronous. Applies the model's `query_prefix`. Errors: `422`, `413` (more than `limits.max_query_texts`), `503` (a local
model is not downloaded, or an API model has no key, rejects the request or is unavailable).

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
docker run -p 8000:8000 -e SERVICE_API_KEYS=... -e OPENROUTER_API_KEY=... embedding-service
```

By default the image installs only `requirements.txt`: small, fast to build and ~60 MB of RAM, enough for API models.
To also run local models:

```bash
docker build --build-arg LOCAL_MODELS=true -t embedding-service-local .
docker run -p 8000:8000 -v embedding-models:/models embedding-service-local
```

- The image contains the default `config/` but **not** local models nor `.env`. Local models live in the `/models`
  volume: the default one is downloaded on first start (needs network) and reused afterwards. API models download
  nothing.
- To use your own models, mount your config: `-v "$(pwd)/config:/app/config:ro"`.
- Pass keys as environment variables (`-e OPENROUTER_API_KEY=...`), or in your platform's secret settings (Render, Azure...).
- `MODEL_DIR` is forced to `/models` inside the container.
