# BGE Embedding Service

FastAPI service that receives a list of text chunks and returns one embedding vector per chunk.

The default model is `BAAI/bge-m3`, loaded through `sentence-transformers`.

## Features

- Batch embedding endpoint for text chunks.
- Response metadata with item count, vector dimensions, and model name.
- Input limits to protect the service from oversized requests.
- Lazy model loading, so the model is loaded only when embeddings are requested.
- Controlled `503` responses when the model cannot be loaded or embeddings cannot be generated.

## Requirements

- Python 3.10+
- pip

## Installation

```bash
pip install -r requirements.txt
```

For authenticated Hugging Face Hub downloads, set `HF_TOKEN` in your `.env` file:

```env
HF_TOKEN=your_hugging_face_token
```

The token is optional for public models, but it enables better rate limits and faster downloads.

## Run Locally

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Configuration

Configuration is defined in `app/core/settings.py`.

| Setting | Default | Description |
| --- | --- | --- |
| `model_name` | `BAAI/bge-m3` | Sentence Transformers model name. |
| `normalize` | `true` | Whether embeddings are normalized. |
| `max_chars` | `2000` | Maximum characters processed per text chunk. Longer chunks are truncated. |
| `trust_remote_code` | `false` | Whether Sentence Transformers can run custom model code from the Hub. |
| `hf_token` | `null` | Optional Hugging Face token used when loading the model. Set it as `HF_TOKEN` in `.env`. |

## API

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "ok"
}
```

### Create Embeddings

```http
POST /embed
```

Request body:

```json
{
  "texts": [
    "First chunk",
    "Second chunk"
  ]
}
```

Input constraints:

- `texts` must contain between 1 and 64 items.
- Each text item must be a non-empty string.
- Each text item is truncated to `max_chars` before embedding.

Successful response:

```json
{
  "count": 2,
  "dimensions": 1024,
  "model": "BAAI/bge-m3",
  "embeddings": [
    [0.0123, -0.0456],
    [0.0789, -0.0123]
  ]
}
```

Notes:

- `count` is the number of returned embeddings.
- `dimensions` is the length of each embedding vector.
- `embeddings` contains one vector per input text, in the same order.
- The example vectors above are shortened for readability.

Error responses:

- `422`: invalid request payload.
- `503`: model loading or embedding generation failed.

## Example

```bash
curl -X POST "http://127.0.0.1:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts":["Pruebas de tokenizacion para embeddings","Estas frases largas hay que tokenizarlas bien"]}'
```

## Tests

```bash
pytest
```

The unit tests use fakes/mocks and do not download or load the real embedding model.

## Docker

Build:

```bash
docker build -t embedding-service .
```

Run:

```bash
docker run -p 8000:8000 embedding-service
```
