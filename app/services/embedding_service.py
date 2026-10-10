import logging
from pathlib import Path
from typing import Callable

from sentence_transformers import SentenceTransformer

from app.core.settings import get_settings
from app.services.chunking import chunk_text

logger = logging.getLogger(__name__)
settings = get_settings()
model = None


class EmbeddingServiceError(Exception):
    """Raised when the embedding model cannot produce vectors."""


def model_path() -> Path:
    """Local folder where the model is stored (see scripts/download_model.py)."""
    return Path(settings.model_dir) / settings.model_name.replace("/", "--")


def get_model():
    global model

    if model is None:
        path = model_path()

        if not path.is_dir():
            logger.error(
                "Model %r not found at %s. Run: python -m scripts.download_model",
                settings.model_name,
                path,
            )
            raise EmbeddingServiceError("Embedding model could not be loaded")

        try:
            model = SentenceTransformer(
                str(path),
                trust_remote_code=settings.trust_remote_code,
                local_files_only=True,
            )
        except Exception as exc:
            logger.exception("Model %r could not be loaded from %s", settings.model_name, path)
            raise EmbeddingServiceError("Embedding model could not be loaded") from exc

    return model


def embed_texts(texts: list[str], is_query: bool = False):
    # Defensive normalization for direct service calls outside the API layer.
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    prefix = settings.query_prefix if is_query else settings.passage_prefix
    return _encode([prefix + text for text in texts])


def embed_documents(
    documents: list[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Chunk each document by tokens and embed every chunk in batches.

    Returns one dict per chunk: document_index, chunk_index, text, embedding.
    on_progress(done, total) is called after each batch.
    """
    loaded = get_model()
    tokenizer = loaded.tokenizer
    prefix = settings.passage_prefix

    def count_tokens(units: list[str]) -> list[int]:
        return [len(ids) for ids in tokenizer(units, add_special_tokens=False)["input_ids"]]

    special = getattr(tokenizer, "num_special_tokens_to_add", lambda pair=False: 0)(False)
    limit = settings.chunk_tokens or loaded.max_seq_length
    max_tokens = limit - special - (count_tokens([prefix])[0] if prefix else 0)
    if max_tokens < 1:
        raise EmbeddingServiceError("Chunk size is smaller than the model prefix")
    overlap = int(max_tokens * settings.chunk_overlap_ratio)

    chunks = [
        {"document_index": d, "chunk_index": c, "text": text}
        for d, document in enumerate(documents)
        for c, text in enumerate(chunk_text(document, count_tokens, max_tokens, overlap))
    ]

    # Similar lengths per batch means less padding.
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["text"]), reverse=True)
    total = len(chunks)

    for start in range(0, total, settings.batch_size):
        batch = order[start : start + settings.batch_size]
        vectors = _encode([prefix + chunks[i]["text"] for i in batch])
        for i, vector in zip(batch, vectors):
            chunks[i]["embedding"] = vector
        if on_progress:
            on_progress(min(start + settings.batch_size, total), total)

    return chunks


def _encode(texts: list[str]):
    try:
        vectors = get_model().encode(
            texts,
            normalize_embeddings=settings.normalize,
            batch_size=settings.batch_size,
        )
    except EmbeddingServiceError:
        raise
    except Exception as exc:
        logger.exception("Embeddings could not be generated")
        raise EmbeddingServiceError("Embeddings could not be generated") from exc

    return vectors.tolist()
