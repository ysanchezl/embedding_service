import logging
import time
from functools import lru_cache
from typing import Callable

from app.core.errors import EmbeddingServiceError
from app.services.chunking import chunk_text
from app.services.model_registry import LoadedModel, ModelRegistry, get_registry

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Produces embeddings with any model known to the registry, local or remote."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def embed_texts(self, texts: list[str], model: str | None = None, is_query: bool = False) -> list[list[float]]:
        """Embed short texts as they are (search queries, or passages already sized)."""
        # Defensive normalization for direct service calls outside the API layer.
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        loaded = self._registry.get(model)
        cfg = loaded.config
        prefix = cfg.query_prefix if is_query else cfg.passage_prefix
        started = time.perf_counter()
        vectors = self._encode(loaded, [prefix + text for text in texts])
        logger.info(
            "embed_texts model=%s query=%s inputs=%d done in %.0f ms",
            loaded.alias,
            is_query,
            len(texts),
            (time.perf_counter() - started) * 1000,
        )
        return vectors

    def embed_documents(
        self,
        documents: list[str],
        model: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Chunk each document by tokens and embed every chunk in batches.

        Returns one dict per chunk: document_index, chunk_index, text, embedding.
        on_progress(done, total) is called after each batch.
        """
        loaded = self._registry.get(model)
        cfg = loaded.config
        backend = loaded.backend

        prefix_tokens = backend.count_tokens([cfg.passage_prefix])[0] if cfg.passage_prefix else 0
        limit = cfg.chunk_tokens or backend.input_limit
        max_tokens = (limit or 0) - backend.special_tokens - prefix_tokens
        if max_tokens < 1:
            raise EmbeddingServiceError(f"Chunk size of model '{loaded.alias}' is smaller than its prefix")
        overlap = int(max_tokens * cfg.chunk_overlap_ratio)

        chunks = [
            {"document_index": d, "chunk_index": c, "text": text}
            for d, document in enumerate(documents)
            for c, text in enumerate(chunk_text(document, backend.count_tokens, max_tokens, overlap))
        ]

        total = len(chunks)
        started = time.perf_counter()
        logger.info(
            "embed_documents model=%s documents=%d chars=%d -> chunks=%d (up to %d tokens each, batches of %d)",
            loaded.alias,
            len(documents),
            sum(len(d) for d in documents),
            total,
            max_tokens,
            cfg.batch_size,
        )

        # Similar lengths per batch means less padding.
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["text"]), reverse=True)

        for start in range(0, total, cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            vectors = self._encode(loaded, [cfg.passage_prefix + chunks[i]["text"] for i in batch])
            for i, vector in zip(batch, vectors):
                chunks[i]["embedding"] = vector
            if on_progress:
                on_progress(min(start + cfg.batch_size, total), total)

        logger.info(
            "embed_documents model=%s done: chunks=%d in %.0f ms",
            loaded.alias,
            total,
            (time.perf_counter() - started) * 1000,
        )
        return chunks

    @staticmethod
    def _encode(loaded: LoadedModel, texts: list[str]) -> list[list[float]]:
        try:
            return loaded.backend.encode(texts)
        except EmbeddingServiceError:
            raise  # already logged and worded for clients by the backend
        except Exception as exc:
            logger.exception("Embeddings could not be generated with model %r", loaded.alias)
            raise EmbeddingServiceError("Embeddings could not be generated") from exc


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_registry())
