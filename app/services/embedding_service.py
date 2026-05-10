from sentence_transformers import SentenceTransformer

from app.core.settings import get_settings

settings = get_settings()
model = None


class EmbeddingServiceError(Exception):
    """Raised when the embedding model cannot produce vectors."""


def get_model():
    global model

    if model is None:
        try:
            model = SentenceTransformer(
                settings.model_name,
                trust_remote_code=settings.trust_remote_code,
                token=settings.hf_token,
            )
        except Exception as exc:
            raise EmbeddingServiceError("Embedding model could not be loaded") from exc

    return model


def embed_texts(texts: list[str]):
    # Defensive normalization for direct service calls outside the API layer.
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    texts = [text[:settings.max_chars] for text in texts]

    try:
        vectors = get_model().encode(
            texts,
            normalize_embeddings=settings.normalize,
        )
    except EmbeddingServiceError:
        raise
    except Exception as exc:
        raise EmbeddingServiceError("Embeddings could not be generated") from exc

    return vectors.tolist()
