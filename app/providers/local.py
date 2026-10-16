from typing import Any

from app.core.config import LocalModelConfig


class SentenceTransformerBackend:
    """A Hugging Face model running in-process with sentence-transformers."""

    def __init__(self, model: Any, config: LocalModelConfig):
        self.model = model
        self._config = config

    @property
    def input_limit(self) -> int | None:
        return self.model.max_seq_length

    @property
    def special_tokens(self) -> int:
        add = getattr(self.model.tokenizer, "num_special_tokens_to_add", None)
        return add(False) if add else 0

    @property
    def dimensions(self) -> int | None:
        # Renamed in recent sentence-transformers; keep the old name as a fallback.
        get = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return get()

    def count_tokens(self, units: list[str]) -> list[int]:
        encoded = self.model.tokenizer(units, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=self._config.normalize,
            batch_size=self._config.batch_size,
        )
        return vectors.tolist()
