from typing import Protocol


class EmbeddingBackend(Protocol):
    """What the service needs from any model, local or remote."""

    @property
    def input_limit(self) -> int | None:
        """Max tokens one input can hold (None if unknown)."""

    @property
    def special_tokens(self) -> int:
        """Tokens the model adds around every input (e.g. [CLS] and [SEP])."""

    @property
    def dimensions(self) -> int | None:
        """Vector size, if known without calling the model."""

    def count_tokens(self, units: list[str]) -> list[int]:
        """Size of each text in the model's tokens (or an estimate of them)."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order. Raises EmbeddingServiceError on controlled failures."""
