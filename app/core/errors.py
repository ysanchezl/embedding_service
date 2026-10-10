class EmbeddingServiceError(Exception):
    """A model could not be loaded or could not produce vectors."""


class UnknownModelError(Exception):
    """The requested model alias is not in config/models.json."""

    def __init__(self, alias: str, available: list[str]):
        self.alias = alias
        self.available = available
        super().__init__(f"Unknown model '{alias}'. Available: {', '.join(available)}")
