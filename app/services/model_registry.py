import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from sentence_transformers import SentenceTransformer

from app.core.config import AppConfig, ModelConfig, get_config
from app.core.errors import EmbeddingServiceError, UnknownModelError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    alias: str
    config: ModelConfig
    model: Any  # SentenceTransformer


class ModelRegistry:
    """Loads configured models on first use and keeps the most recently used ones in memory.

    Nothing here knows about any specific model: everything comes from config/models.json.
    """

    def __init__(self, config: AppConfig, loader: Callable[..., Any] = SentenceTransformer):
        self._config = config
        self._loader = loader
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()

    def resolve(self, alias: str | None) -> str:
        """Turn an optional alias into a configured one (default when None)."""
        alias = alias or self._config.service.default_model
        if alias not in self._config.models:
            raise UnknownModelError(alias, list(self._config.models))
        return alias

    def get(self, alias: str | None = None) -> LoadedModel:
        alias = self.resolve(alias)

        with self._lock:
            if alias in self._loaded:
                self._loaded.move_to_end(alias)
                return self._loaded[alias]

            loaded = self._load(alias)
            self._loaded[alias] = loaded
            while len(self._loaded) > self._config.service.max_loaded_models:
                evicted, _ = self._loaded.popitem(last=False)
                logger.info("Model %r evicted from memory (max_loaded_models reached)", evicted)
            return loaded

    def describe(self) -> list[dict]:
        """State of every configured model, without loading anything."""
        with self._lock:
            loaded = dict(self._loaded)

        return [
            {
                "key": alias,
                "name": cfg.name,
                "default": alias == self._config.service.default_model,
                "downloaded": self._config.model_path(alias).is_dir(),
                "loaded": alias in loaded,
                "dimensions": (
                    loaded[alias].model.get_sentence_embedding_dimension() if alias in loaded else None
                ),
                "metadata": cfg.metadata,
            }
            for alias, cfg in self._config.models.items()
        ]

    def _load(self, alias: str) -> LoadedModel:
        cfg = self._config.models[alias]
        path = self._config.model_path(alias)

        if not path.is_dir():
            logger.error(
                "Model %r (%s) not found at %s. Run: python -m scripts.download_model --model %s",
                alias,
                cfg.name,
                path,
                alias,
            )
            raise EmbeddingServiceError(f"Model '{alias}' is not available")

        try:
            model = self._loader(
                str(path),
                trust_remote_code=cfg.trust_remote_code,
                local_files_only=True,
            )
        except Exception as exc:
            logger.exception("Model %r could not be loaded from %s", alias, path)
            raise EmbeddingServiceError(f"Model '{alias}' could not be loaded") from exc

        return LoadedModel(alias=alias, config=cfg, model=model)


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry(get_config())
