import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from app.core.config import AppConfig, ConfigError, ModelConfig, OpenAIModelConfig, get_api_key, get_config
from app.core.errors import EmbeddingServiceError, UnknownModelError
from app.providers.base import EmbeddingBackend
from app.providers.local import SentenceTransformerBackend
from app.providers.openai_compatible import OpenAICompatibleBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    alias: str
    config: ModelConfig
    backend: EmbeddingBackend


def _import_sentence_transformer() -> Callable[..., Any]:
    """Imported lazily so a deployment that only uses API models never loads torch."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        logger.error("Local models need sentence-transformers and torch. Install requirements-local.txt")
        raise EmbeddingServiceError("Local models are not available in this deployment") from exc
    return SentenceTransformer


class ModelRegistry:
    """Builds each configured model's backend on first use.

    Local models are kept in memory up to max_loaded_models (least recently used is dropped).
    API models hold no weights, so they are never evicted.
    Nothing here knows about any specific model: everything comes from config/models.json.
    """

    def __init__(
        self,
        config: AppConfig,
        loader: Callable[..., Any] | None = None,
        key_lookup: Callable[[str], str | None] = get_api_key,
        remote_factory: Callable[[str, OpenAIModelConfig, str], Any] = OpenAICompatibleBackend,
    ):
        self._config = config
        self._loader = loader
        self._key_lookup = key_lookup
        self._remote_factory = remote_factory
        self._local: OrderedDict[str, LoadedModel] = OrderedDict()
        self._remote: dict[str, LoadedModel] = {}
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
            if alias in self._remote:
                return self._remote[alias]
            if alias in self._local:
                self._local.move_to_end(alias)
                return self._local[alias]

            if self._config.is_local(alias):
                loaded = self._load_local(alias)
                self._local[alias] = loaded
                while len(self._local) > self._config.service.max_loaded_models:
                    evicted, _ = self._local.popitem(last=False)
                    logger.info("Model %r evicted from memory (max_loaded_models reached)", evicted)
            else:
                loaded = self._load_remote(alias)
                self._remote[alias] = loaded
            return loaded

    def check_startup(self) -> None:
        """Fail fast if the default API model has no key; just warn for the others."""
        for alias, cfg in self._config.models.items():
            if self._config.is_local(alias) or self._key_lookup(cfg.api_key_env):
                continue
            message = f"Model '{alias}' needs the {cfg.api_key_env} environment variable, which is not set"
            if alias == self._config.service.default_model:
                raise ConfigError(f"{message} (it is the default model)")
            logger.warning("%s; requests for it will fail", message)

    def describe(self) -> list[dict]:
        """State of every configured model, without loading anything."""
        with self._lock:
            local = dict(self._local)

        models = []
        for alias, cfg in self._config.models.items():
            info = {
                "key": alias,
                "name": cfg.name,
                "provider": cfg.provider,
                "default": alias == self._config.service.default_model,
                "metadata": cfg.metadata,
            }
            if self._config.is_local(alias):
                info.update(
                    downloaded=self._config.model_path(alias).is_dir(),
                    loaded=alias in local,
                    dimensions=local[alias].backend.dimensions if alias in local else None,
                )
            else:  # nothing to download or keep in memory
                info.update(downloaded=None, loaded=None, dimensions=cfg.dimensions)
            models.append(info)
        return models

    def close(self) -> None:
        for loaded in self._remote.values():
            close = getattr(loaded.backend, "close", None)
            if close:
                close()

    def _load_local(self, alias: str) -> LoadedModel:
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

        loader = self._loader or _import_sentence_transformer()
        try:
            model = loader(str(path), trust_remote_code=cfg.trust_remote_code, local_files_only=True)
        except Exception as exc:
            logger.exception("Model %r could not be loaded from %s", alias, path)
            raise EmbeddingServiceError(f"Model '{alias}' could not be loaded") from exc

        return LoadedModel(alias=alias, config=cfg, backend=SentenceTransformerBackend(model, cfg))

    def _load_remote(self, alias: str) -> LoadedModel:
        cfg = self._config.models[alias]
        api_key = self._key_lookup(cfg.api_key_env)

        if not api_key:
            logger.error("Model %r needs the %s environment variable, which is not set", alias, cfg.api_key_env)
            raise EmbeddingServiceError(f"Model '{alias}' is not available")

        return LoadedModel(alias=alias, config=cfg, backend=self._remote_factory(alias, cfg, api_key))


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry(get_config())
