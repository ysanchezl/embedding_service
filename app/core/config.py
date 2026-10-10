"""Service configuration.

- config/models.json   one entry per embedding model, keyed by alias
- config/service.json  service-wide settings, including the default model alias
- environment / .env   secrets and deploy paths only (HF_TOKEN, CONFIG_DIR, MODEL_DIR)
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ConfigError(Exception):
    """The configuration files are missing or invalid."""


class EnvSettings(BaseSettings):
    hf_token: SecretStr | None = None
    config_dir: str = "config"
    model_dir: str | None = None  # overrides service.json (useful in containers)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class _Strict(BaseModel):
    # Reject unknown keys so a typo in a JSON file fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    """Everything specific to one embedding model. Defaults are model-neutral."""

    name: str = Field(min_length=1, description="Hugging Face repo id")
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    trust_remote_code: bool = False
    batch_size: int = Field(32, ge=1)
    chunk_tokens: int | None = Field(None, ge=1, description="None = the model's max_seq_length")
    chunk_overlap_ratio: float = Field(0.12, ge=0, lt=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LimitsConfig(_Strict):
    max_documents: int = Field(100, ge=1)
    max_document_chars: int = Field(2_000_000, ge=1)
    max_total_chars: int = Field(10_000_000, ge=1)
    max_query_texts: int = Field(64, ge=1)


class JobsConfig(_Strict):
    max_pending: int = Field(20, ge=1)
    ttl_seconds: int = Field(3600, ge=0)


class ServiceConfig(_Strict):
    default_model: str
    model_dir: str = "./models"
    max_loaded_models: int = Field(2, ge=1)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)


class AppConfig(_Strict):
    models: dict[str, ModelConfig] = Field(min_length=1)
    service: ServiceConfig

    @model_validator(mode="after")
    def _check_aliases_and_default(self):
        for alias in self.models:
            if not re.match(ALIAS_PATTERN, alias):
                raise ValueError(f"invalid model alias '{alias}': use letters, digits, '.', '_' or '-'")
        if self.service.default_model not in self.models:
            raise ValueError(
                f"default_model '{self.service.default_model}' is not in models.json "
                f"(available: {', '.join(self.models)})"
            )
        return self

    def model_path(self, alias: str) -> Path:
        """Local folder holding a model's files. Aliases sharing a repo share the folder."""
        return Path(self.service.model_dir) / self.models[alias].name.replace("/", "--")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc


def _describe(error: dict) -> str:
    location = ".".join(str(part) for part in error["loc"])
    message = error["msg"].removeprefix("Value error, ")
    return f"{location}: {message}" if location else message


def load_config(config_dir: str | Path, model_dir: str | None = None) -> AppConfig:
    directory = Path(config_dir)
    data = {
        "models": _read_json(directory / "models.json"),
        "service": _read_json(directory / "service.json"),
    }
    if model_dir and isinstance(data["service"], dict):
        data["service"]["model_dir"] = model_dir

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(_describe(err) for err in exc.errors())
        raise ConfigError(f"Invalid configuration in {directory}: {details}") from exc


@lru_cache
def get_env() -> EnvSettings:
    return EnvSettings()


@lru_cache
def get_config() -> AppConfig:
    env = get_env()
    return load_config(env.config_dir, env.model_dir)
