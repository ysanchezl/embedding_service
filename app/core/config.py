"""Service configuration.

- config/models.json   one entry per embedding model, keyed by alias
- config/service.json  service-wide settings, including the default model alias
- environment / .env   secrets and deploy paths only (HF_TOKEN, API keys, CONFIG_DIR, MODEL_DIR)

A model's `provider` decides how it runs:
- "sentence-transformers" (default): a local Hugging Face model, downloaded to model_dir
- "openai": any server speaking the OpenAI `/embeddings` protocol (OpenAI, Groq, Mistral, Ollama, vLLM...)
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SecretStr,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
DEFAULT_PROVIDER = "sentence-transformers"


class ConfigError(Exception):
    """The configuration files are missing or invalid."""


class EnvSettings(BaseSettings):
    hf_token: SecretStr | None = None
    service_api_keys: SecretStr | None = None  # comma-separated keys clients must send (see core/security.py)
    disable_auth: bool = False  # local development only
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR or CRITICAL
    config_dir: str = "config"
    model_dir: str | None = None  # overrides service.json (useful in containers)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return level


class _Strict(BaseModel):
    # Reject unknown keys so a typo in a JSON file fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")


class _ModelBase(_Strict):
    """Settings every provider shares. Defaults are model-neutral."""

    name: str = Field(min_length=1, description="Model id: a Hugging Face repo, or the name the API expects")
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    batch_size: int = Field(32, ge=1)
    chunk_overlap_ratio: float = Field(0.12, ge=0, lt=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalModelConfig(_ModelBase):
    """A Hugging Face model run locally with sentence-transformers."""

    provider: Literal["sentence-transformers"] = "sentence-transformers"
    trust_remote_code: bool = False
    chunk_tokens: int | None = Field(None, ge=1, description="None = the model's max_seq_length")


class OpenAIModelConfig(_ModelBase):
    """Any server that implements the OpenAI `POST {base_url}/embeddings` protocol."""

    provider: Literal["openai"]
    base_url: str
    api_key_env: str = Field(min_length=1, description="Name of the environment variable holding the key")
    dimensions: int | None = Field(None, ge=1, description="Sent to the API when set; also reported by /models")
    # There is no local tokenizer, so chunk size is configured and measured by estimation.
    chunk_tokens: int = Field(ge=1)
    chars_per_token: float = Field(3, gt=0)
    timeout_seconds: float = Field(30, gt=0)
    max_retries: int = Field(3, ge=0)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


def _provider_of(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("provider", DEFAULT_PROVIDER)
    return getattr(value, "provider", None)


# Models without a "provider" are local, so existing configs keep working.
ModelConfig = Annotated[
    Union[
        Annotated[LocalModelConfig, Tag(DEFAULT_PROVIDER)],
        Annotated[OpenAIModelConfig, Tag("openai")],
    ],
    Discriminator(_provider_of),
]


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
    max_loaded_models: int = Field(2, ge=1, description="Local models kept in memory; API models cost nothing")
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

    def is_local(self, alias: str) -> bool:
        return self.models[alias].provider == DEFAULT_PROVIDER

    def model_path(self, alias: str) -> Path:
        """Local folder holding a model's files. Aliases sharing a repo share the folder."""
        if not self.is_local(alias):
            raise ValueError(f"Model '{alias}' is not a local model, it has no folder")
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


def get_api_key(env_var: str) -> str | None:
    """API key from the process environment, falling back to the .env file."""
    value = os.environ.get(env_var)
    if not value:
        from dotenv import dotenv_values

        value = dotenv_values(EnvSettings.model_config["env_file"]).get(env_var)
    return value.strip() if value and value.strip() else None


@lru_cache
def get_env() -> EnvSettings:
    return EnvSettings()


@lru_cache
def get_config() -> AppConfig:
    env = get_env()
    return load_config(env.config_dir, env.model_dir)
