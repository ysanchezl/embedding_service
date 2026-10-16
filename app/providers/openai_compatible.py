import logging
import math
import time
from typing import Any, Callable

import httpx

from app.core.config import OpenAIModelConfig
from app.core.errors import EmbeddingServiceError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 20
MAX_RETRY_AFTER_SECONDS = 60
LOGGED_BODY_CHARS = 300


class OpenAICompatibleBackend:
    """Any server that implements the OpenAI `POST {base_url}/embeddings` protocol.

    Covers OpenAI, Groq, Mistral, Ollama, vLLM... Which one is just configuration.
    """

    def __init__(
        self,
        alias: str,
        config: OpenAIModelConfig,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._alias = alias
        self._config = config
        self._url = f"{config.base_url}/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._sleep = sleep

    @property
    def input_limit(self) -> int | None:
        return self._config.chunk_tokens

    @property
    def special_tokens(self) -> int:
        return 0

    @property
    def dimensions(self) -> int | None:
        return self._config.dimensions

    def close(self) -> None:
        self._client.close()

    def count_tokens(self, units: list[str]) -> list[int]:
        """No local tokenizer: estimate from characters (configure chars_per_token to tune).

        Counts one extra character per unit for the space that joins units into a chunk,
        so the sum of the parts never underestimates the joined chunk.
        """
        per_token = self._config.chars_per_token
        return [math.ceil((len(unit.strip()) + 1) / per_token) if unit.strip() else 0 for unit in units]

    def encode(self, texts: list[str]) -> list[list[float]]:
        size = self._config.batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            vectors.extend(self._request(texts[start : start + size]))

        return [_normalized(v) for v in vectors] if self._config.normalize else vectors

    def _request(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self._config.name, "input": batch, "encoding_format": "float"}
        if self._config.dimensions:
            payload["dimensions"] = self._config.dimensions

        attempt = 0
        while True:
            retry_after = None
            try:
                response = self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:  # timeouts, connection errors...
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, len(batch))
                if response.status_code in (401, 403):
                    logger.error(
                        "Model %r: authentication rejected (HTTP %s). Check the %s environment variable",
                        self._alias,
                        response.status_code,
                        self._config.api_key_env,
                    )
                    raise EmbeddingServiceError(f"Model '{self._alias}' authentication failed")
                if response.status_code not in RETRYABLE_STATUS:
                    logger.error(
                        "Model %r: request rejected (HTTP %s): %s",
                        self._alias,
                        response.status_code,
                        response.text[:LOGGED_BODY_CHARS],
                    )
                    raise EmbeddingServiceError(
                        f"Model '{self._alias}' request was rejected (HTTP {response.status_code})"
                    )
                reason = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After")

            if attempt >= self._config.max_retries:
                logger.error("Model %r: giving up after %s attempts (%s)", self._alias, attempt + 1, reason)
                raise EmbeddingServiceError(f"Model '{self._alias}' is unavailable ({reason.split(':')[0]})")

            delay = _retry_delay(retry_after, attempt)
            logger.warning("Model %r: %s, retrying in %.0fs", self._alias, reason, delay)
            self._sleep(delay)
            attempt += 1

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (ValueError, KeyError, TypeError):
            vectors = []

        if len(vectors) != expected:
            logger.error(
                "Model %r: unexpected response (expected %s vectors): %s",
                self._alias,
                expected,
                response.text[:LOGGED_BODY_CHARS],
            )
            raise EmbeddingServiceError(f"Model '{self._alias}' returned an unexpected response")

        return vectors


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass  # an HTTP date: fall back to backoff
    return float(min(2**attempt, MAX_BACKOFF_SECONDS))


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector
