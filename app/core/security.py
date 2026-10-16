"""API-key authentication for the service itself.

Clients send `Authorization: Bearer <key>`. Valid keys come from SERVICE_API_KEYS
(comma-separated, so each client can have its own and keys can be rotated).
"""
import hmac
import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import ConfigError, EnvSettings, get_env

logger = logging.getLogger(__name__)

MIN_KEY_LENGTH = 16
_bearer = HTTPBearer(auto_error=False, description="A key listed in SERVICE_API_KEYS")


def parse_api_keys(raw: str | None) -> list[str]:
    return [key.strip() for key in (raw or "").split(",") if key.strip()]


def _configured_keys(env: EnvSettings) -> list[str]:
    secret = env.service_api_keys
    return parse_api_keys(secret.get_secret_value() if secret else None)


def check_auth_config(env: EnvSettings) -> None:
    """Fail fast at startup so the service is never deployed open by mistake."""
    if env.disable_auth:
        logger.warning("DISABLE_AUTH is set: authentication is DISABLED. Never use this in production")
        return

    keys = _configured_keys(env)
    if not keys:
        raise ConfigError(
            "SERVICE_API_KEYS is not set. Set one or more comma-separated keys "
            "(or DISABLE_AUTH=true for local development only)"
        )
    if any(len(key) < MIN_KEY_LENGTH for key in keys):
        raise ConfigError(f"Every key in SERVICE_API_KEYS must be at least {MIN_KEY_LENGTH} characters long")


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    env: EnvSettings = Depends(get_env),
) -> None:
    if env.disable_auth:
        return

    supplied = (credentials.credentials if credentials else "").encode()
    # Check every key (no early exit) and compare in constant time.
    valid = False
    for key in _configured_keys(env):
        valid |= hmac.compare_digest(supplied, key.encode())

    if not valid:
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
