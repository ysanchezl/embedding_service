"""Logging setup: one readable line per event on stdout, tagged with the request id.

Render (and any container platform) shows whatever the process writes to stdout/stderr,
so this is what an operator sees when something goes wrong. Request bodies, texts and
credentials are never logged.
"""
import logging
import sys
from contextvars import ContextVar

from app.core.security import describe_auth

HANDLER_TAG = "_embedding_service_handler"
FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Set per request by RequestLoggingMiddleware; "-" for background work (e.g. the job worker).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger. Safe to call more than once."""
    name = str(level).upper()
    if name not in VALID_LEVELS:
        raise ValueError(f"Invalid LOG_LEVEL '{level}'. Use one of: {', '.join(VALID_LEVELS)}")

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, HANDLER_TAG, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter(FORMAT))
    setattr(handler, HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(name)

    # These libraries log every HTTP call with their own format; the service logs its own.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def startup_summary(config, env) -> str:
    """What the service is running with. Never includes secrets."""
    models = ", ".join(f"{alias} ({cfg.provider})" for alias, cfg in config.models.items())
    return (
        f"default model: {config.service.default_model} | models: {models} | "
        f"auth: {describe_auth(env)} | log level: {env.log_level}"
    )
