import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.logging_config import HANDLER_TAG, request_id_var, setup_logging, startup_summary
from app.core.request_logging import RequestLoggingMiddleware
from app.core.security import describe_auth


@pytest.fixture
def clean_root_logger():
    """setup_logging touches the root logger: restore it so other tests are unaffected."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield root
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def tagged(root):
    return [h for h in root.handlers if getattr(h, HANDLER_TAG, False)]


# --- setup --------------------------------------------------------------


def test_setup_logging_is_idempotent(clean_root_logger):
    setup_logging("INFO")
    setup_logging("INFO")

    assert len(tagged(clean_root_logger)) == 1


@pytest.mark.parametrize("name,level", [("debug", logging.DEBUG), ("INFO", logging.INFO), ("Warning", logging.WARNING)])
def test_setup_logging_sets_the_level_case_insensitively(clean_root_logger, name, level):
    setup_logging(name)

    assert clean_root_logger.level == level


def test_setup_logging_rejects_unknown_levels(clean_root_logger):
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        setup_logging("loud")


def test_noisy_libraries_are_quieted(clean_root_logger):
    setup_logging("DEBUG")

    assert logging.getLogger("httpx").level >= logging.WARNING


def test_log_lines_go_to_stdout_with_time_level_and_request_id(clean_root_logger, capsys):
    setup_logging("INFO")
    token = request_id_var.set("abc123")
    try:
        logging.getLogger("app.demo").info("hello flow")
    finally:
        request_id_var.reset(token)

    line = capsys.readouterr().out.strip()
    assert "INFO" in line and "[abc123]" in line and "app.demo" in line and "hello flow" in line
    assert line[:4].isdigit()  # starts with a timestamp


def test_request_id_is_a_dash_outside_requests(clean_root_logger, capsys):
    setup_logging("INFO")

    logging.getLogger("app.demo").info("background work")

    assert "[-]" in capsys.readouterr().out


# --- request middleware -------------------------------------------------


def make_app():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    inner = logging.getLogger("app.demo")

    @app.get("/")
    def health():
        return {"status": "ok"}

    @app.post("/work")
    def work(payload: dict):
        inner.info("inside endpoint")
        return {"ok": True}

    @app.get("/missing")
    def missing():
        raise HTTPException(404, "nope")

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    return app


@pytest.fixture
def client():
    return TestClient(make_app(), raise_server_exceptions=False)


def request_records(caplog):
    return [r for r in caplog.records if r.name == "app.request"]


def test_each_request_logs_a_start_and_an_end_with_status_and_duration(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/work", json={"a": 1})

    messages = [r.getMessage() for r in request_records(caplog)]
    assert messages[0] == "POST /work received"
    assert messages[1].startswith("POST /work -> 200 in ")
    assert messages[1].endswith(" ms")


def test_response_carries_the_request_id_and_logs_inside_the_endpoint_share_it(client, caplog, clean_root_logger, capsys):
    setup_logging("INFO")

    response = client.post("/work", json={"a": 1})

    rid = response.headers["X-Request-ID"]
    out = capsys.readouterr().out
    assert len(rid) == 8
    assert out.count(f"[{rid}]") >= 3  # received, inside endpoint, finished
    assert f"[{rid}] app.demo: inside endpoint" in out


def test_request_ids_differ_between_requests(client):
    first = client.post("/work", json={}).headers["X-Request-ID"]
    second = client.post("/work", json={}).headers["X-Request-ID"]

    assert first != second


def test_the_request_id_does_not_leak_after_the_request(client):
    client.post("/work", json={})

    assert request_id_var.get() == "-"


def test_client_errors_are_logged_as_warnings(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.get("/missing")

    assert response.headers["X-Request-ID"]
    end = request_records(caplog)[-1]
    assert end.levelno == logging.WARNING
    assert "GET /missing -> 404" in end.getMessage()


def test_unhandled_errors_are_logged_with_traceback_and_still_get_a_request_id(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert len(response.headers["X-Request-ID"]) == 8
    failure = [r for r in request_records(caplog) if r.levelno == logging.ERROR]
    assert failure and failure[0].exc_info is not None
    assert "GET /boom" in failure[0].getMessage()
    assert "kaboom" in caplog.text


def test_health_checks_do_not_flood_the_info_log(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/")
        client.get("/")

    assert request_records(caplog) == []


def test_health_checks_are_visible_at_debug(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.get("/")

    assert request_records(caplog)


def test_bodies_and_credentials_are_never_logged(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post(
            "/work",
            json={"texto": "contenido privado del usuario"},
            headers={"Authorization": "Bearer SUPER-SECRET-KEY"},
        )

    assert "contenido privado" not in caplog.text
    assert "SUPER-SECRET-KEY" not in caplog.text


# --- auth failures ------------------------------------------------------


class FakeEnv:
    def __init__(self, keys=None, disable_auth=False, log_level="INFO"):
        self.service_api_keys = SecretStr(keys) if keys is not None else None
        self.disable_auth = disable_auth
        self.log_level = log_level


def test_describe_auth():
    assert describe_auth(FakeEnv(keys="aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb")) == "enabled (2 keys)"
    assert describe_auth(FakeEnv(keys="aaaaaaaaaaaaaaaa")) == "enabled (1 key)"
    assert describe_auth(FakeEnv(disable_auth=True)) == "DISABLED"


def test_startup_summary_lists_models_and_never_the_keys(downloaded):
    config = downloaded(
        models={
            "remote": {"provider": "openai", "name": "m", "base_url": "https://x/v1", "api_key_env": "K", "chunk_tokens": 10},
            "local": {"name": "org/local"},
        }
    )
    env = FakeEnv(keys="super-secret-key-0123456789", log_level="DEBUG")

    text = startup_summary(config, env)

    assert "default model: remote" in text
    assert "remote (openai)" in text and "local (sentence-transformers)" in text
    assert "auth: enabled (1 key)" in text
    assert "log level: DEBUG" in text
    assert "super-secret-key" not in text
