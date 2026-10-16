import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import ConfigError, get_env
from app.core.security import check_auth_config, parse_api_keys
from app.main import app

KEY_A = "key-a-0123456789abcdef"
KEY_B = "key-b-0123456789abcdef"


class FakeEnv:
    def __init__(self, keys=None, disable_auth=False):
        self.service_api_keys = SecretStr(keys) if keys is not None else None
        self.disable_auth = disable_auth


@pytest.fixture
def client():
    """The real auth dependency, with the environment faked; other dependencies stay real but are not reached."""
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_env(**kwargs):
    app.dependency_overrides[get_env] = lambda: FakeEnv(**kwargs)


PROTECTED = [
    ("get", "/models"),
    ("post", "/embed/query"),
    ("post", "/embed/jobs"),
    ("get", "/embed/jobs/some-id"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_endpoints_reject_requests_without_a_key(client, method, path):
    use_env(keys=KEY_A)

    response = getattr(client, method)(path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "Invalid or missing API key"}


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_endpoints_reject_a_wrong_key(client, method, path):
    use_env(keys=KEY_A)

    response = getattr(client, method)(path, headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401
    assert "nope" not in response.text


def test_only_the_bearer_scheme_is_accepted(client):
    use_env(keys=KEY_A)

    for header in (f"Basic {KEY_A}", KEY_A, f"Token {KEY_A}"):
        assert client.get("/models", headers={"Authorization": header}).status_code == 401


def test_a_valid_key_gets_through_the_auth_check(client):
    use_env(keys=KEY_A)

    response = client.get("/embed/jobs/unknown", headers={"Authorization": f"Bearer {KEY_A}"})

    assert response.status_code != 401  # it reaches the endpoint (404: no such job)


def test_every_configured_key_is_valid(client):
    use_env(keys=f"{KEY_A}, {KEY_B}")

    for key in (KEY_A, KEY_B):
        response = client.get("/embed/jobs/unknown", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code != 401


def test_a_key_that_is_only_a_prefix_of_a_valid_one_is_rejected(client):
    use_env(keys=KEY_A)

    response = client.get("/models", headers={"Authorization": f"Bearer {KEY_A[:-1]}"})

    assert response.status_code == 401


def test_health_check_stays_open(client):
    use_env(keys=KEY_A)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_disable_auth_lets_everything_through(client):
    use_env(keys=None, disable_auth=True)

    assert client.get("/embed/jobs/unknown").status_code != 401


# --- configuration ------------------------------------------------------


def test_parse_api_keys_splits_strips_and_drops_empty_entries():
    assert parse_api_keys(f" {KEY_A} ,, {KEY_B},") == [KEY_A, KEY_B]
    assert parse_api_keys("") == []
    assert parse_api_keys(None) == []


def test_startup_requires_keys_unless_auth_is_disabled():
    with pytest.raises(ConfigError, match="SERVICE_API_KEYS.*DISABLE_AUTH"):
        check_auth_config(FakeEnv(keys=None))

    with pytest.raises(ConfigError, match="SERVICE_API_KEYS"):
        check_auth_config(FakeEnv(keys=" , "))


def test_startup_rejects_weak_keys():
    with pytest.raises(ConfigError, match="at least 16"):
        check_auth_config(FakeEnv(keys="short"))


def test_startup_accepts_valid_keys_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        check_auth_config(FakeEnv(keys=f"{KEY_A},{KEY_B}"))

    assert caplog.text == ""


def test_disabling_auth_warns_loudly(caplog):
    with caplog.at_level(logging.WARNING):
        check_auth_config(FakeEnv(keys=None, disable_auth=True))

    assert "authentication is DISABLED" in caplog.text


def test_keys_never_appear_in_logs_or_errors(caplog):
    with caplog.at_level(logging.DEBUG):
        check_auth_config(FakeEnv(keys=KEY_A))
        with pytest.raises(ConfigError) as error:
            check_auth_config(FakeEnv(keys="short"))

    assert KEY_A not in caplog.text
    assert "short" not in str(error.value).replace("shorter", "")
