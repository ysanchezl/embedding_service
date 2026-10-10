import logging

import pytest

from app.core.errors import EmbeddingServiceError, UnknownModelError
from app.services.model_registry import ModelRegistry
from tests.fakes import FakeSentenceTransformer


def registry_for(config, loader=FakeSentenceTransformer):
    return ModelRegistry(config, loader=loader)


def test_resolve_uses_default_when_no_alias(make_config):
    registry = registry_for(make_config())

    assert registry.resolve(None) == "alpha"
    assert registry.resolve("beta") == "beta"


def test_resolve_unknown_alias_lists_available(make_config):
    registry = registry_for(make_config())

    with pytest.raises(UnknownModelError) as error:
        registry.resolve("nope")

    assert error.value.available == ["alpha", "beta"]
    assert "nope" in str(error.value)


def test_loads_from_local_folder_offline_with_model_settings(downloaded):
    config = downloaded(
        models={"alpha": {"name": "org/alpha", "trust_remote_code": True}, "beta": {"name": "org/beta"}}
    )
    registry = registry_for(config)

    loaded = registry.get("alpha")

    assert loaded.alias == "alpha"
    assert loaded.model.path == str(config.model_path("alpha"))
    assert loaded.model.kwargs == {"trust_remote_code": True, "local_files_only": True}
    assert registry.get("beta").model.kwargs["trust_remote_code"] is False


def test_get_returns_default_model_without_alias(downloaded):
    registry = registry_for(downloaded())

    assert registry.get().alias == "alpha"


def test_model_is_loaded_once_and_cached(downloaded):
    calls = []

    def loader(path, **kwargs):
        calls.append(path)
        return FakeSentenceTransformer(path, **kwargs)

    registry = registry_for(downloaded(), loader=loader)

    first = registry.get("alpha")
    second = registry.get("alpha")

    assert first is second
    assert len(calls) == 1


def test_least_recently_used_model_is_evicted(downloaded):
    config = downloaded(service={"max_loaded_models": 1})
    registry = registry_for(config)

    first = registry.get("alpha")
    registry.get("beta")
    reloaded = registry.get("alpha")

    assert reloaded is not first
    assert registry.describe()[1]["loaded"] is False  # beta was evicted


def test_recently_used_model_survives_eviction(downloaded):
    config = downloaded(
        models={"a": {"name": "org/a"}, "b": {"name": "org/b"}, "c": {"name": "org/c"}},
        service={"max_loaded_models": 2},
    )
    registry = registry_for(config)

    registry.get("a")
    registry.get("b")
    registry.get("a")  # a is now the most recent
    registry.get("c")  # evicts b

    loaded = {m["key"]: m["loaded"] for m in registry.describe()}
    assert loaded == {"a": True, "b": False, "c": True}


def test_missing_model_logs_download_command_and_raises(make_config, caplog):
    registry = registry_for(make_config())

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError, match="beta"):
            registry.get("beta")

    assert "org/beta" in caplog.text
    assert "scripts.download_model --model beta" in caplog.text


def test_loader_failure_is_logged_and_wrapped(downloaded, caplog):
    def broken(path, **kwargs):
        raise OSError("corrupted weights")

    registry = registry_for(downloaded(), loader=broken)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError):
            registry.get("alpha")

    assert "corrupted weights" in caplog.text


def test_describe_reports_state_without_loading(downloaded, make_config):
    config = downloaded(
        models={
            "alpha": {"name": "org/alpha", "metadata": {"note": "x"}},
            "beta": {"name": "org/beta"},
        }
    )
    config.model_path("beta").rmdir()
    registry = registry_for(config)

    first = registry.describe()

    assert first == [
        {"key": "alpha", "name": "org/alpha", "default": True, "downloaded": True, "loaded": False,
         "dimensions": None, "metadata": {"note": "x"}},
        {"key": "beta", "name": "org/beta", "default": False, "downloaded": False, "loaded": False,
         "dimensions": None, "metadata": {}},
    ]

    registry.get("alpha")
    assert registry.describe()[0]["loaded"] is True
    assert registry.describe()[0]["dimensions"] == 1
