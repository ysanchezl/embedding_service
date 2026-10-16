import logging
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import ConfigError
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
    assert loaded.backend.model.path == str(config.model_path("alpha"))
    assert loaded.backend.model.kwargs == {"trust_remote_code": True, "local_files_only": True}
    assert registry.get("beta").backend.model.kwargs["trust_remote_code"] is False


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
        {"key": "alpha", "name": "org/alpha", "provider": "sentence-transformers", "default": True,
         "downloaded": True, "loaded": False, "dimensions": None, "metadata": {"note": "x"}},
        {"key": "beta", "name": "org/beta", "provider": "sentence-transformers", "default": False,
         "downloaded": False, "loaded": False, "dimensions": None, "metadata": {}},
    ]

    registry.get("alpha")
    assert registry.describe()[0]["loaded"] is True
    assert registry.describe()[0]["dimensions"] == 1


# --- API models ---------------------------------------------------------

API_MODEL = {
    "provider": "openai",
    "name": "remote-embed",
    "base_url": "https://api.example.com/v1",
    "api_key_env": "REMOTE_KEY",
    "chunk_tokens": 100,
    "dimensions": 256,
    "metadata": {"note": "remote"},
}


class RecordingFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, alias, config, api_key):
        self.calls.append((alias, config.name, api_key))
        return object()


def api_registry(downloaded, keys=None, **service):
    config = downloaded(models={"local": {"name": "org/local"}, "remote": API_MODEL}, service=service)
    factory = RecordingFactory()
    registry = ModelRegistry(
        config,
        loader=FakeSentenceTransformer,
        key_lookup=(keys or {}).get,
        remote_factory=factory,
    )
    return registry, factory


def test_api_model_is_built_with_the_key_from_its_env_var(downloaded):
    registry, factory = api_registry(downloaded, keys={"REMOTE_KEY": "secret"})

    loaded = registry.get("remote")

    assert factory.calls == [("remote", "remote-embed", "secret")]
    assert loaded.alias == "remote"
    assert registry.get("remote") is loaded  # built once
    assert len(factory.calls) == 1


def test_api_model_without_key_names_the_variable_in_the_log(downloaded, caplog):
    registry, factory = api_registry(downloaded, keys={})

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError, match="remote"):
            registry.get("remote")

    assert "REMOTE_KEY" in caplog.text
    assert factory.calls == []


def test_api_models_do_not_count_against_max_loaded_models(downloaded):
    registry, _ = api_registry(downloaded, keys={"REMOTE_KEY": "k"}, max_loaded_models=1)

    local = registry.get("local")
    registry.get("remote")

    assert registry.get("local") is local  # not evicted by the API model


def test_describe_api_model_has_no_download_or_load_state(downloaded):
    registry, _ = api_registry(downloaded, keys={"REMOTE_KEY": "k"})

    remote = next(m for m in registry.describe() if m["key"] == "remote")

    assert remote == {
        "key": "remote",
        "name": "remote-embed",
        "provider": "openai",
        "default": False,
        "metadata": {"note": "remote"},
        "downloaded": None,
        "loaded": None,
        "dimensions": 256,
    }


def test_startup_fails_if_default_api_model_has_no_key(downloaded):
    config = downloaded(models={"remote": API_MODEL, "local": {"name": "org/local"}})
    registry = ModelRegistry(config, key_lookup={}.get)

    with pytest.raises(ConfigError, match="REMOTE_KEY.*default"):
        registry.check_startup()


def test_startup_only_warns_for_non_default_api_models(downloaded, caplog):
    config = downloaded(models={"local": {"name": "org/local"}, "remote": API_MODEL})
    registry = ModelRegistry(config, key_lookup={}.get)

    with caplog.at_level(logging.WARNING):
        registry.check_startup()

    assert "REMOTE_KEY" in caplog.text


def test_startup_passes_when_keys_are_present(downloaded):
    config = downloaded(models={"remote": API_MODEL})
    ModelRegistry(config, key_lookup={"REMOTE_KEY": "k"}.get).check_startup()


def test_local_model_without_sentence_transformers_gives_a_clear_error(downloaded, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail
    registry = ModelRegistry(downloaded())

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError, match="not available in this deployment"):
            registry.get("alpha")

    assert "requirements-local.txt" in caplog.text


def test_api_only_usage_never_imports_torch():
    """A deployment with only API models must not pay for torch's memory."""
    code = "\n".join(
        [
            "import sys",
            "from app.main import app",
            "from app.services.model_registry import get_registry",
            "get_registry()",
            "bad = [m for m in ('torch', 'sentence_transformers') if m in sys.modules]",
            "assert not bad, bad",
        ]
    )
    root = Path(__file__).parent.parent

    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
