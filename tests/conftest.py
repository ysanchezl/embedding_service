import json

import pytest

from app.core.config import AppConfig


@pytest.fixture
def make_config(tmp_path):
    """Build an AppConfig from plain dicts, with models stored under tmp_path."""

    def build(models=None, service=None):
        models = models or {
            "alpha": {"name": "org/alpha", "query_prefix": "query: ", "passage_prefix": "passage: "},
            "beta": {"name": "org/beta"},
        }
        service = {"default_model": next(iter(models)), "model_dir": str(tmp_path / "models"), **(service or {})}
        return AppConfig.model_validate({"models": models, "service": service})

    return build


@pytest.fixture
def config_dir(tmp_path):
    """A config directory on disk with valid files; tests overwrite what they need."""
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "models.json").write_text(json.dumps({"alpha": {"name": "org/alpha"}}))
    (directory / "service.json").write_text(json.dumps({"default_model": "alpha"}))
    return directory


@pytest.fixture
def downloaded(make_config):
    """Make a config whose models exist on disk (empty folders are enough for the fake loader)."""

    def build(**kwargs):
        config = make_config(**kwargs)
        for alias in config.models:
            if config.is_local(alias):  # API models have nothing on disk
                config.model_path(alias).mkdir(parents=True, exist_ok=True)
        return config

    return build
