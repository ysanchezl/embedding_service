import json
from pathlib import Path

import pytest

from app.core.config import ConfigError, load_config


def write(config_dir, name, content):
    (config_dir / name).write_text(content if isinstance(content, str) else json.dumps(content))


def test_loads_models_and_service(config_dir):
    write(config_dir, "models.json", {"a": {"name": "org/a"}, "b": {"name": "org/b"}})
    write(config_dir, "service.json", {"default_model": "b"})

    config = load_config(config_dir)

    assert list(config.models) == ["a", "b"]
    assert config.service.default_model == "b"


def test_model_defaults_are_neutral(config_dir):
    config = load_config(config_dir)
    model = config.models["alpha"]

    assert model.query_prefix == ""
    assert model.passage_prefix == ""
    assert model.trust_remote_code is False
    assert model.chunk_tokens is None
    assert model.metadata == {}


def test_model_specific_values_and_metadata_are_kept(config_dir):
    write(
        config_dir,
        "models.json",
        {
            "alpha": {
                "name": "org/alpha",
                "query_prefix": "q: ",
                "passage_prefix": "p: ",
                "trust_remote_code": True,
                "chunk_tokens": 256,
                "metadata": {"license": "mit", "anything": [1, 2]},
            }
        },
    )

    model = load_config(config_dir).models["alpha"]

    assert (model.query_prefix, model.passage_prefix) == ("q: ", "p: ")
    assert model.trust_remote_code is True
    assert model.chunk_tokens == 256
    assert model.metadata == {"license": "mit", "anything": [1, 2]}


def test_default_model_must_be_configured(config_dir):
    write(config_dir, "service.json", {"default_model": "missing"})

    with pytest.raises(ConfigError, match="default_model 'missing'.*alpha"):
        load_config(config_dir)


def test_typo_in_model_field_is_rejected(config_dir):
    write(config_dir, "models.json", {"alpha": {"name": "org/alpha", "querry_prefix": "x"}})

    with pytest.raises(ConfigError, match="querry_prefix"):
        load_config(config_dir)


def test_invalid_json_names_the_file(config_dir):
    write(config_dir, "models.json", "{ not json")

    with pytest.raises(ConfigError, match="models.json"):
        load_config(config_dir)


def test_missing_file_names_the_file(config_dir):
    (config_dir / "service.json").unlink()

    with pytest.raises(ConfigError, match="service.json"):
        load_config(config_dir)


def test_at_least_one_model_is_required(config_dir):
    write(config_dir, "models.json", {})

    with pytest.raises(ConfigError):
        load_config(config_dir)


def test_alias_must_be_url_safe(config_dir):
    write(config_dir, "models.json", {"bad alias/x": {"name": "org/a"}})
    write(config_dir, "service.json", {"default_model": "bad alias/x"})

    with pytest.raises(ConfigError, match="alias"):
        load_config(config_dir)


def test_model_dir_override(config_dir, tmp_path):
    config = load_config(config_dir, model_dir=str(tmp_path / "other"))

    assert config.service.model_dir == str(tmp_path / "other")


def test_model_path_is_per_hf_name(config_dir, tmp_path):
    write(config_dir, "service.json", {"default_model": "alpha", "model_dir": str(tmp_path / "m")})

    config = load_config(config_dir)

    assert config.model_path("alpha") == Path(tmp_path / "m" / "org--alpha")


def test_repo_config_files_are_valid():
    """The config shipped in the repo must always load."""
    config = load_config(Path(__file__).parent.parent / "config")

    assert config.service.default_model in config.models


def test_error_message_has_no_empty_location(config_dir):
    write(config_dir, "service.json", {"default_model": "missing"})

    with pytest.raises(ConfigError) as error:
        load_config(config_dir)

    assert ": : " not in str(error.value)
    assert "Value error" not in str(error.value)
