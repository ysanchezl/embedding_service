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


# --- providers ----------------------------------------------------------

OPENAI = {
    "provider": "openai",
    "name": "text-embedding-3-small",
    "base_url": "https://api.example.com/v1",
    "api_key_env": "EXAMPLE_API_KEY",
    "chunk_tokens": 512,
}


def with_models(config_dir, models, default="alpha"):
    write(config_dir, "models.json", models)
    write(config_dir, "service.json", {"default_model": default})
    return load_config(config_dir)


def test_models_are_local_sentence_transformers_by_default(config_dir):
    config = load_config(config_dir)

    assert config.models["alpha"].provider == "sentence-transformers"
    assert config.is_local("alpha")


def test_openai_compatible_model_loads_with_its_own_fields(config_dir):
    extra = {**OPENAI, "dimensions": 256, "chars_per_token": 4, "timeout_seconds": 10, "max_retries": 1,
             "query_prefix": "q: "}
    config = with_models(config_dir, {"alpha": extra})
    model = config.models["alpha"]

    assert model.provider == "openai"
    assert (model.base_url, model.api_key_env, model.dimensions) == ("https://api.example.com/v1", "EXAMPLE_API_KEY", 256)
    assert (model.chars_per_token, model.timeout_seconds, model.max_retries) == (4, 10, 1)
    assert model.query_prefix == "q: "
    assert not config.is_local("alpha")


def test_openai_defaults(config_dir):
    model = with_models(config_dir, {"alpha": OPENAI}).models["alpha"]

    assert model.dimensions is None
    assert model.chars_per_token == 3
    assert model.max_retries == 3
    assert model.normalize is True


@pytest.mark.parametrize("missing", ["base_url", "api_key_env", "chunk_tokens", "name"])
def test_openai_model_requires_its_fields(config_dir, missing):
    model = {k: v for k, v in OPENAI.items() if k != missing}

    with pytest.raises(ConfigError, match=missing):
        with_models(config_dir, {"alpha": model})


def test_local_fields_are_rejected_on_openai_models(config_dir):
    with pytest.raises(ConfigError, match="trust_remote_code"):
        with_models(config_dir, {"alpha": {**OPENAI, "trust_remote_code": True}})


def test_api_fields_are_rejected_on_local_models(config_dir):
    with pytest.raises(ConfigError, match="base_url"):
        with_models(config_dir, {"alpha": {"name": "org/a", "base_url": "https://x"}})


def test_unknown_provider_is_rejected(config_dir):
    with pytest.raises(ConfigError, match="weird"):
        with_models(config_dir, {"alpha": {**OPENAI, "provider": "weird"}})


def test_base_url_is_normalized_and_validated(config_dir):
    model = with_models(config_dir, {"alpha": {**OPENAI, "base_url": "https://api.example.com/v1/"}}).models["alpha"]
    assert model.base_url == "https://api.example.com/v1"

    with pytest.raises(ConfigError, match="base_url"):
        with_models(config_dir, {"alpha": {**OPENAI, "base_url": "api.example.com"}})


def test_chars_per_token_must_be_positive(config_dir):
    with pytest.raises(ConfigError, match="chars_per_token"):
        with_models(config_dir, {"alpha": {**OPENAI, "chars_per_token": 0}})


def test_model_path_only_exists_for_local_models(config_dir):
    config = with_models(config_dir, {"alpha": OPENAI})

    with pytest.raises(ValueError, match="not a local model"):
        config.model_path("alpha")


def test_default_model_can_be_an_api_model(config_dir):
    config = with_models(config_dir, {"api": OPENAI, "alpha": {"name": "org/a"}}, default="api")

    assert config.service.default_model == "api"


# --- api keys -----------------------------------------------------------


def test_api_key_comes_from_the_environment(monkeypatch):
    from app.core.config import get_api_key

    monkeypatch.setenv("SOME_KEY", "from-env")

    assert get_api_key("SOME_KEY") == "from-env"


def test_api_key_falls_back_to_the_dotenv_file(monkeypatch, tmp_path):
    from app.core.config import get_api_key

    monkeypatch.delenv("SOME_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('SOME_KEY="from-dotenv"\n')

    assert get_api_key("SOME_KEY") == "from-dotenv"


def test_missing_api_key_is_none(monkeypatch, tmp_path):
    from app.core.config import get_api_key

    monkeypatch.delenv("SOME_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    assert get_api_key("SOME_KEY") is None
