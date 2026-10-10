import pytest
from pydantic import SecretStr

from scripts import download_model


class FakeApi:
    def __init__(self, files):
        self.files = files

    def list_repo_files(self, *_args, **_kwargs):
        return self.files


@pytest.fixture
def run(monkeypatch, downloaded):
    """Run the script against a two-model config; returns (call_main, snapshot_calls, config)."""
    config = downloaded(
        models={"alpha": {"name": "org/alpha"}, "beta": {"name": "org/beta"}},
        service={"default_model": "alpha"},
    )
    for alias in config.models:  # start with nothing downloaded
        config.model_path(alias).rmdir()

    calls = []
    repo_files = {"files": ["model.safetensors"]}
    env = type("Env", (), {"hf_token": SecretStr("hf_secret")})()
    monkeypatch.setattr(download_model, "get_config", lambda: config)
    monkeypatch.setattr(download_model, "get_env", lambda: env)
    monkeypatch.setattr(download_model, "snapshot_download", lambda **kw: calls.append(kw))
    monkeypatch.setattr(download_model, "HfApi", lambda: FakeApi(repo_files["files"]))

    def call(*argv, files=None):
        if files is not None:
            repo_files["files"] = files
        download_model.main(list(argv))

    return call, calls, config


def downloaded_repos(calls):
    return [c["repo_id"] for c in calls]


def mark_present(config, alias):
    path = config.model_path(alias)
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")


def test_downloads_only_the_default_model_without_arguments(run):
    call, calls, config = run

    call()

    assert downloaded_repos(calls) == ["org/alpha"]
    assert calls[0]["local_dir"] == config.model_path("alpha")


def test_downloads_the_requested_alias(run):
    call, calls, _ = run

    call("--model", "beta")

    assert downloaded_repos(calls) == ["org/beta"]


def test_all_downloads_every_configured_model(run):
    call, calls, _ = run

    call("--all")

    assert downloaded_repos(calls) == ["org/alpha", "org/beta"]


def test_unknown_alias_exits_listing_available(run):
    call, calls, _ = run

    with pytest.raises(SystemExit, match="nope.*alpha, beta"):
        call("--model", "nope")

    assert calls == []


def test_skips_models_that_are_already_present(run):
    call, calls, config = run
    mark_present(config, "alpha")

    call("--all")

    assert downloaded_repos(calls) == ["org/beta"]


def test_force_downloads_again(run):
    call, calls, config = run
    mark_present(config, "alpha")

    call("--force")

    assert downloaded_repos(calls) == ["org/alpha"]


def test_token_is_passed_for_download(run):
    call, calls, _ = run

    call()

    assert calls[0]["token"] == "hf_secret"


def test_duplicate_pytorch_bin_is_skipped_when_safetensors_exist(run):
    call, calls, _ = run

    call(files=["model.safetensors", "pytorch_model.bin"])

    assert "pytorch_model*.bin" in calls[0]["ignore_patterns"]


def test_bin_weights_are_kept_when_repo_has_no_safetensors(run):
    call, calls, _ = run

    call(files=["pytorch_model.bin"])

    assert "pytorch_model*.bin" not in calls[0]["ignore_patterns"]
