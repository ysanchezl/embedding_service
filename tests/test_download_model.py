from scripts import download_model


def point_to(monkeypatch, tmp_path):
    monkeypatch.setattr(download_model.settings, "model_dir", str(tmp_path))
    monkeypatch.setattr(download_model.settings, "model_name", "org/model")
    calls = []
    monkeypatch.setattr(download_model, "snapshot_download", lambda **kw: calls.append(kw))
    monkeypatch.setattr(
        download_model,
        "HfApi",
        lambda: type("Api", (), {"list_repo_files": lambda self, *a, **k: ["model.safetensors"]})(),
    )
    return calls


def test_downloads_when_model_is_missing(monkeypatch, tmp_path):
    calls = point_to(monkeypatch, tmp_path)

    download_model.main([])

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "org/model"
    assert "pytorch_model*.bin" in calls[0]["ignore_patterns"]


def test_skips_download_when_model_already_present(monkeypatch, tmp_path):
    calls = point_to(monkeypatch, tmp_path)
    target = tmp_path / "org--model"
    target.mkdir()
    (target / "config.json").write_text("{}")

    download_model.main([])

    assert calls == []


def test_force_downloads_again(monkeypatch, tmp_path):
    calls = point_to(monkeypatch, tmp_path)
    target = tmp_path / "org--model"
    target.mkdir()
    (target / "config.json").write_text("{}")

    download_model.main(["--force"])

    assert len(calls) == 1


def test_keeps_bin_weights_when_repo_has_no_safetensors(monkeypatch, tmp_path):
    calls = point_to(monkeypatch, tmp_path)
    monkeypatch.setattr(
        download_model,
        "HfApi",
        lambda: type("Api", (), {"list_repo_files": lambda self, *a, **k: ["pytorch_model.bin"]})(),
    )

    download_model.main([])

    assert "pytorch_model*.bin" not in calls[0]["ignore_patterns"]
