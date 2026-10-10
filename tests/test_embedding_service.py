import logging

import pytest

from app.services import embedding_service
from app.services.embedding_service import EmbeddingServiceError


class FakeVectors:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.seen = None
        self.received_normalize = None
        self.received_batch_size = None

    def encode(self, texts, normalize_embeddings, batch_size):
        self.seen = texts
        self.received_normalize = normalize_embeddings
        self.received_batch_size = batch_size
        return FakeVectors([[1.0, 2.0]] * len(texts))


@pytest.fixture(autouse=True)
def reset_model(monkeypatch):
    monkeypatch.setattr(embedding_service, "model", None)


def test_get_model_logs_hint_when_model_not_downloaded(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(embedding_service.settings, "model_dir", str(tmp_path))
    monkeypatch.setattr(embedding_service.settings, "model_name", "org/not-downloaded")

    with caplog.at_level(logging.ERROR, logger=embedding_service.logger.name):
        with pytest.raises(EmbeddingServiceError):
            embedding_service.get_model()

    assert "org/not-downloaded" in caplog.text
    assert "download_model" in caplog.text


def test_get_model_logs_loader_exception(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(embedding_service.settings, "model_dir", str(tmp_path))
    embedding_service.model_path().mkdir(parents=True)

    def boom(*_args, **_kwargs):
        raise OSError("corrupted weights")

    monkeypatch.setattr(embedding_service, "SentenceTransformer", boom)

    with caplog.at_level(logging.ERROR, logger=embedding_service.logger.name):
        with pytest.raises(EmbeddingServiceError):
            embedding_service.get_model()

    assert "corrupted weights" in caplog.text


def test_embed_texts_uses_passage_prefix_by_default(monkeypatch):
    monkeypatch.setattr(embedding_service.settings, "passage_prefix", "passage: ")
    fake = FakeModel()
    monkeypatch.setattr(embedding_service, "get_model", lambda: fake)

    embedding_service.embed_texts(["hola"])

    assert fake.seen == ["passage: hola"]


def test_embed_texts_uses_query_prefix_for_queries(monkeypatch):
    monkeypatch.setattr(embedding_service.settings, "query_prefix", "query: ")
    fake = FakeModel()
    monkeypatch.setattr(embedding_service, "get_model", lambda: fake)

    embedding_service.embed_texts(["hola"], is_query=True)

    assert fake.seen == ["query: hola"]


def test_embed_texts_normalizes_and_batches(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embedding_service, "get_model", lambda: fake)
    monkeypatch.setattr(embedding_service.settings, "passage_prefix", "passage: ")
    monkeypatch.setattr(embedding_service.settings, "normalize", True)
    monkeypatch.setattr(embedding_service.settings, "batch_size", 8)

    result = embedding_service.embed_texts(["abcdef"])

    assert result == [[1.0, 2.0]]
    assert fake.seen == ["passage: abcdef"]
    assert fake.received_normalize is True
    assert fake.received_batch_size == 8


def test_embed_texts_accepts_single_string(monkeypatch):
    monkeypatch.setattr(embedding_service.settings, "passage_prefix", "passage: ")
    fake = FakeModel()
    monkeypatch.setattr(embedding_service, "get_model", lambda: fake)

    result = embedding_service.embed_texts("single chunk")

    assert result == [[1.0, 2.0]]
    assert fake.seen == ["passage: single chunk"]


def test_embed_texts_returns_empty_list_without_loading_model(monkeypatch):
    def fail_get_model():
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(embedding_service, "get_model", fail_get_model)

    assert embedding_service.embed_texts([]) == []


def test_get_model_loads_from_local_dir_offline(monkeypatch, tmp_path):
    received = {}

    def fake_sentence_transformer(path, trust_remote_code, local_files_only):
        received.update(
            path=path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return object()

    monkeypatch.setattr(embedding_service, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(embedding_service.settings, "model_dir", str(tmp_path))
    monkeypatch.setattr(embedding_service.settings, "model_name", "org/test-model")
    monkeypatch.setattr(embedding_service.settings, "trust_remote_code", True)
    (tmp_path / "org--test-model").mkdir()

    loaded = embedding_service.get_model()

    assert loaded is embedding_service.model
    assert received == {
        "path": str(tmp_path / "org--test-model"),
        "trust_remote_code": True,
        "local_files_only": True,
    }


def test_prefixes_are_not_hardcoded_in_settings_defaults(monkeypatch):
    from app.core.settings import Settings

    monkeypatch.delenv("QUERY_PREFIX", raising=False)
    monkeypatch.delenv("PASSAGE_PREFIX", raising=False)
    defaults = Settings(_env_file=None)

    assert defaults.query_prefix == ""
    assert defaults.passage_prefix == ""


class FakeTokenizer:
    """One token per word; mimics the HF tokenizer call used by the service."""

    def __call__(self, units, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [u.split() for u in units]}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeDocModel(FakeModel):
    max_seq_length = 21  # 21 - 2 special tokens - 1 prefix token = 18 per chunk
    tokenizer = FakeTokenizer()

    def encode(self, texts, normalize_embeddings, batch_size):
        self.batches = getattr(self, "batches", [])
        self.batches.append(len(texts))
        return FakeVectors([[float(len(t.split()))] for t in texts])


def configure_docs(monkeypatch, **overrides):
    fake = FakeDocModel()
    monkeypatch.setattr(embedding_service, "get_model", lambda: fake)
    monkeypatch.setattr(embedding_service.settings, "passage_prefix", "passage: ")
    monkeypatch.setattr(embedding_service.settings, "chunk_tokens", None)
    monkeypatch.setattr(embedding_service.settings, "chunk_overlap_ratio", 0.0)
    monkeypatch.setattr(embedding_service.settings, "batch_size", 4)
    for name, value in overrides.items():
        monkeypatch.setattr(embedding_service.settings, name, value)
    return fake


def test_embed_documents_chunk_size_comes_from_model(monkeypatch):
    configure_docs(monkeypatch)
    text = " ".join(f"w{i}" for i in range(100))

    result = embedding_service.embed_documents([text])

    assert len(result) > 1
    assert all(len(c["text"].split()) <= 18 for c in result)


def test_embed_documents_chunk_tokens_setting_overrides_model(monkeypatch):
    configure_docs(monkeypatch, chunk_tokens=10)
    text = " ".join(f"w{i}" for i in range(100))

    result = embedding_service.embed_documents([text])

    assert all(len(c["text"].split()) <= 10 for c in result)


def test_embed_documents_keeps_document_and_chunk_positions(monkeypatch):
    configure_docs(monkeypatch)
    long_doc = " ".join(f"w{i}" for i in range(40))

    result = embedding_service.embed_documents([long_doc, "corto"])

    assert [(c["document_index"], c["chunk_index"]) for c in result] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
    ]
    assert result[-1]["text"] == "corto"
    assert all("embedding" in c for c in result)


def test_embed_documents_embeds_with_prefix_and_in_batches(monkeypatch):
    fake = configure_docs(monkeypatch, batch_size=2)
    text = " ".join(f"w{i}" for i in range(40))

    result = embedding_service.embed_documents([text])

    assert len(result) == 3
    assert fake.batches == [2, 1]
    # the prefix is part of what is embedded (the fake counts 1 word for it)
    assert result[-1]["embedding"] == [float(len(result[-1]["text"].split()) + 1)]


def test_embed_documents_reports_progress(monkeypatch):
    configure_docs(monkeypatch, batch_size=2)
    text = " ".join(f"w{i}" for i in range(40))
    progress = []

    embedding_service.embed_documents([text], on_progress=lambda done, total: progress.append((done, total)))

    assert progress == [(2, 3), (3, 3)]


def test_embed_documents_skips_blank_documents(monkeypatch):
    configure_docs(monkeypatch)

    result = embedding_service.embed_documents(["   ", "hola"])

    assert [(c["document_index"], c["text"]) for c in result] == [(1, "hola")]
