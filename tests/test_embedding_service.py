from app.services import embedding_service


class FakeVectors:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.received_texts = None
        self.received_normalize = None

    def encode(self, texts, normalize_embeddings):
        self.received_texts = texts
        self.received_normalize = normalize_embeddings
        return FakeVectors([[1.0, 2.0]])


def test_embed_texts_truncates_and_normalizes(monkeypatch):
    fake_model = FakeModel()

    monkeypatch.setattr(embedding_service, "get_model", lambda: fake_model)
    monkeypatch.setattr(embedding_service.settings, "max_chars", 4)
    monkeypatch.setattr(embedding_service.settings, "normalize", True)

    result = embedding_service.embed_texts(["abcdef"])

    assert result == [[1.0, 2.0]]
    assert fake_model.received_texts == ["abcd"]
    assert fake_model.received_normalize is True


def test_embed_texts_accepts_single_string(monkeypatch):
    fake_model = FakeModel()

    monkeypatch.setattr(embedding_service, "get_model", lambda: fake_model)
    monkeypatch.setattr(embedding_service.settings, "max_chars", 100)

    result = embedding_service.embed_texts("single chunk")

    assert result == [[1.0, 2.0]]
    assert fake_model.received_texts == ["single chunk"]


def test_embed_texts_returns_empty_list_without_loading_model(monkeypatch):
    def fail_get_model():
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(embedding_service, "get_model", fail_get_model)

    assert embedding_service.embed_texts([]) == []


def test_get_model_passes_hf_token(monkeypatch):
    received = {}

    def fake_sentence_transformer(model_name, trust_remote_code, token):
        received["model_name"] = model_name
        received["trust_remote_code"] = trust_remote_code
        received["token"] = token
        return object()

    monkeypatch.setattr(embedding_service, "model", None)
    monkeypatch.setattr(embedding_service, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(embedding_service.settings, "model_name", "test-model")
    monkeypatch.setattr(embedding_service.settings, "trust_remote_code", True)
    monkeypatch.setattr(embedding_service.settings, "hf_token", "hf_test")

    loaded_model = embedding_service.get_model()

    assert loaded_model is embedding_service.model
    assert received == {
        "model_name": "test-model",
        "trust_remote_code": True,
        "token": "hf_test",
    }
