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
