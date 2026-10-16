import logging

import pytest

from app.core.errors import EmbeddingServiceError, UnknownModelError
from app.services.embedding_service import EmbeddingService
from app.services.model_registry import ModelRegistry
from tests.fakes import FakeSentenceTransformer

MODELS = {
    "alpha": {"name": "org/alpha", "query_prefix": "query: ", "passage_prefix": "passage: ", "batch_size": 4,
              "chunk_overlap_ratio": 0.0},
    "beta": {"name": "org/beta", "normalize": False, "batch_size": 2, "chunk_overlap_ratio": 0.0},
}


@pytest.fixture
def setup(downloaded):
    def build(models=None, **service):
        registry = ModelRegistry(downloaded(models=models or MODELS, service=service), loader=FakeSentenceTransformer)
        return EmbeddingService(registry), registry

    return build


def encoded(registry, alias):
    return registry.get(alias).backend.model.encoded


# --- embed_texts --------------------------------------------------------


def test_passage_prefix_by_default_and_query_prefix_for_queries(setup):
    service, registry = setup()

    service.embed_texts(["hola"])
    service.embed_texts(["hola"], is_query=True)

    assert encoded(registry, "alpha") == [["passage: hola"], ["query: hola"]]


def test_model_without_prefixes_gets_none(setup):
    service, registry = setup()

    service.embed_texts(["hola"], model="beta")
    service.embed_texts(["hola"], model="beta", is_query=True)

    assert encoded(registry, "beta") == [["hola"], ["hola"]]


def test_each_model_uses_its_own_options(setup):
    service, registry = setup()

    service.embed_texts(["x"], model="alpha")
    service.embed_texts(["x"], model="beta")

    assert registry.get("alpha").backend.model.options == [{"normalize": True, "batch_size": 4}]
    assert registry.get("beta").backend.model.options == [{"normalize": False, "batch_size": 2}]


def test_embed_texts_accepts_single_string_and_empty_list(setup):
    service, registry = setup()

    assert service.embed_texts("single") == [[2.0]]
    assert service.embed_texts([]) == []
    assert len(encoded(registry, "alpha")) == 1  # the empty call never reached the model


def test_unknown_model_raises(setup):
    service, _ = setup()

    with pytest.raises(UnknownModelError):
        service.embed_texts(["x"], model="nope")


def test_encode_failure_is_logged_and_wrapped(setup, caplog):
    service, registry = setup()

    def boom(*_a, **_k):
        raise RuntimeError("out of memory")

    registry.get("alpha").backend.model.encode = boom

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError, match="could not be generated"):
            service.embed_texts(["x"])

    assert "out of memory" in caplog.text


# --- embed_documents ----------------------------------------------------


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_chunk_size_comes_from_the_model(setup):
    service, _ = setup()

    result = service.embed_documents([words(100)])

    assert len(result) > 1
    assert all(len(c["text"].split()) <= 18 for c in result)  # 21 - 2 special - 1 prefix


def test_model_without_prefix_gets_more_room_per_chunk(setup):
    service, _ = setup()

    result = service.embed_documents([words(100)], model="beta")

    assert max(len(c["text"].split()) for c in result) == 19  # 21 - 2 special, no prefix


def test_chunk_tokens_setting_overrides_the_model(setup):
    models = {"alpha": {**MODELS["alpha"], "chunk_tokens": 10}}
    service, _ = setup(models=models)

    result = service.embed_documents([words(100)])

    assert all(len(c["text"].split()) <= 10 - 2 - 1 for c in result)


def test_document_and_chunk_positions(setup):
    service, _ = setup()

    result = service.embed_documents([words(40), "corto"])

    assert [(c["document_index"], c["chunk_index"]) for c in result] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert result[-1]["text"] == "corto"
    assert all("embedding" in c for c in result)


def test_chunks_are_embedded_with_prefix_in_batches_of_the_model(setup):
    service, registry = setup()

    result = service.embed_documents([words(40)])

    batches = encoded(registry, "alpha")
    assert [len(b) for b in batches] == [3]  # 3 chunks, batch_size 4
    assert all(text.startswith("passage: ") for text in batches[0])
    assert len(result) == 3


def test_batches_follow_the_models_batch_size_and_report_progress(setup):
    service, registry = setup()
    progress = []

    service.embed_documents([words(40)], model="beta", on_progress=lambda d, t: progress.append((d, t)))

    assert [len(b) for b in encoded(registry, "beta")] == [2, 1]
    assert progress == [(2, 3), (3, 3)]


def test_blank_documents_are_skipped(setup):
    service, _ = setup()

    result = service.embed_documents(["   ", "hola"])

    assert [(c["document_index"], c["text"]) for c in result] == [(1, "hola")]


def test_prefix_longer_than_chunk_size_is_an_error(setup):
    models = {"alpha": {**MODELS["alpha"], "chunk_tokens": 3}}  # 3 - 2 special - 1 prefix = 0
    service, _ = setup(models=models)

    with pytest.raises(EmbeddingServiceError, match="smaller than"):
        service.embed_documents(["hola"])


# --- API models through the same service ---------------------------------


def test_documents_are_chunked_and_embedded_through_an_api_model(downloaded):
    """The service works unchanged with a remote model: only config differs."""
    import json

    import httpx

    from app.providers.openai_compatible import OpenAICompatibleBackend

    requests = []

    def server(request):
        body = json.loads(request.content)
        requests.append(body)
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})

    def factory(alias, cfg, api_key):
        client = httpx.Client(transport=httpx.MockTransport(server))
        return OpenAICompatibleBackend(alias, cfg, api_key, client=client)

    config = downloaded(
        models={
            "remote": {
                "provider": "openai",
                "name": "remote-embed",
                "base_url": "https://api.example.com/v1",
                "api_key_env": "KEY",
                "passage_prefix": "doc: ",
                "chunk_tokens": 20,
                "chars_per_token": 1,
                "chunk_overlap_ratio": 0,
                "batch_size": 2,
                "normalize": False,
            }
        }
    )
    service = EmbeddingService(ModelRegistry(config, key_lookup=lambda _: "secret", remote_factory=factory))
    text = " ".join(f"w{i:02d}" for i in range(30))  # 30 words of 3 chars, 119 chars in total

    result = service.embed_documents([text])

    assert len(result) > 1
    assert all(len(c["text"]) <= 20 - len("doc: ") for c in result)  # estimate: 1 char = 1 token, minus the prefix
    sent = [t for body in requests for t in body["input"]]
    assert all(t.startswith("doc: ") for t in sent)
    assert [len(b["input"]) for b in requests][0] == 2  # batch_size
    assert " ".join(c["text"] for c in result) == text  # nothing lost
