import json
import logging
import math

import httpx
import pytest

from app.core.config import OpenAIModelConfig
from app.core.errors import EmbeddingServiceError
from app.providers.openai_compatible import OpenAICompatibleBackend

KEY = "sk-test-super-secret"


def make_cfg(**overrides):
    values = {
        "provider": "openai",
        "name": "embed-model",
        "base_url": "https://api.example.com/v1",
        "api_key_env": "EXAMPLE_KEY",
        "chunk_tokens": 100,
        "normalize": False,
        "batch_size": 2,
        "max_retries": 2,
    }
    values.update(overrides)
    return OpenAIModelConfig.model_validate(values)


def embeddings_response(inputs, shuffle=False):
    """A valid OpenAI response where vector i is [len(text), i]."""
    data = [{"object": "embedding", "index": i, "embedding": [float(len(t)), float(i)]} for i, t in enumerate(inputs)]
    return {"object": "list", "data": list(reversed(data)) if shuffle else data}


class Server:
    """Scripted fake of the API: pops one behaviour per request, then answers 200."""

    def __init__(self, *script, shuffle=False):
        self.script = list(script)
        self.requests = []
        self.shuffle = shuffle

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        body = json.loads(request.content)
        return httpx.Response(200, json=embeddings_response(body["input"], self.shuffle))

    @property
    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def build():
    sleeps = []

    def make(server, **overrides):
        client = httpx.Client(transport=httpx.MockTransport(server))
        backend = OpenAICompatibleBackend("m", make_cfg(**overrides), KEY, client=client, sleep=sleeps.append)
        return backend, sleeps

    return make


# --- request ------------------------------------------------------------


def test_posts_to_embeddings_with_model_input_and_bearer_key(build):
    server = Server()
    backend, _ = build(server)

    backend.encode(["uno"])

    request = server.requests[0]
    assert str(request.url) == "https://api.example.com/v1/embeddings"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert server.bodies[0] == {"model": "embed-model", "input": ["uno"], "encoding_format": "float"}


def test_dimensions_are_sent_only_when_configured(build):
    server = Server()
    backend, _ = build(server, dimensions=256)

    backend.encode(["uno"])

    assert server.bodies[0]["dimensions"] == 256


def test_inputs_are_split_in_batches_and_order_is_kept(build):
    server = Server()
    backend, _ = build(server)

    vectors = backend.encode(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [b["input"] for b in server.bodies] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_response_items_are_reordered_by_index(build):
    backend, _ = build(Server(shuffle=True))

    vectors = backend.encode(["a", "bb"])

    assert vectors == [[1.0, 0.0], [2.0, 1.0]]


def test_vectors_are_normalized_when_configured(build):
    backend, _ = build(Server(), normalize=True)

    [vector] = backend.encode(["abc"])  # raw [3, 0]

    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)


def test_empty_input_makes_no_request(build):
    server = Server()
    backend, _ = build(server)

    assert backend.encode([]) == []
    assert server.requests == []


# --- retries and errors -------------------------------------------------


def test_rate_limit_is_retried_honoring_retry_after(build):
    server = Server(httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow down"}))
    backend, sleeps = build(server)

    vectors = backend.encode(["a"])

    assert len(server.requests) == 2
    assert sleeps == [7.0]
    assert vectors == [[1.0, 0.0]]


def test_server_errors_back_off_exponentially_then_give_up(build):
    server = Server(*[httpx.Response(503) for _ in range(5)])
    backend, sleeps = build(server)

    with pytest.raises(EmbeddingServiceError, match="unavailable"):
        backend.encode(["a"])

    assert len(server.requests) == 3  # 1 try + max_retries (2)
    assert sleeps == [1, 2]


def test_timeouts_are_retried(build):
    server = Server(httpx.ReadTimeout("slow"))
    backend, sleeps = build(server)

    assert backend.encode(["a"]) == [[1.0, 0.0]]
    assert len(sleeps) == 1


def test_connection_errors_give_up_after_max_retries(build):
    server = Server(*[httpx.ConnectError("down") for _ in range(5)])
    backend, _ = build(server)

    with pytest.raises(EmbeddingServiceError, match="unavailable"):
        backend.encode(["a"])

    assert len(server.requests) == 3


def test_retry_after_is_capped(build):
    server = Server(httpx.Response(429, headers={"Retry-After": "3600"}))
    backend, sleeps = build(server)

    backend.encode(["a"])

    assert sleeps == [60.0]


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_errors_are_not_retried_and_never_leak_the_key(build, caplog, status):
    server = Server(httpx.Response(status, json={"error": {"message": "bad key"}}))
    backend, sleeps = build(server)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbeddingServiceError, match="authentication") as error:
            backend.encode(["a"])

    assert len(server.requests) == 1 and sleeps == []
    assert KEY not in str(error.value)
    assert KEY not in caplog.text
    assert "EXAMPLE_KEY" in caplog.text  # tells the operator which variable to check


def test_client_errors_are_not_retried_and_report_the_status(build, caplog):
    server = Server(httpx.Response(400, json={"error": {"message": "input too long"}}))
    backend, sleeps = build(server)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError, match=r"rejected \(HTTP 400\)"):
            backend.encode(["a"])

    assert len(server.requests) == 1 and sleeps == []
    assert "input too long" in caplog.text  # the real reason goes to the log, not to the client


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},
        {"data": [{"index": 0}]},
        {"data": []},  # fewer vectors than inputs
    ],
)
def test_malformed_responses_are_errors(build, body):
    backend, _ = build(Server(httpx.Response(200, json=body)))

    with pytest.raises(EmbeddingServiceError, match="unexpected response"):
        backend.encode(["a"])


def test_non_json_response_is_an_error(build):
    backend, _ = build(Server(httpx.Response(200, text="<html>gateway</html>")))

    with pytest.raises(EmbeddingServiceError, match="unexpected response"):
        backend.encode(["a"])


# --- sizing -------------------------------------------------------------


def test_token_count_is_estimated_from_characters(build):
    backend, _ = build(Server(), chars_per_token=4)

    # one extra character per unit: the space that will join it to the next one
    assert backend.count_tokens(["", "   ", "a", "abc", "abcde", "x" * 40]) == [0, 0, 1, 1, 2, 11]


def test_limits_come_from_the_config(build):
    backend, _ = build(Server(), chunk_tokens=300, dimensions=64)

    assert backend.input_limit == 300
    assert backend.special_tokens == 0
    assert backend.dimensions == 64


def test_dimensions_unknown_when_not_configured(build):
    backend, _ = build(Server())

    assert backend.dimensions is None
