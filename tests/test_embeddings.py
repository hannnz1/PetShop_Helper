"""Embedding client contract tests; never call the paid provider."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.core import embeddings


def _settings(key: str | None = "test-key", base_url: str = "https://api.siliconflow.cn/v1") -> Settings:
    return Settings(
        chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test",
        siliconflow_api_key=SecretStr(key) if key is not None else None,
        embed_base_url=base_url,
        _env_file=None,
    )


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.calls = []
        self.embeddings = self
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.data)


@pytest.mark.asyncio
async def test_embed_texts_preserves_input_order_and_closes_client(monkeypatch):
    client = FakeClient([
        SimpleNamespace(index=1, embedding=[0.2] * 1024),
        SimpleNamespace(index=0, embedding=[0.1] * 1024),
    ])
    monkeypatch.setattr(embeddings, "get_settings", lambda: _settings())
    monkeypatch.setattr(embeddings, "_client", lambda: client)
    assert await embeddings.embed_texts(["a", "b"]) == [[0.1] * 1024, [0.2] * 1024]
    assert client.calls == [{"model": "BAAI/bge-m3", "input": ["a", "b"]}]
    assert client.closed


@pytest.mark.asyncio
async def test_embed_query_returns_one_vector(monkeypatch):
    client = FakeClient([SimpleNamespace(index=0, embedding=[0.3] * 1024)])
    monkeypatch.setattr(embeddings, "get_settings", lambda: _settings())
    monkeypatch.setattr(embeddings, "_client", lambda: client)
    assert await embeddings.embed_query("邮费") == [0.3] * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [
    [SimpleNamespace(index=0, embedding=[0.1] * 1024)],
    [SimpleNamespace(index=0, embedding=[0.1] * 8), SimpleNamespace(index=1, embedding=[0.2] * 1024)],
    [SimpleNamespace(index=0, embedding=[0.1] * 1024), SimpleNamespace(index=0, embedding=[0.2] * 1024)],
])
async def test_embed_texts_rejects_missing_bad_dim_or_duplicate_index(monkeypatch, data):
    client = FakeClient(data)
    monkeypatch.setattr(embeddings, "get_settings", lambda: _settings())
    monkeypatch.setattr(embeddings, "_client", lambda: client)
    with pytest.raises(ValueError, match="embedding response"):
        await embeddings.embed_texts(["a", "b"])
    assert client.closed


@pytest.mark.asyncio
async def test_embed_texts_empty_input_does_not_call_provider(monkeypatch):
    monkeypatch.setattr(embeddings, "_client", lambda: (_ for _ in ()).throw(AssertionError("called")))
    assert await embeddings.embed_texts([]) == []


def test_client_requires_siliconflow_key(monkeypatch):
    monkeypatch.setattr(embeddings, "get_settings", lambda: _settings(None))
    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        embeddings._client()


@pytest.mark.parametrize("base_url", [
    "http://api.siliconflow.cn/v1",
    "https://example.test/v1",
    "https://api.siliconflow.cn:444/v1",
    "https://api.siliconflow.cn/v1?token=oops",
])
def test_client_rejects_untrusted_endpoint_before_using_secret(monkeypatch, base_url):
    monkeypatch.setattr(embeddings, "get_settings", lambda: _settings(base_url=base_url))
    with pytest.raises(ValueError, match="EMBED_BASE_URL"):
        embeddings._client()
