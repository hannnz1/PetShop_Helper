"""Offline tests for the live embedding gate; no paid API calls."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.config import Settings
from scripts.smoke_embed import run_smoke


def settings(key: str | None = "test-key") -> Settings:
    return Settings(
        chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test",
        siliconflow_api_key=SecretStr(key) if key is not None else None,
        _env_file=None,
    )


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.called = False
        self.embeddings = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def create(self, **kwargs):
        self.called = True
        assert kwargs == {"model": "BAAI/bge-m3", "input": ["邮费是多少", "运费怎么算"]}
        return self.response


@pytest.mark.asyncio
async def test_missing_key_is_no_go_without_creating_client():
    def forbidden(**kwargs):
        raise AssertionError("must not make a network client")

    assert not await run_smoke(settings(None), client_factory=forbidden)


@pytest.mark.asyncio
async def test_two_1024_dimensional_vectors_are_go():
    client = FakeClient(SimpleNamespace(data=[
        SimpleNamespace(index=0, embedding=[0.1] * 1024),
        SimpleNamespace(index=1, embedding=[0.2] * 1024),
    ]))
    assert await run_smoke(settings(), client_factory=lambda **kwargs: client)
    assert client.called


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [
    [SimpleNamespace(index=0, embedding=[0.1] * 1024)],
    [SimpleNamespace(index=0, embedding=[0.1] * 1024), SimpleNamespace(index=1, embedding=[0.2] * 8)],
])
async def test_missing_or_wrong_dimension_vector_is_no_go(vectors):
    client = FakeClient(SimpleNamespace(data=vectors))
    assert not await run_smoke(settings(), client_factory=lambda **kwargs: client)
