"""Dense BGE-M3 search over a real temporary Milvus Lite collection."""

import pytest
from types import SimpleNamespace

from app.core import retrieval
from app.kb import milvus_client


def _vector(first: float, second: float) -> list[float]:
    return [first, second] + [0.0] * (milvus_client.DIM - 2)


@pytest.fixture()
def milvus(tmp_path):
    client = milvus_client.get_client(uri=str(tmp_path / "retrieval.db"))
    milvus_client.ensure_collection(client)
    milvus_client.upsert_vectors(client, [
        {"id": 1, "vector": _vector(1, 0), "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": _vector(0, 1), "question": "发货时效", "answer": "48小时"},
    ])
    yield client
    client.close()


@pytest.mark.asyncio
async def test_search_returns_top_hit_above_threshold(monkeypatch, milvus):
    async def embed(_query):
        return _vector(1, 0)

    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    hits = await retrieval.search_knowledge("邮费是多少", top_k=1, min_score=0.5, client=milvus)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["answer"] == "满99包邮"


@pytest.mark.asyncio
async def test_below_threshold_is_filtered(monkeypatch, milvus):
    async def embed(_query):
        return _vector(1, 0)

    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    assert await retrieval.search_knowledge("邮费", top_k=2, min_score=1.1, client=milvus) == []


@pytest.mark.asyncio
async def test_empty_collection_returns_empty(monkeypatch, tmp_path):
    client = milvus_client.get_client(uri=str(tmp_path / "empty.db"))
    try:
        async def embed(_query):
            return _vector(1, 0)

        monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
        assert await retrieval.search_knowledge("邮费", client=client) == []
    finally:
        client.close()


@pytest.mark.asyncio
async def test_defaults_and_runtime_client_remains_open_on_success(monkeypatch):
    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()
    monkeypatch.setattr(retrieval, "get_settings", lambda: SimpleNamespace(
        retrieval_top_k=2, retrieval_min_score=0.5,
    ))

    async def embed(_query):
        return _vector(1, 0)

    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    monkeypatch.setattr(retrieval.milvus_client, "get_runtime_client", lambda: client)
    monkeypatch.setattr(retrieval.milvus_client, "ensure_collection", lambda _client: None)
    calls = []

    def search(_client, _vector, limit):
        calls.append(limit)
        return [{"score": 0.5, "id": 1}, {"score": 0.49, "id": 2}]

    monkeypatch.setattr(retrieval.milvus_client, "search", search)
    assert await retrieval.search_knowledge("邮费") == [{"score": 0.5, "id": 1}]
    assert calls == [2]
    assert not client.closed


@pytest.mark.asyncio
async def test_runtime_client_remains_open_when_search_fails(monkeypatch):
    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    async def embed(_query):
        return _vector(1, 0)

    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    monkeypatch.setattr(retrieval.milvus_client, "get_runtime_client", lambda: client)
    monkeypatch.setattr(retrieval.milvus_client, "ensure_collection", lambda _client: None)

    def fail(*_args):
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(retrieval.milvus_client, "search", fail)
    with pytest.raises(RuntimeError, match="search unavailable"):
        await retrieval.search_knowledge("邮费")
    assert not client.closed
