"""Strategy routing for the isolated Chapter 4 retrieval pipeline."""

from types import SimpleNamespace

import pytest

from app.core import retrieval


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: SimpleNamespace(
        rerank_top_k=3, recall_top_k=5,
    ))
    monkeypatch.setattr(
        retrieval.milvus_client, "ensure_collection",
        lambda client, collection: None,
    )


def test_arrange_head_tail_keeps_best_at_front_and_second_at_end():
    assert retrieval.arrange_head_tail([]) == []
    assert retrieval.arrange_head_tail(["a", "b"]) == ["a", "b"]
    assert retrieval.arrange_head_tail(["a", "b", "c", "d"]) == ["a", "c", "d", "b"]


@pytest.mark.asyncio
async def test_vector_strategy_uses_dense_search(monkeypatch):
    async def embed(query):
        assert query == "运费"
        return [0.1] * 1024

    calls = []
    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    monkeypatch.setattr(retrieval.milvus_client, "dense_search", lambda *a, **kw: calls.append((a, kw)) or [{"id": 1}])
    hits = await retrieval.search_knowledge(
        "运费", strategy="vector", client=object(), collection="isolated",
        category="运费", top_k=2,
    )
    assert hits == [{"id": 1}]
    assert calls[0][1] == {"top_k": 2, "category": "运费", "collection": "isolated"}


@pytest.mark.asyncio
async def test_bm25_strategy_skips_embedding(monkeypatch):
    async def fail_embed(query):
        raise AssertionError("BM25 should not embed a query")

    monkeypatch.setattr(retrieval.embeddings, "embed_query", fail_embed)
    monkeypatch.setattr(retrieval.milvus_client, "bm25_search", lambda *a, **kw: [{"id": 2}])
    assert await retrieval.search_knowledge(
        "Pro", strategy="bm25", client=object(), collection="isolated",
    ) == [{"id": 2}]


@pytest.mark.asyncio
async def test_hybrid_strategy_uses_recall_budget_then_final_limit(monkeypatch):
    async def embed(query):
        return [0.1] * 1024

    calls = []
    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    monkeypatch.setattr(retrieval.milvus_client, "hybrid_search", lambda *a, **kw: calls.append((a, kw)) or [{"id": 1}, {"id": 2}])
    hits = await retrieval.search_knowledge(
        "运费", strategy="hybrid", client=object(), collection="isolated", top_k=1,
    )
    assert hits == [{"id": 1}]
    assert calls[0][1]["top_k"] == calls[0][1]["recall"] == 5


@pytest.mark.asyncio
async def test_hybrid_expansion_only_reaches_bm25(monkeypatch):
    async def embed(query):
        assert query == "邮费多少"
        return [0.1] * 1024

    async def rerank(query, docs, top_n):
        assert query == "邮费多少"
        return [(0, 0.9)]

    def hybrid(client, vector, text, **kwargs):
        assert text == "邮费多少 运费"
        return [{"id": 1, "question": "运费", "answer": "满99包邮"}]

    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    monkeypatch.setattr(retrieval.milvus_client, "hybrid_search", hybrid)
    monkeypatch.setattr(retrieval.rerank, "rerank", rerank)
    hits = await retrieval.search_knowledge(
        "邮费多少", bm25_query="邮费多少 运费", strategy="hybrid_rerank", client=object(),
    )
    assert hits[0]["id"] == 1


@pytest.mark.asyncio
async def test_hybrid_rerank_preserves_original_hit_and_attaches_score(monkeypatch):
    async def embed(query):
        return [0.1] * 1024

    async def rerank(query, docs, top_n):
        assert docs == ["运费 满99包邮", "Pro 自动清理"]
        assert top_n == 2
        return [(1, 0.95), (0, 0.2)]

    monkeypatch.setattr(retrieval.embeddings, "embed_query", embed)
    monkeypatch.setattr(retrieval.milvus_client, "hybrid_search", lambda *a, **kw: [
        {"id": 1, "score": 0.5, "question": "运费", "answer": "满99包邮"},
        {"id": 2, "score": 0.4, "question": "Pro", "answer": "自动清理"},
    ])
    monkeypatch.setattr(retrieval.rerank, "rerank", rerank)
    hits = await retrieval.search_knowledge(
        "Pro", strategy="hybrid_rerank", client=object(), collection="isolated", top_k=2,
    )
    assert [(hit["id"], hit["rerank_score"]) for hit in hits] == [(2, 0.95), (1, 0.2)]
    assert hits[0]["score"] == 0.4


@pytest.mark.asyncio
async def test_invalid_strategy_fails_before_query(monkeypatch):
    with pytest.raises(ValueError, match="strategy"):
        await retrieval.search_knowledge("Pro", strategy="unknown", client=object())
