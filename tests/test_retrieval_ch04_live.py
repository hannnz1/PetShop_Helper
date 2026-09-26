"""End-to-end strategy routing over a disposable Standalone collection."""

import uuid

import pytest

from app.core import retrieval
from app.kb import milvus_client as mc


def _vector(axis: int) -> list[float]:
    values = [0.0] * mc.DIM
    values[axis] = 1.0
    return values


@pytest.mark.asyncio
async def test_four_strategies_over_real_isolated_collection(monkeypatch):
    client = mc.get_client(uri="http://127.0.0.1:19530")
    name = f"test_ch04_route_{uuid.uuid4().hex[:12]}"
    try:
        mc.ensure_collection(client, collection=name)
        mc.upsert_vectors(client, [
            {
                "id": 1, "dense": _vector(0), "text": "运费满99元包邮",
                "question": "运费怎么算", "answer": "满99元包邮",
                "section_path": "运费", "content_type": "faq", "category": "运费",
            },
            {
                "id": 2, "dense": _vector(1), "text": "Pro 猫砂盆自动清理",
                "question": "Pro 功能", "answer": "自动清理",
                "section_path": "手册", "content_type": "manual", "category": "商品",
            },
        ], collection=name)
        mc.flush(client, collection=name)

        async def fake_embed(query):
            return _vector(0)

        async def fake_rerank(query, docs, top_n):
            return [(0, 0.91)]

        monkeypatch.setattr(retrieval.embeddings, "embed_query", fake_embed)
        monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
        for strategy in ("vector", "bm25", "hybrid", "hybrid_rerank"):
            hits = await retrieval.search_knowledge(
                "运费 包邮", strategy=strategy, top_k=1, category="运费",
                client=client, collection=name,
            )
            assert hits and hits[0]["id"] == 1
            assert hits[0]["category"] == "运费"
            if strategy == "hybrid_rerank":
                assert hits[0]["rerank_score"] == 0.91
    finally:
        mc.drop(client, name)
        client.close()
