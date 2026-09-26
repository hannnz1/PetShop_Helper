"""Legacy Lite retrieval and opt-in Chapter 4 hybrid strategies."""

from app.config import get_settings
from app.core import embeddings, rerank
from app.kb import milvus_client


def arrange_head_tail(items: list) -> list:
    """Keep the strongest item first and the second strongest last."""
    if len(items) <= 2:
        return list(items)
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str, top_k: int | None = None,
    min_score: float | None = None, client=None,
    *, strategy: str | None = None, category: str | None = None,
    collection: str = milvus_client.COLLECTION,
) -> list[dict]:
    """Use the old Lite path by default; opt into Chapter 4 routes explicitly."""
    settings = get_settings()
    if strategy is not None and strategy not in {"vector", "bm25", "hybrid", "hybrid_rerank"}:
        raise ValueError("unsupported retrieval strategy")
    if top_k is None:
        limit = settings.retrieval_top_k if strategy is None else settings.rerank_top_k
    else:
        limit = top_k
    if limit <= 0:
        raise ValueError("top_k must be positive")
    if strategy is not None and min_score is not None:
        raise ValueError("min_score applies only to legacy retrieval")

    if strategy is None:
        threshold = settings.retrieval_min_score if min_score is None else min_score
        vector = await embeddings.embed_query(query)
        if client is None:
            client = milvus_client.get_runtime_client()
        milvus_client.ensure_collection(client)
        hits = milvus_client.search(client, vector, limit)
        return [hit for hit in hits if hit["score"] >= threshold]

    if client is None:
        client = milvus_client.get_runtime_client()
    milvus_client.ensure_collection(client, collection=collection)
    if strategy == "bm25":
        return milvus_client.bm25_search(
            client, query, top_k=limit, category=category, collection=collection,
        )

    vector = await embeddings.embed_query(query)
    if strategy == "vector":
        return milvus_client.dense_search(
            client, vector, top_k=limit, category=category, collection=collection,
        )

    hits = milvus_client.hybrid_search(
        client, vector, query, top_k=settings.recall_top_k,
        recall=settings.recall_top_k, category=category, collection=collection,
    )
    if strategy == "hybrid":
        return hits[:limit]
    if not hits:
        return []
    docs = [f"{hit['question']} {hit['answer']}" for hit in hits]
    ranked = await rerank.rerank(query, docs, top_n=limit)
    return [dict(hits[index], rerank_score=score) for index, score in ranked]
