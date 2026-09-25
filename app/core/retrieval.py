"""Single-route dense retrieval through BGE-M3 and Milvus Lite."""

from app.config import get_settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None,
    min_score: float | None = None, client=None,
) -> list[dict]:
    """Return COSINE hits at or above the selected score threshold."""
    settings = get_settings()
    limit = settings.retrieval_top_k if top_k is None else top_k
    threshold = settings.retrieval_min_score if min_score is None else min_score
    if limit <= 0:
        raise ValueError("top_k must be positive")
    vector = await embeddings.embed_query(query)
    if client is None:
        client = milvus_client.get_runtime_client()
    milvus_client.ensure_collection(client)
    hits = milvus_client.search(client, vector, limit)
    return [hit for hit in hits if hit["score"] >= threshold]
