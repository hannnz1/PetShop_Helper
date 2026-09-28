"""Web-owned Milvus client and existing MySQL write lock for vector retries."""

from app.kb import dualwrite, milvus_client


async def vectorize_pending_knowledge() -> int:
    client = milvus_client.get_runtime_client()
    milvus_client.ensure_collection(client)
    return await dualwrite.vectorize_pending(client)
