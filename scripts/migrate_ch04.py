"""Copy all MySQL knowledge rows to local Milvus Standalone's hybrid collection.

Run while the web and knowledge jobs are stopped. No MySQL statuses or Lite data
are changed. A failed batch can be retried because Milvus IDs equal MySQL IDs.
"""

import asyncio

from app.core import embeddings
from app.db import repository
from app.kb import milvus_client


STANDALONE_URI = "http://127.0.0.1:19530"


async def materialize_rows(client, rows: list, batch_size: int = 32) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [f"{row.category}\n{row.questions}\n{row.answer}" for row in batch]
        vectors = await embeddings.embed_texts(texts)
        if len(vectors) != len(batch):
            raise ValueError("embedding count differs from MySQL batch")
        if any(len(vector) != milvus_client.DIM for vector in vectors):
            raise ValueError("embedding dimension differs from Milvus schema")
        data = [
            {
                "id": row.id, "dense": vector, "text": text,
                "question": row.questions, "answer": row.answer,
                "section_path": row.section_path or "",
                "content_type": row.content_type or "",
                "category": row.category or "",
            }
            for row, vector, text in zip(batch, vectors, texts, strict=True)
        ]
        milvus_client.upsert_vectors(client, data, collection=milvus_client.COLLECTION)
        milvus_client.flush(client, collection=milvus_client.COLLECTION)
        print(f"Standalone upserted {start + len(batch)}/{len(rows)} MySQL rows")


async def migrate() -> int:
    client = milvus_client.get_client(uri=STANDALONE_URI)
    try:
        async with repository.knowledge_write_lock():
            rows = await repository.list_all_chunks()
            milvus_client.ensure_collection(client, hybrid=True)
            await materialize_rows(client, rows)
            actual_count = milvus_client.count(client)
            if actual_count != len(rows):
                raise RuntimeError(f"Standalone count {actual_count} differs from MySQL {len(rows)}")
            # Check every authority ID, not only the aggregate count. Current
            # local data is well below Milvus's query result cap.
            if len(rows) > 16000:
                raise RuntimeError("ID verification requires paginated query for over 16000 rows")
            observed = {
                item["id"] for item in client.query(
                    milvus_client.COLLECTION, filter="id >= 0",
                    output_fields=["id"], limit=max(len(rows), 1),
                )
            }
            expected = {row.id for row in rows}
            if observed != expected:
                raise RuntimeError("Standalone IDs differ from MySQL authority IDs")
            print(f"Standalone migration verified: {len(rows)} matching MySQL IDs")
            return len(rows)
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(migrate())
