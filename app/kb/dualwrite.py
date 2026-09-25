"""Persist knowledge in MySQL, then safely materialize pending rows in Milvus."""

from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """Atomically insert and link a document, or reuse its exact prior run."""
    return await repository.insert_knowledge_document([
        {
            "category": chunk.category, "questions": chunk.questions,
            "answer": chunk.answer, "section_path": chunk.section_path,
            "content_type": chunk.content_type,
            "is_key_clause": chunk.is_key_clause,
        }
        for chunk in chunks
    ])


async def vectorize_pending(client, batch_size: int = 64) -> int:
    """Upsert complete batches before marking each corresponding MySQL row done.

    If embedding, upsert, or a status update fails, the unconfirmed rows remain
    pending. A later run repeats their upsert by the same MySQL primary key.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pending = await repository.list_pending_chunks()
    done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        texts = [f"{row.category}\n{row.questions}\n{row.answer}" for row in batch]
        vectors = await embeddings.embed_texts(texts)
        if len(vectors) != len(batch):
            raise ValueError("embedding count differs from pending batch")
        if any(len(vector) != milvus_client.DIM for vector in vectors):
            raise ValueError("embedding dimension differs from Milvus collection")
        rows = [
            {
                "id": row.id, "vector": vector,
                "question": row.questions, "answer": row.answer,
            }
            for row, vector in zip(batch, vectors, strict=True)
        ]
        milvus_client.upsert_vectors(client, rows)
        for row in batch:
            await repository.mark_chunk_vectorized(row.id, str(row.id))
            done += 1
    return done
