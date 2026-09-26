"""Persist knowledge in MySQL, then safely materialize pending rows in Milvus."""

from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """Atomically insert and link a document, or reuse its exact prior run."""
    return await repository.insert_knowledge_document(_rows(chunks))


async def write_pending_report(chunks: list[Chunk]) -> tuple[list[int], int]:
    """Return IDs and the number newly inserted by the same locked transaction."""
    return await repository.insert_knowledge_document_report(_rows(chunks))


async def write_manual_report(chunks: list[Chunk]) -> tuple[list[int], int]:
    """Use the manual question/body fingerprint under the MySQL lock."""
    return await repository.insert_manual_knowledge_report(_rows(chunks))


def _rows(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "category": chunk.category, "questions": chunk.questions,
            "answer": chunk.answer, "section_path": chunk.section_path,
            "content_type": chunk.content_type,
            "is_key_clause": chunk.is_key_clause,
        }
        for chunk in chunks
    ]


async def vectorize_pending(
    client, batch_size: int = 64, collection: str = milvus_client.COLLECTION,
) -> int:
    """Upsert complete batches before marking each corresponding MySQL row done.

    If embedding, upsert, or a status update fails, the unconfirmed rows remain
    pending. A later run repeats their upsert by the same MySQL primary key.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    async with repository.knowledge_write_lock():
        return await _vectorize_pending_locked(client, batch_size, collection)


async def _vectorize_pending_locked(client, batch_size: int, collection: str) -> int:
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
        if collection == milvus_client.COLLECTION:
            rows = [
                {
                    "id": row.id, "vector": vector,
                    "question": row.questions, "answer": row.answer,
                }
                for row, vector in zip(batch, vectors, strict=True)
            ]
            milvus_client.upsert_vectors(client, rows)
        else:
            rows = [
                {
                    "id": row.id, "dense": vector, "text": text,
                    "question": row.questions, "answer": row.answer,
                    "section_path": row.section_path or "",
                    "content_type": row.content_type or "",
                    "category": row.category or "",
                }
                for row, vector, text in zip(batch, vectors, texts, strict=True)
            ]
            milvus_client.upsert_vectors(client, rows, collection=collection)
            # A failed flush leaves MySQL rows pending for an idempotent retry.
            milvus_client.flush(client, collection=collection)
        for row in batch:
            await repository.mark_chunk_vectorized(row.id, str(row.id))
            done += 1
    return done
