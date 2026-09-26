"""Copy all MySQL knowledge rows to Standalone without changing source status."""

import pytest

from app.db import repository
from app.kb.documents import Chunk
from scripts import migrate_ch04


@pytest.mark.asyncio
async def test_migration_includes_done_and_pending_without_marking(db_session_factory, db_clean, monkeypatch):
    chunks = [
        Chunk(category="商品", questions="W20", answer="每2周", section_path="演示 / W20", content_type="faq"),
        Chunk(category="商品", questions="W40", answer="每3周", section_path="演示 / W40", content_type="faq"),
    ]
    ids = await repository.insert_knowledge_document([{
        "category": c.category, "questions": c.questions, "answer": c.answer,
        "section_path": c.section_path, "content_type": c.content_type, "is_key_clause": 0,
    } for c in chunks])
    await repository.mark_chunk_vectorized(ids[0], str(ids[0]))
    rows = await repository.list_all_chunks()
    assert [row.id for row in rows] == ids
    assert [row.vectorize_status for row in rows] == ["done", "pending"]

    async def embed(texts):
        assert len(texts) == 2
        return [[0.1] * 1024 for _ in texts]

    submitted = []
    flushed = []
    monkeypatch.setattr(migrate_ch04.embeddings, "embed_texts", embed)
    monkeypatch.setattr(migrate_ch04.milvus_client, "upsert_vectors", lambda client, data, collection: submitted.extend(data))
    monkeypatch.setattr(migrate_ch04.milvus_client, "flush", lambda client, collection: flushed.append(collection))
    await migrate_ch04.materialize_rows(object(), rows, batch_size=2)
    assert [row["id"] for row in submitted] == ids
    assert all({"dense", "text", "section_path", "content_type", "category"} <= row.keys() for row in submitted)
    assert flushed == ["knowledge"]
    assert [row.vectorize_status for row in await repository.list_all_chunks()] == ["done", "pending"]
