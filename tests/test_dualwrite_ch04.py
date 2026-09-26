"""Chapter 4 write path against isolated MySQL and Milvus collections."""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.db import repository
from app.kb import dualwrite, milvus_client as mc
from app.kb.documents import Chunk


@pytest.fixture
def hybrid_client():
    client = mc.get_client(uri="http://127.0.0.1:19530")
    name = f"test_ch04_write_{uuid.uuid4().hex[:12]}"
    try:
        mc.ensure_collection(client, collection=name)
        yield client, name
    finally:
        mc.drop(client, name)
        client.close()


def _chunk() -> Chunk:
    return Chunk(
        category="运费", questions="运费怎么算", answer="满99元包邮",
        section_path="运费政策", content_type="faq", is_key_clause=1,
    )


@pytest.mark.asyncio
async def test_hybrid_upsert_flushes_before_mark_done(monkeypatch):
    events = []
    row = SimpleNamespace(
        id=42, category="运费", questions="运费怎么算", answer="满99元包邮",
        section_path="运费政策", content_type="faq",
    )

    @asynccontextmanager
    async def fake_lock():
        events.append("lock")
        yield

    async def fake_pending():
        return [row]

    async def fake_embed(texts):
        assert texts == ["运费\n运费怎么算\n满99元包邮"]
        return [[1.0] + [0.0] * (mc.DIM - 1)]

    def fake_upsert(client, rows, *, collection):
        events.append("upsert")
        assert collection == "scratch"
        assert rows[0] == {
            "id": 42, "dense": [1.0] + [0.0] * (mc.DIM - 1),
            "text": "运费\n运费怎么算\n满99元包邮",
            "question": "运费怎么算", "answer": "满99元包邮",
            "section_path": "运费政策", "content_type": "faq", "category": "运费",
        }

    def fake_flush(client, *, collection):
        events.append("flush")
        assert collection == "scratch"

    async def fake_mark(chunk_id, vector_id):
        events.append("mark")
        assert (chunk_id, vector_id) == (42, "42")

    monkeypatch.setattr(dualwrite.repository, "knowledge_write_lock", fake_lock)
    monkeypatch.setattr(dualwrite.repository, "list_pending_chunks", fake_pending)
    monkeypatch.setattr(dualwrite.repository, "mark_chunk_vectorized", fake_mark)
    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", fake_embed)
    monkeypatch.setattr(mc, "upsert_vectors", fake_upsert)
    monkeypatch.setattr(mc, "flush", fake_flush)
    assert await dualwrite.vectorize_pending(None, collection="scratch") == 1
    assert events == ["lock", "upsert", "flush", "mark"]


@pytest.mark.asyncio
async def test_hybrid_flush_failure_never_marks_done(monkeypatch):
    events = []
    row = SimpleNamespace(
        id=42, category="运费", questions="运费怎么算", answer="满99元包邮",
        section_path=None, content_type=None,
    )

    @asynccontextmanager
    async def fake_lock():
        yield

    async def fake_pending():
        return [row]

    async def fake_embed(texts):
        return [[1.0] + [0.0] * (mc.DIM - 1)]

    async def fake_mark(chunk_id, vector_id):
        events.append("mark")

    def fake_upsert(client, rows, *, collection):
        events.append("upsert")
        assert rows[0]["section_path"] == rows[0]["content_type"] == ""

    def fail_flush(client, *, collection):
        events.append("flush")
        raise RuntimeError("flush interrupted")

    monkeypatch.setattr(dualwrite.repository, "knowledge_write_lock", fake_lock)
    monkeypatch.setattr(dualwrite.repository, "list_pending_chunks", fake_pending)
    monkeypatch.setattr(dualwrite.repository, "mark_chunk_vectorized", fake_mark)
    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", fake_embed)
    monkeypatch.setattr(mc, "upsert_vectors", fake_upsert)
    monkeypatch.setattr(mc, "flush", fail_flush)
    with pytest.raises(RuntimeError, match="flush interrupted"):
        await dualwrite.vectorize_pending(None, collection="scratch")
    assert events == ["upsert", "flush"]


@pytest.mark.asyncio
async def test_vectorize_writes_bm25_and_metadata(
    db_session_factory, db_clean, hybrid_client, monkeypatch,
):
    async def fake_embed(texts):
        assert texts == ["运费\n运费怎么算\n满99元包邮"]
        return [[1.0] + [0.0] * (mc.DIM - 1)]

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", fake_embed)
    ids = await dualwrite.write_pending([_chunk()])
    client, name = hybrid_client
    assert await dualwrite.vectorize_pending(client, collection=name) == 1
    assert await repository.count_chunks_by_status("done") == 1
    assert await dualwrite.vectorize_pending(client, collection=name) == 0

    hits = mc.bm25_search(client, "运费 包邮", top_k=1, collection=name)
    assert hits and hits[0]["id"] == ids[0]
    assert (hits[0]["section_path"], hits[0]["content_type"], hits[0]["category"]) == (
        "运费政策", "faq", "运费",
    )
    stored = client.get(collection_name=name, ids=ids, output_fields=["text", "question", "answer"])
    assert stored[0]["text"] == "运费\n运费怎么算\n满99元包邮"
    assert (stored[0]["question"], stored[0]["answer"]) == ("运费怎么算", "满99元包邮")


@pytest.mark.asyncio
async def test_flush_failure_keeps_mysql_pending_for_retry(
    db_session_factory, db_clean, hybrid_client, monkeypatch,
):
    async def fake_embed(texts):
        return [[1.0] + [0.0] * (mc.DIM - 1) for _ in texts]

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", fake_embed)
    ids = await dualwrite.write_pending([_chunk()])
    client, name = hybrid_client
    def fail_flush(_client, *, collection):
        raise RuntimeError("flush interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(mc, "flush", fail_flush)
        with pytest.raises(RuntimeError, match="flush interrupted"):
            await dualwrite.vectorize_pending(client, collection=name)
    assert await repository.count_chunks_by_status("pending") == 1
    assert await repository.count_chunks_by_status("done") == 0

    assert await dualwrite.vectorize_pending(client, collection=name) == 1
    assert await repository.count_chunks_by_status("done") == 1
    assert mc.count(client, collection=name) == 1
    assert client.get(collection_name=name, ids=ids, output_fields=["category"])[0]["category"] == "运费"
