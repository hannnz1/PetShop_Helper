"""Dual-write failure recovery against an isolated MySQL schema and Milvus Lite."""

import pytest

from app.db import repository
from app.db.models import KnowledgeChunk
from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk


def _chunk(index: int) -> Chunk:
    return Chunk(
        category="物流", questions=f"运费问题 {index}", answer=f"运费答案 {index}",
        section_path=f"规则 / 运费 {index}", content_type="policy",
        is_key_clause=int(index == 0),
    )


def _vector(seed: float = 1.0) -> list[float]:
    return [seed, 1.0] + [0.0] * (milvus_client.DIM - 2)


def _milvus_ids(client) -> set[int]:
    return {
        row["id"] for row in client.query(
            milvus_client.COLLECTION, filter="id >= 0", output_fields=["id"]
        )
    }


@pytest.fixture()
def milvus(tmp_path):
    client = milvus_client.get_client(uri=str(tmp_path / "dualwrite.db"))
    milvus_client.ensure_collection(client)
    yield client
    client.close()


@pytest.mark.asyncio
async def test_write_pending_links_neighbors_and_preserves_fields(db_session_factory, db_clean):
    chunks = [_chunk(i) for i in range(3)]
    ids = await dualwrite.write_pending(chunks)
    assert await dualwrite.write_pending(chunks) == ids
    rows = {row.id: row for row in await repository.list_pending_chunks()}
    assert len(ids) == len(rows) == 3
    assert [(rows[cid].prev_chunk_id, rows[cid].next_chunk_id) for cid in ids] == [
        (None, ids[1]), (ids[0], ids[2]), (ids[1], None),
    ]
    assert (rows[ids[0]].section_path, rows[ids[0]].content_type, rows[ids[0]].is_key_clause) == (
        "规则 / 运费 0", "policy", 1,
    )


@pytest.mark.asyncio
async def test_vectorize_resumes_after_second_batch_crash(
    db_session_factory, db_clean, milvus, monkeypatch
):
    ids = await dualwrite.write_pending([_chunk(i) for i in range(4)])
    calls = 0

    async def flaky_embed(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("embedding interrupted")
        assert texts == [
            "物流\n运费问题 0\n运费答案 0", "物流\n运费问题 1\n运费答案 1"
        ]
        return [_vector() for _ in texts]

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", flaky_embed)
    with pytest.raises(RuntimeError, match="embedding interrupted"):
        await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert await repository.count_chunks_by_status("done") == 2
    assert [row.id for row in await repository.list_pending_chunks()] == ids[2:]
    assert _milvus_ids(milvus) == set(ids[:2])

    async def good_embed(texts):
        return [_vector() for _ in texts]

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", good_embed)
    assert await dualwrite.vectorize_pending(milvus, batch_size=2) == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    assert _milvus_ids(milvus) == set(ids)
    async with db_session_factory() as session:
        rows = [await session.get(KnowledgeChunk, cid) for cid in ids]
    assert [row.vector_id for row in rows] == [str(cid) for cid in ids]
    assert await dualwrite.vectorize_pending(milvus, batch_size=2) == 0
    assert _milvus_ids(milvus) == set(ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["short", "wrong_dimension"])
async def test_invalid_embedding_batch_stays_pending(
    db_session_factory, db_clean, milvus, monkeypatch, kind
):
    ids = await dualwrite.write_pending([_chunk(i) for i in range(2)])

    async def invalid_embed(texts):
        return [_vector()] if kind == "short" else [_vector(), [0.0] * (milvus_client.DIM - 1)]

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", invalid_embed)
    with pytest.raises(ValueError, match="embedding"):
        await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert [row.id for row in await repository.list_pending_chunks()] == ids
    assert _milvus_ids(milvus) == set()


@pytest.mark.asyncio
async def test_upsert_failure_stays_pending(db_session_factory, db_clean, milvus, monkeypatch):
    ids = await dualwrite.write_pending([_chunk(i) for i in range(2)])

    async def good_embed(texts):
        return [_vector() for _ in texts]

    def fail_upsert(client, rows):
        raise RuntimeError("Milvus unavailable")

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", good_embed)
    with monkeypatch.context() as patch:
        patch.setattr(dualwrite.milvus_client, "upsert_vectors", fail_upsert)
        with pytest.raises(RuntimeError, match="Milvus unavailable"):
            await dualwrite.vectorize_pending(milvus)
    assert [row.id for row in await repository.list_pending_chunks()] == ids
    assert _milvus_ids(milvus) == set()
    assert await dualwrite.vectorize_pending(milvus) == 2
    assert _milvus_ids(milvus) == set(ids)


@pytest.mark.asyncio
async def test_mysql_mark_done_failure_is_recoverable(
    db_session_factory, db_clean, milvus, monkeypatch
):
    ids = await dualwrite.write_pending([_chunk(i) for i in range(2)])

    async def good_embed(texts):
        return [_vector() for _ in texts]

    real_mark = repository.mark_chunk_vectorized
    calls = 0

    async def fail_second_mark(chunk_id, vector_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("MySQL mark failed")
        await real_mark(chunk_id, vector_id)

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", good_embed)
    with monkeypatch.context() as patch:
        patch.setattr(dualwrite.repository, "mark_chunk_vectorized", fail_second_mark)
        with pytest.raises(RuntimeError, match="MySQL mark failed"):
            await dualwrite.vectorize_pending(milvus)
    assert await repository.count_chunks_by_status("done") == 1
    assert [row.id for row in await repository.list_pending_chunks()] == ids[1:]
    assert _milvus_ids(milvus) == set(ids)
    assert await dualwrite.vectorize_pending(milvus) == 1
    assert await repository.count_chunks_by_status("done") == 2
    assert _milvus_ids(milvus) == set(ids)


@pytest.mark.asyncio
async def test_empty_input_and_invalid_batch_size(db_session_factory, db_clean, milvus):
    assert await dualwrite.write_pending([]) == []
    with pytest.raises(ValueError, match="batch_size"):
        await dualwrite.vectorize_pending(milvus, batch_size=0)


@pytest.mark.asyncio
async def test_default_standalone_pending_write_uses_hybrid_shape(db_session_factory, db_clean, monkeypatch):
    ids = await dualwrite.write_pending([_chunk(0)])

    async def good_embed(texts):
        return [_vector() for _ in texts]

    submitted = []
    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", good_embed)
    monkeypatch.setattr(dualwrite.milvus_client, "client_uses_hybrid", lambda client: True, raising=False)
    monkeypatch.setattr(dualwrite.milvus_client, "upsert_vectors", lambda client, rows, collection=None: submitted.extend(rows))
    monkeypatch.setattr(dualwrite.milvus_client, "flush", lambda client, collection=None: None)
    assert await dualwrite.vectorize_pending(object()) == 1
    assert submitted[0]["id"] == ids[0]
    assert "dense" in submitted[0] and "text" in submitted[0]
    assert "vector" not in submitted[0]
