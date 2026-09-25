"""Knowledge and extraction staging persistence in the isolated MySQL schema."""

import pytest

from app.db import repository
from app.db.models import KnowledgeChunk


@pytest.mark.asyncio
async def test_insert_and_list_pending(db_session_factory, db_clean):
    first = await repository.insert_knowledge_chunk(
        "物流", "运费说明", "满 99 包邮", "规则 / 运费", "policy", 1
    )
    second = await repository.insert_knowledge_chunk("售后", "退货", "七天无理由")
    pending = await repository.list_pending_chunks()
    assert [row.id for row in pending] == [first, second]
    assert (pending[0].category, pending[0].questions, pending[0].answer) == (
        "物流", "运费说明", "满 99 包邮"
    )
    assert (pending[0].section_path, pending[0].content_type, pending[0].is_key_clause) == (
        "规则 / 运费", "policy", 1
    )
    assert pending[1].vectorize_status == "pending"


@pytest.mark.asyncio
async def test_mark_vectorized_persists_id_and_status(db_session_factory, db_clean):
    first = await repository.insert_knowledge_chunk("物流", "运费", "满 99 包邮")
    second = await repository.insert_knowledge_chunk("售后", "退货", "七天无理由")
    await repository.mark_chunk_vectorized(first, str(first))
    assert [row.id for row in await repository.list_pending_chunks()] == [second]
    assert await repository.count_chunks_by_status("pending") == 1
    assert await repository.count_chunks_by_status("done") == 1
    async with db_session_factory() as session:
        row = await session.get(KnowledgeChunk, first)
    assert row is not None and row.vector_id == str(first)


@pytest.mark.asyncio
async def test_set_neighbors_persists_both_links(db_session_factory, db_clean):
    first = await repository.insert_knowledge_chunk("c", "q1", "a1")
    second = await repository.insert_knowledge_chunk("c", "q2", "a2")
    third = await repository.insert_knowledge_chunk("c", "q3", "a3")
    await repository.set_chunk_neighbors(second, first, third)
    rows = {row.id: row for row in await repository.list_pending_chunks()}
    assert (rows[second].prev_chunk_id, rows[second].next_chunk_id) == (first, third)


@pytest.mark.asyncio
async def test_staging_flow(db_session_factory, db_clean):
    first = await repository.insert_staging("batch-a", "conv:1", "邮费多少", "满 99 包邮")
    second = await repository.insert_staging("batch-a", None, "如何退货", "七天无理由")
    extracted = await repository.list_staging_by_status("extracted")
    assert [row.id for row in extracted] == [first, second]
    assert (extracted[0].batch_no, extracted[0].source_ref) == ("batch-a", "conv:1")
    await repository.set_staging_status([first], "discarded")
    assert [row.id for row in await repository.list_staging_by_status("extracted")] == [second]
    assert [row.id for row in await repository.list_staging_by_status("discarded")] == [first]
    await repository.set_staging_status([], "kept")
    assert await repository.list_staging_by_status("kept") == []


@pytest.mark.asyncio
async def test_list_all_questions(db_session_factory, db_clean):
    await repository.insert_knowledge_chunk("c", "运费怎么算", "满 99 包邮")
    await repository.insert_knowledge_chunk("c", "退货\n退款", "七天无理由")
    assert await repository.list_all_questions() == ["运费怎么算", "退货\n退款"]
