"""Low-confidence question log in the isolated MySQL test schema."""

import pytest
from sqlalchemy import text

from app.db import repository


@pytest.mark.asyncio
async def test_insert_low_confidence_with_conversation(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("acceptance-user")
    entry_id = await repository.insert_low_confidence(
        conversation_id, "邮费到底多少", "self_check", "证据不足",
    )
    assert entry_id > 0
    async with db_session_factory() as session:
        row = (await session.execute(text(
            "SELECT conversation_id, raw_question, source, reason "
            "FROM low_confidence_questions WHERE id = :id"
        ), {"id": entry_id})).one()
    assert tuple(row) == (conversation_id, "邮费到底多少", "self_check", "证据不足")


@pytest.mark.asyncio
async def test_insert_low_confidence_without_conversation(db_session_factory, db_clean):
    entry_id = await repository.insert_low_confidence(
        None, "未知型号保修", "retrieval_low_conf", None,
    )
    async with db_session_factory() as session:
        row = (await session.execute(text(
            "SELECT conversation_id, source, reason FROM low_confidence_questions WHERE id = :id"
        ), {"id": entry_id})).one()
    assert tuple(row) == (None, "retrieval_low_conf", None)


@pytest.mark.asyncio
async def test_reject_unsupported_low_confidence_source(db_session_factory, db_clean):
    with pytest.raises(ValueError, match="source"):
        await repository.insert_low_confidence(None, "问题", "unknown", None)
