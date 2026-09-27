"""Unresolved-feedback intake uses the conversation's own user message."""

import pytest
from sqlalchemy import text

from app.db import repository


@pytest.mark.asyncio
async def test_feedback_uses_nearest_owned_question_and_is_idempotent(
    db_session_factory, db_clean,
):
    from app.db.flywheel import record_unresolved_feedback

    conversation_id = await repository.create_conversation("alice")
    await repository.append_message(conversation_id, "user", "之前的问题")
    await repository.append_message(conversation_id, "assistant", "之前的回答")
    await repository.append_message(conversation_id, "user", "我的猫粮能退吗？")
    answer_id = await repository.append_message(conversation_id, "assistant", "请联系客服")

    first = await record_unresolved_feedback("alice", conversation_id, answer_id)
    second = await record_unresolved_feedback("alice", conversation_id, answer_id)
    assert first == second
    async with db_session_factory() as session:
        rows = (await session.execute(text(
            "SELECT raw_question, source, source_ref FROM low_confidence_questions "
            "WHERE conversation_id = :conversation_id AND source = 'user_feedback'"
        ), {"conversation_id": conversation_id})).all()
    assert rows == [("我的猫粮能退吗？", "user_feedback", f"assistant:{answer_id}")]


@pytest.mark.asyncio
async def test_feedback_rejects_foreign_or_non_assistant_message(db_session_factory, db_clean):
    from app.db.flywheel import record_unresolved_feedback

    conversation_id = await repository.create_conversation("alice")
    question_id = await repository.append_message(conversation_id, "user", "为什么没解决")
    answer_id = await repository.append_message(conversation_id, "assistant", "抱歉")
    with pytest.raises(LookupError):
        await record_unresolved_feedback("bob", conversation_id, answer_id)
    with pytest.raises(LookupError):
        await record_unresolved_feedback("alice", conversation_id, question_id)
    with pytest.raises(LookupError):
        await record_unresolved_feedback("alice", conversation_id, answer_id + 1000)


@pytest.mark.asyncio
async def test_existing_automatic_low_confidence_row_still_writes(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("alice")
    row_id = await repository.insert_low_confidence(
        conversation_id, "没有保修证据", "self_check", "资料不足",
    )
    async with db_session_factory() as session:
        row = (await session.execute(text(
            "SELECT source_ref FROM low_confidence_questions WHERE id = :id"
        ), {"id": row_id})).scalar_one()
    assert row is None
