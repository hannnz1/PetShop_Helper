"""Chapter 7 persistence boundaries against isolated MySQL."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from app.db import repository


@pytest.mark.asyncio
async def test_migration_twice_preserves_existing_conversation_and_message(
    _test_engine, db_session_factory, db_clean,
):
    from scripts.migrate_ch07 import apply_migration
    conversation_id = await repository.create_conversation("owner")
    message_id = await repository.append_message(conversation_id, "user", "原文")
    async with _test_engine.begin() as conn:
        await conn.execute(text("DROP TABLE conversation_summaries"))
        await conn.execute(text("ALTER TABLE conversations DROP COLUMN summary"))
        await conn.execute(text("ALTER TABLE conversations DROP COLUMN summary_upto_msg_id"))
        await conn.execute(text("ALTER TABLE conversations DROP COLUMN layer1_from_msg_id"))
    assert await apply_migration(_test_engine) == "applied"
    assert await apply_migration(_test_engine) == "already applied"
    assert [(row.id, row.content) for row in await repository.list_messages(conversation_id)] == [
        (message_id, "原文"),
    ]


@pytest.mark.asyncio
async def test_snapshot_filters_legacy_tool_rows_and_respects_global_ids(
    db_session_factory, db_clean,
):
    first = await repository.create_conversation("owner")
    other = await repository.create_conversation("owner")
    first_user = await repository.append_message(first, "user", "甲一")
    await repository.append_message(other, "user", "乙一")
    first_reply = await repository.append_message(first, "assistant", "甲答")
    await repository.append_message(first, "tool", "旧工具", tool_call_id="legacy")
    await repository.append_message(other, "assistant", "乙答")
    second_user = await repository.append_message(first, "user", "甲二")
    second_reply = await repository.append_message(first, "assistant", "甲再答")

    assert await repository.advance_layer1(first, first_user) is False
    assert await repository.advance_layer1(first, first_reply) is True
    snapshot = await repository.get_context_snapshot(first, "owner")
    assert snapshot is not None
    assert snapshot.conversation_id == first
    assert snapshot.summary_upto_msg_id == 0
    assert snapshot.layer1_from_msg_id == first_reply
    assert [(m.id, m.role, m.content) for m in snapshot.messages] == [
        (first_user, "user", "甲一"),
        (first_reply, "assistant", "甲答"),
        (second_user, "user", "甲二"),
        (second_reply, "assistant", "甲再答"),
    ]
    assert snapshot.summaries == ()
    assert await repository.get_context_snapshot(first, "someone-else") is None
    assert await repository.advance_layer1(first, first_reply) is False
    assert await repository.advance_layer1(first, second_user) is False
    assert await repository.advance_layer1(first, second_reply) is True
    assert (await repository.get_context_snapshot(first, "owner")).layer1_from_msg_id == second_reply
    assert (await repository.get_context_snapshot(other, "owner")).layer1_from_msg_id == 0


@pytest.mark.asyncio
async def test_snapshot_uses_immutable_ordered_summary_segments_and_summary_anchor(
    _test_engine, db_session_factory, db_clean,
):
    conversation_id = await repository.create_conversation("owner")
    first_user = await repository.append_message(conversation_id, "user", "旧问")
    first_reply = await repository.append_message(conversation_id, "assistant", "旧答")
    later_user = await repository.append_message(conversation_id, "user", "新问")
    later_reply = await repository.append_message(conversation_id, "assistant", "新答")
    async with _test_engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO conversation_summaries
              (conversation_id, seq, from_msg_id, upto_msg_id, content)
            VALUES (:cid, 1, :from_id, :upto, '旧事实')
        """), {"cid": conversation_id, "from_id": first_user, "upto": first_reply})
        await conn.execute(text("""
            UPDATE conversations SET summary='旧事实', summary_upto_msg_id=:upto,
              layer1_from_msg_id=:last WHERE id=:cid
        """), {"cid": conversation_id, "upto": first_reply, "last": later_reply})
    snapshot = await repository.get_context_snapshot(conversation_id, "owner")
    assert snapshot.summary_upto_msg_id == first_reply
    assert [(m.id, m.content) for m in snapshot.messages] == [
        (later_user, "新问"), (later_reply, "新答"),
    ]
    assert [(s.seq, s.from_msg_id, s.upto_msg_id, s.content) for s in snapshot.summaries] == [
        (1, first_user, first_reply, "旧事实"),
    ]
    with pytest.raises(AttributeError):
        snapshot.summaries[0].content = "篡改"


@pytest.mark.asyncio
async def test_summary_segment_rejects_reverse_range_and_duplicate_sequence(
    _test_engine, db_session_factory, db_clean,
):
    conversation_id = await repository.create_conversation("owner")
    async with _test_engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO conversation_summaries
              (conversation_id, seq, from_msg_id, upto_msg_id, content)
            VALUES (:cid, 1, 1, 2, 'first')
        """), {"cid": conversation_id})
    with pytest.raises(IntegrityError):
        async with _test_engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO conversation_summaries
                  (conversation_id, seq, from_msg_id, upto_msg_id, content)
                VALUES (:cid, 1, 3, 4, 'duplicate')
            """), {"cid": conversation_id})
    with pytest.raises((IntegrityError, OperationalError)):
        async with _test_engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO conversation_summaries
                  (conversation_id, seq, from_msg_id, upto_msg_id, content)
                VALUES (:cid, 2, 5, 4, 'reverse')
            """), {"cid": conversation_id})
