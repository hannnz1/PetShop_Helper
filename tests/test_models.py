"""Verify the four ORM mappings against the authoritative MySQL DDL."""

import asyncio

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.dialects.mysql import BIGINT, ENUM
from sqlalchemy.schema import FetchedValue

from app.db.models import Conversation, Faq, Message, Ticket
from app.db.base import Base


def test_mappings_match_key_mysql_ddl_properties():
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}
    for table_name in ("conversations", "messages", "faq"):
        table = Base.metadata.tables[table_name]
        assert isinstance(table.c.id.type, BIGINT)
        assert table.c.id.type.unsigned is True
        assert table.c.id.primary_key and table.c.id.autoincrement is True

    assert isinstance(Message.__table__.c.conversation_id.type, BIGINT)
    assert Message.__table__.c.conversation_id.type.unsigned is True
    assert isinstance(Ticket.__table__.c.conversation_id.type, BIGINT)
    assert Ticket.__table__.c.conversation_id.type.unsigned is True
    assert isinstance(Conversation.__table__.c.status.type, ENUM)
    assert Conversation.__table__.c.status.type.enums == ["进行中", "已转人工", "已结束"]
    assert Message.__table__.c.role.type.enums == ["user", "assistant", "tool"]
    assert Ticket.__table__.c.ticket_type.type.enums == ["售后", "投诉", "咨询"]
    assert Ticket.__table__.c.status.type.enums == ["待处理", "已处理"]
    for model in (Conversation, Faq):
        updated_at = model.__table__.c.updated_at
        assert "ON UPDATE CURRENT_TIMESTAMP" in str(updated_at.server_default.arg)
        assert isinstance(updated_at.server_onupdate, FetchedValue)

    assert {column.name for column in inspect(Ticket).primary_key} == {"ticket_no"}
    for model in (Message, Ticket):
        assert {fk.target_fullname for fk in model.__table__.foreign_keys} == {"conversations.id"}


@pytest.mark.asyncio
async def test_conversation_defaults_and_autoincrement(db_session_factory, db_clean):
    async with db_session_factory() as session:
        conversation = Conversation(user_id="u1")
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        assert conversation.id is not None
        assert conversation.status == "进行中"
        assert conversation.created_at is not None
        assert conversation.updated_at is not None


@pytest.mark.asyncio
async def test_message_json_and_ticket_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as session:
        conversation = Conversation(user_id="u1")
        session.add(conversation)
        await session.flush()
        message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        )
        ticket = Ticket(
            ticket_no="T20260701008",
            conversation_id=conversation.id,
            description="商品损坏",
            ticket_type="售后",
        )
        session.add_all([message, ticket])
        await session.commit()
        await session.refresh(message)
        await session.refresh(ticket)
        assert message.tool_calls[0]["name"] == "query_order"
        assert message.role == "assistant"
        assert message.content is None
        assert ticket.status == "待处理"


@pytest.mark.asyncio
async def test_faq_server_updates_timestamp(db_session_factory, db_clean):
    async with db_session_factory() as session:
        faq = Faq(question="退货政策？", answer="七天内", category="售后")
        session.add(faq)
        await session.commit()
        await session.refresh(faq)
        original = faq.updated_at
        await asyncio.sleep(1.1)
        faq.answer = "十四天内"
        await session.commit()
        await session.refresh(faq)
        assert faq.updated_at > original
        assert (await session.execute(select(Faq).where(Faq.id == faq.id))).scalar_one().answer == "十四天内"
