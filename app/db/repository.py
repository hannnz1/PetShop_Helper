"""Asynchronous persistence for customer-service conversations and tool data."""

from uuid import uuid4

from sqlalchemy import select

import app.db.base as db
from app.db.models import Conversation, Faq, Message, Ticket


def _new_ticket_no() -> str:
    # The DDL limits ticket_no to 32 characters. UUID randomness works across
    # process restarts and workers; the database primary key remains authoritative.
    return f"T{uuid4().hex[:31]}"


async def create_conversation(user_id: str) -> int:
    async with db.async_session.begin() as session:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.flush()
        return conversation.id


async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as session:
        return await session.get(Conversation, conversation_id)


async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session.begin() as session:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        session.add(message)
        await session.flush()
        return message.id


async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as session:
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())


async def search_faq(keyword: str) -> list[Faq]:
    """Match the supplied text literally in FAQ questions (no semantic rewrite)."""

    literal = keyword.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    async with db.async_session() as session:
        result = await session.execute(
            select(Faq).where(Faq.question.like(f"%{literal}%", escape="!")).order_by(Faq.id)
        )
        return list(result.scalars())


async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    """Persist a ticket and its conversation handoff in one transaction."""

    ticket_no = _new_ticket_no()
    async with db.async_session.begin() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError(f"conversation {conversation_id} does not exist")
        session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conversation.status = "已转人工"
    return ticket_no
