"""Asynchronous persistence for customer-service conversations and tool data."""

from uuid import uuid4

from sqlalchemy import func, select

import app.db.base as db
from app.db.models import Conversation, Faq, KnowledgeChunk, Message, QaExtractionStaging, Ticket


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
    """Match text literally and return at most ten FAQ rows in stable order."""

    literal = keyword.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    async with db.async_session() as session:
        result = await session.execute(
            select(Faq)
            .where(Faq.question.like(f"%{literal}%", escape="!"))
            .order_by(Faq.id)
            .limit(10)
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


async def insert_knowledge_chunk(
    category: str,
    questions: str,
    answer: str,
    section_path: str | None = None,
    content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session.begin() as session:
        row = KnowledgeChunk(
            category=category,
            questions=questions,
            answer=answer,
            section_path=section_path,
            content_type=content_type,
            is_key_clause=is_key_clause,
        )
        session.add(row)
        await session.flush()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as session:
        result = await session.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session.begin() as session:
        row = await session.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"


async def set_chunk_neighbors(
    chunk_id: int, prev_id: int | None, next_id: int | None
) -> None:
    async with db.async_session.begin() as session:
        row = await session.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as session:
        result = await session.execute(
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as session:
        result = await session.execute(
            select(KnowledgeChunk.questions).order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def insert_staging(
    batch_no: str, source_ref: str | None, question: str, answer: str
) -> int:
    async with db.async_session.begin() as session:
        row = QaExtractionStaging(
            batch_no=batch_no,
            source_ref=source_ref,
            question=question,
            answer=answer,
        )
        session.add(row)
        await session.flush()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as session:
        result = await session.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session.begin() as session:
        for staging_id in ids:
            row = await session.get(QaExtractionStaging, staging_id)
            if row is not None:
                row.status = status
