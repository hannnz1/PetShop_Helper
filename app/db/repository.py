"""Asynchronous persistence for customer-service conversations and tool data."""

from __future__ import annotations

from uuid import uuid4
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
import unicodedata

from sqlalchemy import and_, case, delete, func, insert, or_, select, text, update
from sqlalchemy.exc import IntegrityError

import app.db.base as db
from app.db.models import (
    Conversation, ConversationSummary, FaithCase, Faq, KnowledgeChunk, LowConfidenceQuestion,
    Message, QaExtractionStaging, RefundRequest, SampleOrder, Ticket,
)


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


async def list_owned_conversations(
    user_id: str, limit: int, before: tuple[datetime, int] | None = None,
) -> tuple[list[dict], tuple[datetime, int] | None]:
    """Page owned conversations by update time and ID; include empty conversations."""
    first_question = (select(Message.content).where(
        Message.conversation_id == Conversation.id,
        Message.role == "user",
        Message.content.is_not(None),
    ).order_by(Message.id).limit(1).correlate(Conversation).scalar_subquery())
    factual_segment = (select(ConversationSummary.id).where(
        ConversationSummary.conversation_id == Conversation.id,
        func.length(func.trim(ConversationSummary.content)) > 0,
    ).limit(1).correlate(Conversation).scalar_subquery())
    statement = select(Conversation, first_question, factual_segment).where(Conversation.user_id == user_id)
    if before is not None:
        when, conversation_id = before
        statement = statement.where(or_(
            Conversation.updated_at < when,
            and_(Conversation.updated_at == when, Conversation.id < conversation_id),
        ))
    statement = statement.order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit + 1)
    async with db.async_session() as session:
        rows = (await session.execute(statement)).all()
    page = rows[:limit]
    items = [{
        "id": conversation.id,
        "preview": (question or "")[:100],
        "has_summary": segment_id is not None,
        "updated_at": conversation.updated_at.isoformat(),
    } for conversation, question, segment_id in page]
    cursor = (page[-1][0].updated_at, page[-1][0].id) if len(rows) > limit else None
    return items, cursor


async def list_owned_visible_messages(
    conversation_id: int, user_id: str, limit: int, after: int | None = None,
) -> tuple[list[VisibleMessage], int | None] | None:
    """Return one bounded page of original user/assistant text for an owner."""
    async with db.async_session() as session:
        owned = await session.scalar(select(Conversation.id).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id,
        ))
        if owned is None:
            return None
        statement = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.role.in_(("user", "assistant")),
            Message.content.is_not(None),
            Message.tool_call_id.is_(None),
            or_(Message.tool_calls.is_(None), func.json_type(Message.tool_calls) == "NULL"),
        )
        if after is not None:
            statement = statement.where(Message.id > after)
        rows = (await session.scalars(statement.order_by(Message.id).limit(limit + 1))).all()
    page = rows[:limit]
    return [VisibleMessage(row.id, row.role, row.content) for row in page], (
        page[-1].id if len(rows) > limit else None
    )


async def last_message_id(conversation_id: int) -> int | None:
    """Return the last committed MySQL audit row for a conversation."""
    async with db.async_session() as session:
        return await session.scalar(select(func.max(Message.id)).where(
            Message.conversation_id == conversation_id,
        ))


@dataclass(frozen=True)
class VisibleMessage:
    id: int
    role: str
    content: str


@dataclass(frozen=True)
class SummarySegment:
    seq: int
    from_msg_id: int
    upto_msg_id: int
    content: str


@dataclass(frozen=True)
class ContextSnapshot:
    conversation_id: int
    summary_upto_msg_id: int
    layer1_from_msg_id: int
    messages: tuple[VisibleMessage, ...]
    summaries: tuple[SummarySegment, ...]


@dataclass(frozen=True)
class SummaryWork:
    summary_upto_msg_id: int
    layer1_from_msg_id: int
    messages: tuple[VisibleMessage, ...]


async def list_summary_candidates() -> list[int]:
    """Find persisted Layer2 gaps after a restart; the worker rechecks tokens."""
    async with db.async_session() as session:
        rows = await session.scalars(select(Conversation.id).where(
            Conversation.layer1_from_msg_id.is_not(None),
            Conversation.layer1_from_msg_id > func.coalesce(Conversation.summary_upto_msg_id, 0),
        ).order_by(Conversation.id))
        return list(rows)


async def get_summary_work(conversation_id: int) -> SummaryWork | None:
    """Return only unsummarized visible rows through the completed Layer2 boundary."""
    async with db.async_session() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            return None
        upto = conversation.summary_upto_msg_id or 0
        boundary = conversation.layer1_from_msg_id or 0
        rows = (await session.scalars(select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id > upto, Message.id <= boundary,
            Message.role.in_(("user", "assistant")),
            Message.content.is_not(None),
        ).order_by(Message.id))).all()
        return SummaryWork(upto, boundary, tuple(
            VisibleMessage(row.id, row.role, row.content) for row in rows
        ))


async def commit_summary_segment(conversation_id: int, expected_upto: int,
                                 from_msg_id: int, upto_msg_id: int, content: str) -> bool:
    """Serialize segment append and anchor movement across workers."""
    async with db.async_session.begin() as session:
        conversation = await session.get(Conversation, conversation_id, with_for_update=True)
        if (conversation is None or (conversation.summary_upto_msg_id or 0) != expected_upto
                or from_msg_id <= expected_upto or upto_msg_id > (conversation.layer1_from_msg_id or 0)):
            return False
        last = await session.scalar(select(ConversationSummary).where(
            ConversationSummary.conversation_id == conversation_id,
        ).order_by(ConversationSummary.seq.desc()).limit(1))
        if last is not None and (last.upto_msg_id != expected_upto or from_msg_id <= last.upto_msg_id):
            return False
        endpoint = await session.get(Message, upto_msg_id)
        if (endpoint is None or endpoint.conversation_id != conversation_id
                or endpoint.role != "assistant" or endpoint.tool_calls is not None):
            return False
        session.add(ConversationSummary(conversation_id=conversation_id,
                                        seq=(last.seq + 1 if last else 1),
                                        from_msg_id=from_msg_id, upto_msg_id=upto_msg_id,
                                        content=content))
        conversation.summary_upto_msg_id = upto_msg_id
        conversation.summary = content[:200] if content else ""
        return True


async def get_context_snapshot(conversation_id: int, user_id: str) -> ContextSnapshot | None:
    """Read one user's visible history and append-only summary segments."""
    async with db.async_session.begin() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return None
        summary_upto = conversation.summary_upto_msg_id or 0
        messages = (await session.scalars(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.id > summary_upto,
                Message.role.in_(("user", "assistant")),
                Message.content.is_not(None),
            ).order_by(Message.id)
        )).all()
        summaries = (await session.scalars(
            select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id,
            ).order_by(ConversationSummary.seq)
        )).all()
        return ContextSnapshot(
            conversation_id=conversation_id,
            summary_upto_msg_id=summary_upto,
            layer1_from_msg_id=conversation.layer1_from_msg_id or 0,
            messages=tuple(VisibleMessage(m.id, m.role, m.content) for m in messages),
            summaries=tuple(SummarySegment(s.seq, s.from_msg_id, s.upto_msg_id, s.content) for s in summaries),
        )


async def advance_layer1(conversation_id: int, upto_msg_id: int) -> bool:
    """Move the recent-history boundary only through a complete visible turn."""
    async with db.async_session.begin() as session:
        conversation = await session.get(Conversation, conversation_id, with_for_update=True)
        if conversation is None or upto_msg_id <= max(
            conversation.layer1_from_msg_id or 0, conversation.summary_upto_msg_id or 0,
        ):
            return False
        target = await session.scalar(select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id == upto_msg_id,
            Message.role == "assistant",
            Message.content.is_not(None),
        ))
        if target is None or target.tool_calls is not None:
            return False
        prior = await session.scalar(select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id < upto_msg_id,
            Message.role.in_(("user", "assistant")),
        ).order_by(Message.id.desc()).limit(1))
        if prior is None or prior.role != "user":
            return False
        conversation.layer1_from_msg_id = upto_msg_id
        return True


async def append_turn_messages(
    conversation_id: int, query: str, tool_results: list[dict], answer: str,
) -> int:
    """Commit visible originals atomically; return final assistant audit marker."""
    async with db.async_session.begin() as session:
        rows = [Message(conversation_id=conversation_id, role="user", content=query)]
        final = Message(conversation_id=conversation_id, role="assistant", content=answer)
        rows.append(final)
        session.add_all(rows)
        await session.flush()
        await session.execute(update(Conversation).where(
            Conversation.id == conversation_id,
        ).values(updated_at=func.now()))
        return final.id


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


class TicketRequestConflict(ValueError):
    """An idempotency key has already been used for a different ticket request."""


class RefundRequestConflict(ValueError):
    """The request ID belongs to a different refund application payload."""


class OrderNotOwned(ValueError):
    """The sample order does not belong to this conversation's user."""


def _sample_order_dict(order: SampleOrder) -> dict:
    return {
        "order_id": order.order_id, "user_id": order.user_id,
        "status": order.status, "product": order.product,
        "amount": float(order.amount),
    }


async def list_sample_orders(user_id: str) -> list[dict]:
    async with db.async_session() as session:
        rows = (await session.scalars(
            select(SampleOrder).where(SampleOrder.user_id == user_id).order_by(SampleOrder.order_id)
        )).all()
    return [_sample_order_dict(row) for row in rows]


async def get_owned_sample_order(user_id: str, order_id: str) -> dict | None:
    async with db.async_session() as session:
        order = await session.get(SampleOrder, order_id)
    return _sample_order_dict(order) if order is not None and order.user_id == user_id else None


async def create_refund_request(
    conversation_id: int, user_id: str, order_id: str, reason: str, request_id: str,
) -> str:
    """Persist one pending application only for an owned sample order."""
    refund_no = f"R{uuid4().hex[:31]}"
    try:
        async with db.async_session.begin() as session:
            conversation = await session.get(Conversation, conversation_id)
            order = await session.get(SampleOrder, order_id)
            if conversation is None or conversation.user_id != user_id or order is None or order.user_id != user_id:
                raise OrderNotOwned("conversation or sample order not owned")
            session.add(RefundRequest(
                refund_no=refund_no, request_id=request_id, conversation_id=conversation_id,
                user_id=user_id, order_id=order_id, reason=reason, status="待人工审核",
            ))
            await session.flush()
        return refund_no
    except IntegrityError:
        async with db.async_session() as session:
            prior = await session.scalar(select(RefundRequest).where(RefundRequest.request_id == request_id))
        if prior is None:
            raise
        if (prior.conversation_id, prior.user_id, prior.order_id, prior.reason) != (
            conversation_id, user_id, order_id, reason,
        ):
            raise RefundRequestConflict("request_id already used for another refund payload") from None
        return prior.refund_no


async def create_ticket_only(
    conversation_id: int, description: str, ticket_type: str, request_id: str,
) -> str:
    """Create one ticket per request ID without changing handoff state."""
    ticket_no = _new_ticket_no()
    try:
        async with db.async_session.begin() as session:
            if await session.get(Conversation, conversation_id) is None:
                raise ValueError(f"conversation {conversation_id} does not exist")
            session.add(Ticket(
                ticket_no=ticket_no, conversation_id=conversation_id,
                description=description, ticket_type=ticket_type, request_id=request_id,
            ))
            await session.flush()
        return ticket_no
    except IntegrityError:
        # The failed transaction has ended before this lookup. The unique key
        # serializes concurrent retries, including across workers.
        async with db.async_session() as session:
            prior = await session.scalar(select(Ticket).where(Ticket.request_id == request_id))
        if prior is None:
            raise
        if (prior.conversation_id, prior.description, prior.ticket_type) != (
            conversation_id, description, ticket_type,
        ):
            raise TicketRequestConflict("request_id already used for another payload") from None
        return prior.ticket_no


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


_KNOWLEDGE_FIELDS = (
    "category", "questions", "answer", "section_path", "content_type", "is_key_clause",
)


def _knowledge_fingerprint(chunk: dict) -> tuple:
    """Treat formatting-only Unicode/spacing changes as the same source text."""
    return tuple(
        int(chunk[key]) if key == "is_key_clause" else
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", chunk[key] or "")).strip()
        for key in _KNOWLEDGE_FIELDS
    )


def _matching_document(rows: list[dict], chunks: list[dict]) -> list[int] | None:
    """Find an exact, already linked document; a changed document is a new version."""
    by_id = {row["id"]: row for row in rows}
    wanted = [_knowledge_fingerprint(chunk) for chunk in chunks]
    for first in rows:
        if first["prev_chunk_id"] is not None or _knowledge_fingerprint(first) != wanted[0]:
            continue
        ids: list[int] = []
        current = first
        for expected in wanted:
            if current is None or _knowledge_fingerprint(current) != expected:
                break
            ids.append(current["id"])
            current = by_id.get(current["next_chunk_id"])
        else:
            if current is None:
                return ids
    return None


async def _insert_document_row(connection, chunk: dict) -> int:
    result = await connection.execute(
        insert(KnowledgeChunk.__table__).values(
            **{field: chunk[field] for field in _KNOWLEDGE_FIELDS}
        )
    )
    return int(result.inserted_primary_key[0])


async def insert_knowledge_document(chunks: list[dict]) -> list[int]:
    ids, _inserted = await insert_knowledge_document_report(chunks)
    return ids


def manual_pair_key(questions: str, answer: str) -> tuple[str, str]:
    """The approved manual-ingest identity: normalized question and body."""
    def normalize(value: str) -> str:
        # Preserve internal punctuation such as the decimal point in 1.5.
        folded = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"\s+", "", folded).rstrip("。.!?！？；;")

    return normalize(questions), normalize(answer)


async def insert_manual_knowledge_report(chunks: list[dict]) -> tuple[list[int], int]:
    """Atomically insert only new manual question/body pairs.

    The named lock is shared with source-document ingestion and mining so a
    concurrent repeat cannot both pass the read-then-insert check. Existing
    rows are reused; only new rows are linked, avoiding mutation of old docs.
    """
    if not chunks:
        return [], 0
    engine = db.async_session.kw["bind"]
    name = _knowledge_lock_name(engine)
    async with engine.connect() as connection:
        acquired = await connection.scalar(
            text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": 10}
        )
        await connection.commit()
        if acquired != 1:
            raise TimeoutError("knowledge ingestion lock unavailable")
        try:
            async with connection.begin():
                table = KnowledgeChunk.__table__
                existing = (await connection.execute(select(
                    table.c.id, table.c.questions, table.c.answer
                ).order_by(table.c.id))).mappings()
                seen = {
                    manual_pair_key(row["questions"], row["answer"]): row["id"]
                    for row in existing
                }
                ids: list[int] = []
                new_ids: list[int] = []
                for chunk in chunks:
                    key = manual_pair_key(chunk["questions"], chunk["answer"])
                    chunk_id = seen.get(key)
                    if chunk_id is None:
                        chunk_id = await _insert_document_row(connection, chunk)
                        seen[key] = chunk_id
                        new_ids.append(chunk_id)
                    ids.append(chunk_id)
                for index, chunk_id in enumerate(new_ids):
                    await connection.execute(update(table).where(table.c.id == chunk_id).values(
                        prev_chunk_id=new_ids[index - 1] if index else None,
                        next_chunk_id=new_ids[index + 1] if index + 1 < len(new_ids) else None,
                    ))
                return ids, len(new_ids)
        finally:
            await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
            await connection.commit()


async def insert_knowledge_document_report(chunks: list[dict]) -> tuple[list[int], int]:
    """Atomically insert and link a document, or return an exact prior run's IDs.

    One MySQL named lock serializes the read/insert/commit across workers. A
    changed answer is never written over existing knowledge; it forms a new
    document version. The lock belongs to the same physical connection through
    commit, since MySQL named locks do not follow transaction boundaries.
    """
    if not chunks:
        return [], 0
    engine = db.async_session.kw["bind"]
    name = "mewhelp:kb:" + hashlib.sha256(
        (engine.url.database or "").encode("utf-8")
    ).hexdigest()[:32]
    async with engine.connect() as connection:
        acquired = await connection.scalar(
            text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": 10}
        )
        # GET_LOCK starts an implicit SQLAlchemy transaction, but the MySQL
        # named lock itself persists across this commit on the connection.
        await connection.commit()
        if acquired != 1:
            raise TimeoutError("knowledge ingestion lock unavailable")
        try:
            async with connection.begin():
                table = KnowledgeChunk.__table__
                result = await connection.execute(select(table).order_by(table.c.id))
                existing = list(result.mappings())
                prior = _matching_document(existing, chunks)
                if prior is not None:
                    return prior, 0
                ids = [await _insert_document_row(connection, chunk) for chunk in chunks]
                for index, chunk_id in enumerate(ids):
                    await connection.execute(
                        update(table).where(table.c.id == chunk_id).values(
                            prev_chunk_id=ids[index - 1] if index else None,
                            next_chunk_id=ids[index + 1] if index + 1 < len(ids) else None,
                        )
                    )
                return ids, len(ids)
        finally:
            await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
            await connection.commit()


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as session:
        result = await session.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def list_all_chunks() -> list[KnowledgeChunk]:
    """Read the MySQL authority set, including already-vectorized rows."""
    async with db.async_session() as session:
        result = await session.execute(select(KnowledgeChunk).order_by(KnowledgeChunk.id))
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session.begin() as session:
        row = await session.get(KnowledgeChunk, chunk_id)
        if row is None:
            raise LookupError(f"knowledge chunk {chunk_id} does not exist")
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
            .where(or_(QaExtractionStaging.question != "", QaExtractionStaging.answer != ""))
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


async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    """Return conversations in ID order, with their messages in ID order."""
    async with db.async_session() as session:
        ids = list((await session.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        if not ids:
            return []
        messages = list((await session.execute(
            select(Message).where(Message.conversation_id.in_(ids)).order_by(Message.id)
        )).scalars())
    by_conversation: dict[int, list[Message]] = {cid: [] for cid in ids}
    for message in messages:
        by_conversation[message.conversation_id].append(message)
    return [(cid, by_conversation[cid]) for cid in ids]


def _knowledge_lock_name(engine) -> str:
    return "mewhelp:kb:" + hashlib.sha256(
        (engine.url.database or "").encode("utf-8")
    ).hexdigest()[:32]


@asynccontextmanager
async def _named_knowledge_lock(suffix: str):
    engine = db.async_session.kw["bind"]
    name = _knowledge_lock_name(engine) + suffix
    async with engine.connect() as connection:
        acquired = await connection.scalar(
            text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": 10}
        )
        await connection.commit()
        if acquired != 1:
            raise TimeoutError("knowledge lock unavailable")
        try:
            yield
        finally:
            await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
            await connection.commit()


def knowledge_write_lock():
    """Serialize vector materialization with knowledge ingestion and reset."""
    return _named_knowledge_lock("")


def knowledge_lifecycle_lock():
    """Keep a whole ingest/mining operation outside the reset window."""
    return _named_knowledge_lock(":flow")


async def reset_knowledge_tables(drop_vectors) -> None:
    """Clear only Chapter 3 data, with the vector drop before MySQL deletion.

    If the vector drop fails, MySQL stays untouched. If the subsequent SQL
    commit fails, rerunning reset can finish the clear; missing vectors never
    expose deleted knowledge through retrieval in the meantime.
    """
    async with knowledge_lifecycle_lock():
        engine = db.async_session.kw["bind"]
        name = _knowledge_lock_name(engine)
        async with engine.connect() as connection:
            acquired = await connection.scalar(
                text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": 10}
            )
            await connection.commit()
            if acquired != 1:
                raise TimeoutError("knowledge reset lock unavailable")
            try:
                async with connection.begin():
                    drop_vectors()
                    await connection.execute(delete(QaExtractionStaging.__table__))
                    await connection.execute(delete(KnowledgeChunk.__table__))
            finally:
                await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
                await connection.commit()


async def insert_staging_batch(
    batch_no: str, source_ref: str, pairs: list[tuple[str, str]],
) -> int:
    """Persist one source's complete extraction once, even with concurrent jobs."""
    engine = db.async_session.kw["bind"]
    name = _knowledge_lock_name(engine)
    async with engine.connect() as connection:
        acquired = await connection.scalar(
            text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": 10}
        )
        await connection.commit()
        if acquired != 1:
            raise TimeoutError("knowledge ingestion lock unavailable")
        try:
            async with connection.begin():
                table = QaExtractionStaging.__table__
                prior = await connection.scalar(select(table.c.id).where(
                    table.c.batch_no == batch_no,
                    table.c.source_ref == source_ref,
                ).limit(1))
                if prior is not None:
                    return 0
                if pairs:
                    await connection.execute(insert(table), [
                        {"batch_no": batch_no, "source_ref": source_ref,
                         "question": question, "answer": answer}
                        for question, answer in pairs
                    ])
                else:
                    # The existing staging schema has no batch header row.
                    # A discarded empty row marks an evaluated no-pairs source
                    # without entering knowledge or extracted/kept statistics.
                    await connection.execute(insert(table).values(
                        batch_no=batch_no, source_ref=source_ref,
                        question="", answer="", status="discarded",
                    ))
                return len(pairs)
        finally:
            await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
            await connection.commit()


async def staging_batch_exists(batch_no: str, source_ref: str) -> bool:
    """Avoid another paid extraction for an unchanged, already staged source."""
    table = QaExtractionStaging.__table__
    async with db.async_session() as session:
        existing = await session.scalar(select(table.c.id).where(
            table.c.batch_no == batch_no,
            table.c.source_ref == source_ref,
        ).limit(1))
    return existing is not None


async def _insert_mined_chunk_row(connection, row: dict) -> int:
    result = await connection.execute(insert(KnowledgeChunk.__table__).values(
        category="历史对话", questions=row["question"], answer=row["answer"],
        section_path="mined", content_type="mined", is_key_clause=0,
    ))
    return int(result.inserted_primary_key[0])


async def finalize_mined_staging() -> dict[str, int]:
    """Atomically classify staged QA and insert pending knowledge.

    A legacy `kept` row with no matching mined chunk is recovered on rerun.
    The same named lock used by document ingestion prevents concurrent jobs
    from both accepting one question.
    """
    from app.kb.dedup import normalize_question

    engine = db.async_session.kw["bind"]
    name = _knowledge_lock_name(engine)
    stats = {"kept": 0, "discarded": 0, "recovered": 0}
    async with engine.connect() as connection:
        acquired = await connection.scalar(
            text("SELECT GET_LOCK(:name, :timeout)"), {"name": name, "timeout": 10}
        )
        await connection.commit()
        if acquired != 1:
            raise TimeoutError("knowledge ingestion lock unavailable")
        try:
            async with connection.begin():
                staging = QaExtractionStaging.__table__
                knowledge = KnowledgeChunk.__table__
                rows = list((await connection.execute(
                    select(staging).where(staging.c.status.in_(["extracted", "kept"]))
                    .order_by(staging.c.id)
                )).mappings())
                existing = list((await connection.execute(
                    select(knowledge.c.questions, knowledge.c.answer,
                           knowledge.c.category, knowledge.c.content_type)
                )).mappings())
                seen = {normalize_question(row["questions"]) for row in existing}
                mined_pairs = {
                    (row["questions"], row["answer"])
                    for row in existing
                    if row["category"] == "历史对话" and row["content_type"] == "mined"
                }
                # Recover old kept rows first; then classify fresh candidates.
                for row in sorted(rows, key=lambda item: (item["status"] != "kept", item["id"])):
                    pair = (row["question"], row["answer"])
                    key = normalize_question(row["question"])
                    if row["status"] == "kept":
                        if pair not in mined_pairs:
                            await _insert_mined_chunk_row(connection, row)
                            mined_pairs.add(pair)
                            seen.add(key)
                            stats["recovered"] += 1
                        continue
                    if not key or key in seen:
                        status = "discarded"
                        stats["discarded"] += 1
                    else:
                        await _insert_mined_chunk_row(connection, row)
                        mined_pairs.add(pair)
                        seen.add(key)
                        status = "kept"
                        stats["kept"] += 1
                    await connection.execute(
                        update(staging).where(staging.c.id == row["id"]).values(status=status)
                    )
        finally:
            await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
            await connection.commit()
    return stats


async def knowledge_stats() -> dict:
    """Summarize MySQL knowledge inventory for the KB and admin dashboards."""
    async with db.async_session() as session:
        total = int((await session.execute(
            select(func.count()).select_from(KnowledgeChunk)
        )).scalar_one())
        status_rows = (await session.execute(
            select(KnowledgeChunk.vectorize_status, func.count())
            .group_by(KnowledgeChunk.vectorize_status)
        )).all()
        key_clauses = int((await session.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.is_key_clause == 1)
        )).scalar_one())
        category_rows = (await session.execute(
            select(KnowledgeChunk.category, func.count())
            .group_by(KnowledgeChunk.category)
        )).all()
        content_type = func.coalesce(KnowledgeChunk.content_type, "unspecified")
        type_rows = (await session.execute(
            select(content_type, func.count()).group_by(content_type)
        )).all()
    statuses = dict(status_rows)
    return {
        "total": total,
        "pending": int(statuses.get("pending", 0)),
        "done": int(statuses.get("done", 0)),
        "key_clauses": key_clauses,
        "by_category": {name: int(count) for name, count in category_rows},
        "by_content_type": {name: int(count) for name, count in type_rows},
    }


async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    """Read the newest chunks with deterministic ID ordering."""
    if limit <= 0:
        return []
    async with db.async_session() as session:
        result = await session.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(min(limit, 100))
        )
        return list(result.scalars())


async def list_chunk_pairs() -> list[tuple[str, str]]:
    """Read exact question/answer pairs for preview duplicate markers."""
    async with db.async_session() as session:
        result = await session.execute(
            select(KnowledgeChunk.questions, KnowledgeChunk.answer)
            .order_by(KnowledgeChunk.id)
        )
        return [(question, answer) for question, answer in result.all()]


async def staging_stats() -> dict[str, int]:
    """Count extraction states and distinct mining batch numbers."""
    async with db.async_session() as session:
        state_rows = (await session.execute(
            select(QaExtractionStaging.status, func.count())
            .where(or_(QaExtractionStaging.question != "", QaExtractionStaging.answer != ""))
            .group_by(QaExtractionStaging.status)
        )).all()
        batches = int((await session.execute(
            select(func.count(func.distinct(QaExtractionStaging.batch_no)))
        )).scalar_one())
    states = dict(state_rows)
    return {
        "extracted": int(states.get("extracted", 0)),
        "kept": int(states.get("kept", 0)),
        "discarded": int(states.get("discarded", 0)),
        "batches": batches,
    }


async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None,
) -> int:
    """Record one low-confidence user question in the MySQL authority store."""
    if source not in {"retrieval_low_conf", "self_check", "user_feedback"}:
        raise ValueError("unsupported low-confidence source")
    if not raw_question.strip():
        raise ValueError("raw_question must not be blank")
    async with db.async_session.begin() as session:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        session.add(row)
        await session.flush()
        return row.id


FAITH_STATUSES = ("未解决", "已解决", "无需解决")


def _utc_naive() -> datetime:
    """MySQL DATETIME stores the chosen UTC instant without zone metadata."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _faith_case_dict(row: FaithCase) -> dict:
    return {
        "id": row.id, "eval_id": row.eval_id, "bucket": row.bucket,
        "query": row.query, "strategy": row.strategy,
        "answer": row.answer, "reason": row.reason,
        "citations": row.citations or [], "judge_model": row.judge_model,
        "status": row.status, "seen_count": row.seen_count,
        "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        "resolution": row.resolution,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
        "reopened": row.status == "未解决" and row.resolved_at is not None,
    }


async def upsert_faith_case(
    eval_id: str, *, bucket: str, query: str, answer: str, reason: str,
    strategy: str = "hybrid_rerank", citations: list[dict] | None = None,
    judge_model: str | None = None,
) -> tuple[int, bool]:
    """Keep one row per labeled question; recurrence reopens a reviewed case."""
    if not eval_id.strip() or not query.strip():
        raise ValueError("eval_id and query must not be blank")
    async with db.async_session.begin() as session:
        row = await session.scalar(
            select(FaithCase).where(FaithCase.eval_id == eval_id).with_for_update()
        )
        if row is None:
            row = FaithCase(
                eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                answer=answer, reason=reason, citations=citations,
                judge_model=judge_model, status="未解决", seen_count=1,
                first_seen_at=_utc_naive(), last_seen_at=_utc_naive(),
            )
            session.add(row)
            await session.flush()
            return row.id, False
        reopened = row.status != "未解决"
        row.bucket = bucket
        row.query = query
        row.strategy = strategy
        row.answer = answer
        row.reason = reason
        row.citations = citations
        row.judge_model = judge_model
        row.seen_count += 1
        row.last_seen_at = _utc_naive()
        if reopened:
            row.status = "未解决"
        return row.id, reopened


async def list_faith_cases(status: str | None = None, page: int = 1, size: int = 20) -> dict:
    if status is not None and status not in FAITH_STATUSES:
        raise ValueError("unsupported faith-case status")
    if page <= 0 or not 1 <= size <= 100:
        raise ValueError("invalid faith-case pagination")
    async with db.async_session() as session:
        status_rows = (await session.execute(
            select(FaithCase.status, func.count()).group_by(FaithCase.status)
        )).all()
        statement = select(FaithCase)
        if status is not None:
            statement = statement.where(FaithCase.status == status)
        total = int((await session.scalar(
            select(func.count()).select_from(statement.subquery())
        )) or 0)
        rows = (await session.execute(
            statement.order_by(
                case((FaithCase.status == "未解决", 0), else_=1),
                FaithCase.last_seen_at.desc(), FaithCase.id.desc(),
            ).offset((page - 1) * size).limit(size)
        )).scalars().all()
    counts = {key: 0 for key in FAITH_STATUSES}
    counts.update({key: int(value) for key, value in status_rows})
    return {"items": [_faith_case_dict(row) for row in rows], "total": total,
            "page": page, "size": size, "pages": (total + size - 1) // size,
            "counts": counts}


async def set_faith_case_status(
    case_id: int, status: str, resolution: str | None = None,
) -> dict | None:
    if status not in FAITH_STATUSES:
        raise ValueError("unsupported faith-case status")
    explanation = (resolution or "").strip()
    if status != "未解决" and not explanation:
        raise ValueError("resolution is required for reviewed cases")
    async with db.async_session.begin() as session:
        row = await session.get(FaithCase, case_id, with_for_update=True)
        if row is None:
            return None
        row.status = status
        row.resolution = explanation if status != "未解决" else None
        row.resolved_at = _utc_naive() if status != "未解决" else None
        await session.flush()
        return _faith_case_dict(row)


async def faith_case_status_map(eval_ids: list[str]) -> dict[str, str]:
    if not eval_ids:
        return {}
    async with db.async_session() as session:
        rows = (await session.execute(
            select(FaithCase.eval_id, FaithCase.status).where(FaithCase.eval_id.in_(eval_ids))
        )).all()
    return dict(rows)
