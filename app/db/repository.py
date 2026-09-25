"""Asynchronous persistence for customer-service conversations and tool data."""

from uuid import uuid4
import hashlib
import re
import unicodedata

from sqlalchemy import func, insert, select, text, update

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
