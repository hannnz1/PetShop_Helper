"""ORM mappings for customer-service and knowledge MySQL tables.

The DDL is applied independently; these classes never create or migrate tables.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text, text
from sqlalchemy.dialects.mysql import BIGINT, ENUM, INTEGER, TINYINT
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import FetchedValue

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_user_id", "user_id"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        ENUM("进行中", "已转人工", "已结束"), server_default=text("'进行中'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id", name="fk_messages_conversation"),
    )
    role: Mapped[str] = mapped_column(ENUM("user", "assistant", "tool"))
    content: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[list[dict[str, object]] | None] = mapped_column(JSON)
    tool_call_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )


class Faq(Base):
    __tablename__ = "faq"
    __table_args__ = (Index("idx_category", "category"),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
    )


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"),)

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id", name="fk_tickets_conversation"),
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(ENUM("售后", "投诉", "咨询"))
    status: Mapped[str] = mapped_column(
        ENUM("待处理", "已处理"), server_default=text("'待处理'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("idx_category", "category"),
        Index("idx_vectorize_status", "vectorize_status"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(32))
    is_key_clause: Mapped[int] = mapped_column(TINYINT(unsigned=False), server_default=text("0"))
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", name="fk_chunks_prev", ondelete="SET NULL"),
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", name="fk_chunks_next", ondelete="SET NULL"),
    )
    vector_id: Mapped[str | None] = mapped_column(String(64))
    vectorize_status: Mapped[str] = mapped_column(
        ENUM("pending", "done"), server_default=text("'pending'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"
    __table_args__ = (Index("idx_batch_no", "batch_no"), Index("idx_status", "status"))

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        ENUM("extracted", "kept", "discarded"), server_default=text("'extracted'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"
    __table_args__ = (
        Index("idx_low_confidence_conversation", "conversation_id"),
        Index("idx_low_confidence_source", "source"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id", name="fk_low_confidence_conversation", ondelete="SET NULL"),
    )
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        ENUM("retrieval_low_conf", "self_check", "user_feedback")
    )
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class FaithCase(Base):
    """One persistent row per evaluation question judged unfaithful."""

    __tablename__ = "faith_cases"
    __table_args__ = (
        Index("idx_faith_status", "status"),
        Index("idx_faith_last_seen_at", "last_seen_at"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), server_default=text("'hybrid_rerank'"))
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict] | None] = mapped_column(JSON)
    judge_model: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        ENUM("未解决", "已解决", "无需解决"), server_default=text("'未解决'")
    )
    seen_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), server_default=text("1"))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    resolution: Mapped[str | None] = mapped_column(String(300))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
