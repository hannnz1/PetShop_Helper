"""ORM mappings for customer-service and knowledge MySQL tables.

The DDL is applied independently; these classes never create or migrate tables.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, JSON, Numeric, String, Text, text, Float
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
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)


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


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (Index("uq_conversation_summaries_seq", "conversation_id", "seq", unique=True),)

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_conversation_summaries_conversation")
    )
    seq: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    from_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    upto_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


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
    __table_args__ = (
        Index("idx_conversation_id", "conversation_id"),
        Index("uq_tickets_request_id", "request_id", unique=True),
    )

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
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


class SampleOrder(Base):
    """Opt-in demonstration order; never inferred from a random order number."""

    __tablename__ = "sample_orders"
    __table_args__ = (Index("idx_sample_orders_user", "user_id"),)

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(64))
    product: Mapped[str] = mapped_column(String(255))
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class RefundRequest(Base):
    """Confirmed application pending human review, not an executed refund."""

    __tablename__ = "refund_requests"
    __table_args__ = (
        Index("uq_refund_requests_request_id", "request_id", unique=True),
        Index("idx_refund_requests_conversation", "conversation_id"),
        Index("idx_refund_requests_order", "order_id"),
    )

    refund_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("conversations.id"))
    user_id: Mapped[str] = mapped_column(String(64))
    order_id: Mapped[str] = mapped_column(String(64), ForeignKey("sample_orders.order_id"))
    reason: Mapped[str] = mapped_column(ENUM("七天无理由", "质量问题", "发错货", "不想要了", "其他"))
    status: Mapped[str] = mapped_column(ENUM("待人工审核"), server_default=text("'待人工审核'"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class ToolAuditLog(Base):
    """Independent record of every attempted tool call, including denials."""

    __tablename__ = "tool_audit_logs"
    __table_args__ = (
        Index("idx_conversation_id", "conversation_id"),
        Index("idx_tool_audit_turn", "conversation_id", "turn_id"),
        Index("idx_tool_name", "tool_name"),
        Index("idx_status", "status"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    turn_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(ENUM("builtin", "mcp"))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(ENUM("成功", "失败", "超时", "校验拦下", "权限拒绝"))
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(TINYINT(unsigned=True), server_default=text("0"))
    duration_ms: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class ModelUsageEvent(Base):
    """One actual or unavailable usage observation per LangChain model run."""

    __tablename__ = "model_usage_events"
    __table_args__ = (Index("idx_usage_turn", "conversation_id", "turn_id"),
                      Index("idx_usage_daily", "created_at", "intent", "model_name"))

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    turn_id: Mapped[str] = mapped_column(String(36))
    intent: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    input_tokens: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    usage_status: Mapped[str] = mapped_column(ENUM("available", "unavailable"))
    turn_status: Mapped[str] = mapped_column(ENUM("completed", "failed", "interrupted"))
    is_estimated: Mapped[bool] = mapped_column(TINYINT(unsigned=False), server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class EvalRun(Base):
    """Immutable summary of one Chapter 4 evaluation run and strategy."""

    __tablename__ = "eval_runs"
    __table_args__ = (Index("idx_eval_comparable", "dataset_hash", "strategy", "started_at"),)

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(32))
    git_sha: Mapped[str] = mapped_column(String(40))
    model_name: Mapped[str] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime] = mapped_column(DateTime)
    sample_count: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    status: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict] = mapped_column(JSON)
    denominators: Mapped[dict] = mapped_column(JSON)
    report_path: Mapped[str] = mapped_column(String(512))
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)


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
        Index("uq_low_confidence_source_ref", "source_ref", unique=True),
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
    source_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class CanonicalQuestion(Base):
    __tablename__ = "canonical_questions"
    __table_args__ = (
        Index("idx_canonical_status", "status"),
        Index("idx_canonical_knowledge_chunk", "knowledge_chunk_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    canonical_question: Mapped[str] = mapped_column(String(512))
    canonical_key: Mapped[str | None] = mapped_column(String(64), unique=True)
    draft_answer: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        ENUM("pending_review", "deferred", "rejected", "approved", "approved_pending_vector"),
        server_default=text("'pending_review'"),
    )
    approved_answer: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))
    merged_into_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("canonical_questions.id", name="fk_canonical_merged_into", ondelete="SET NULL"),
    )
    knowledge_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id", name="fk_canonical_knowledge_chunk", ondelete="SET NULL"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
    )


class CanonicalOccurrence(Base):
    __tablename__ = "canonical_occurrences"
    __table_args__ = (
        Index("uq_canonical_occurrence_raw", "raw_question_id", unique=True),
        Index("idx_canonical_occurrences_canonical", "canonical_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    canonical_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("canonical_questions.id", name="fk_canonical_occurrence_canonical"),
    )
    raw_question_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("low_confidence_questions.id", name="fk_canonical_occurrence_raw"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


class FlywheelReviewAction(Base):
    __tablename__ = "flywheel_review_actions"
    __table_args__ = (
        Index("uq_flywheel_review_request", "request_id", unique=True),
        Index("idx_flywheel_review_canonical", "canonical_id"),
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    canonical_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("canonical_questions.id", name="fk_flywheel_review_canonical"),
    )
    request_id: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(ENUM("reject", "defer", "merge", "approve", "publish"))
    reason: Mapped[str | None] = mapped_column(Text)
    approved_answer: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))
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


class TopicClassification(Base):
    __tablename__ = 'topic_classifications'
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey('low_confidence_questions.id'), unique=True)
    labels: Mapped[list] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(64))
    taxonomy_hash: Mapped[str] = mapped_column(String(64))
    threshold: Mapped[float] = mapped_column(Float)
    input_hash: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(String(64))
    classified_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'))


class TopicClassificationRun(Base):
    __tablename__ = 'topic_classification_runs'
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_version: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    pending_count: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    success_count: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    failed_count: Mapped[int] = mapped_column(INTEGER(unsigned=True))
    status: Mapped[str] = mapped_column(String(24))
    report_path: Mapped[str | None] = mapped_column(String(255))
