"""Transactional human review of canonical questions."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

import app.db.base as db
from app.db.models import CanonicalOccurrence, CanonicalQuestion, FlywheelReviewAction, KnowledgeChunk
from app.flywheel.privacy import mask_sensitive
from app.kb.documents import Chunk
from app.kb.dualwrite import write_manual_report


@dataclass(frozen=True)
class ReviewResult:
    canonical_id: int
    status: str
    merged_into_id: int | None = None


@dataclass(frozen=True)
class PublishResult:
    canonical_id: int
    knowledge_chunk_id: int
    status: str


async def _link_published_chunk(canonical_id: int, chunk_id: int, request_id: str) -> None:
    async with db.async_session.begin() as session:
        row = await session.scalar(select(CanonicalQuestion).where(
            CanonicalQuestion.id == canonical_id,
        ).with_for_update())
        if row is None or row.status not in {"approved", "approved_pending_vector"}:
            raise ValueError("canonical question is not approved")
        if row.knowledge_chunk_id is not None and row.knowledge_chunk_id != chunk_id:
            raise ValueError("published chunk identity changed")
        prior = await session.scalar(select(FlywheelReviewAction).where(
            FlywheelReviewAction.request_id == request_id,
        ))
        if prior is not None and (prior.canonical_id != canonical_id or prior.action != "publish"):
            raise ValueError("request_id conflicts with prior decision")
        row.knowledge_chunk_id = chunk_id
        row.status = "approved_pending_vector"


async def _reserve_publish(canonical_id: int, request_id: str) -> tuple[str, str, str, int | None]:
    """Reserve the idempotency key before any irreversible knowledge insert."""
    try:
        async with db.async_session.begin() as session:
            row = await session.scalar(select(CanonicalQuestion).where(
                CanonicalQuestion.id == canonical_id,
            ).with_for_update())
            if row is None:
                raise LookupError("canonical question not found")
            if row.status not in {"approved", "approved_pending_vector"} or not row.approved_answer or not row.category:
                raise ValueError("canonical question is not approved")
            prior = await session.scalar(select(FlywheelReviewAction).where(
                FlywheelReviewAction.request_id == request_id,
            ))
            if prior is not None and (prior.canonical_id != canonical_id or prior.action != "publish"):
                raise ValueError("request_id conflicts with prior decision")
            if prior is None:
                session.add(FlywheelReviewAction(
                    canonical_id=canonical_id, request_id=request_id, action="publish",
                ))
                await session.flush()
            return row.canonical_question, row.approved_answer, row.category, row.knowledge_chunk_id
    except IntegrityError:
        raise ValueError("request_id conflicts with prior decision") from None


async def publish_approved(canonical_id: int, request_id: str) -> PublishResult:
    """Publish approved human text to MySQL; vectorization runs separately."""
    if not request_id or len(request_id) > 64:
        raise ValueError("invalid request_id")
    question, answer, category, linked_id = await _reserve_publish(canonical_id, request_id)
    if linked_id is None:
        ids, _ = await write_manual_report([Chunk(
            category=category, questions=question, answer=answer,
            section_path=f"知识飞轮 / {category}", content_type="faq",
        )])
        linked_id = ids[0]
        await _link_published_chunk(canonical_id, linked_id, request_id)
    async with db.async_session() as session:
        chunk_status = await session.scalar(select(KnowledgeChunk.vectorize_status).where(
            KnowledgeChunk.id == linked_id,
        ))
    return PublishResult(canonical_id, linked_id,
                         "vectorized" if chunk_status == "done" else "approved_pending_vector")


async def list_review_questions(*, offset: int = 0, limit: int = 20) -> dict:
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("invalid pagination")
    async with db.async_session() as session:
        total = await session.scalar(select(func.count(CanonicalQuestion.id)))
        rows = (await session.scalars(select(CanonicalQuestion).order_by(
            CanonicalQuestion.id.desc(),
        ).offset(offset).limit(limit))).all()
        items = []
        for row in rows:
            count = await session.scalar(select(func.count(CanonicalOccurrence.id)).where(
                CanonicalOccurrence.canonical_id == row.id,
            ))
            items.append({
                "id": row.id, "canonical_question": mask_sensitive(row.canonical_question),
                "draft_answer": mask_sensitive(row.draft_answer or ""),
                "status": row.status, "category": row.category,
                "occurrence_count": count or 0, "merged_into_id": row.merged_into_id,
            })
        return {"total": total or 0, "items": items}


async def _review_question_transaction(
    canonical_id: int, action: str, request_id: str, *, reason: str | None = None,
    category: str | None = None, approved_answer: str | None = None,
    merge_target_id: int | None = None,
) -> ReviewResult:
    if action not in {"reject", "defer", "merge", "approve"}:
        raise ValueError("invalid action")
    if not request_id or len(request_id) > 64:
        raise ValueError("invalid request_id")
    reason = (reason or "").strip() or None
    category = (category or "").strip() or None
    approved_answer = (approved_answer or "").strip() or None
    if action in {"reject", "defer"} and not reason:
        raise ValueError("reason required")
    if action == "approve":
        if not category or not approved_answer:
            raise ValueError("category and human-approved answer required")
        if mask_sensitive(approved_answer) != approved_answer:
            raise ValueError("approved answer contains private identifier")
    if action == "merge" and (not merge_target_id or merge_target_id == canonical_id):
        raise ValueError("valid merge target required")
    audit_reason = f"merge:{merge_target_id}" if action == "merge" else reason

    async with db.async_session.begin() as session:
        row = await session.scalar(select(CanonicalQuestion).where(
            CanonicalQuestion.id == canonical_id,
        ).with_for_update())
        if row is None:
            raise LookupError("canonical question not found")
        existing = await session.scalar(select(FlywheelReviewAction).where(
            FlywheelReviewAction.request_id == request_id,
        ))
        if existing is not None:
            if (existing.canonical_id, existing.action, existing.reason,
                existing.category, existing.approved_answer) != (
                canonical_id, action, audit_reason, category, approved_answer,
            ):
                raise ValueError("request_id conflicts with prior decision")
            return ReviewResult(row.id, row.status, row.merged_into_id)
        if row.status not in {"pending_review", "deferred"}:
            raise ValueError("invalid state transition")
        if action == "merge":
            target = await session.scalar(select(CanonicalQuestion).where(
                CanonicalQuestion.id == merge_target_id,
            ).with_for_update())
            if target is None or target.status in {"rejected", "approved", "approved_pending_vector"}:
                raise ValueError("invalid merge target")
            occurrences = (await session.scalars(select(CanonicalOccurrence).where(
                CanonicalOccurrence.canonical_id == canonical_id,
            ))).all()
            for occurrence in occurrences:
                occurrence.canonical_id = target.id
            row.merged_into_id = target.id
            row.status = "rejected"
        elif action == "approve":
            row.approved_answer = approved_answer
            row.category = category
            row.status = "approved"
        else:
            row.status = "rejected" if action == "reject" else "deferred"
        session.add(FlywheelReviewAction(
            canonical_id=canonical_id, request_id=request_id, action=action,
            reason=audit_reason, category=category, approved_answer=approved_answer,
        ))
        return ReviewResult(row.id, row.status, row.merged_into_id)


async def review_question(
    canonical_id: int, action: str, request_id: str, *, reason: str | None = None,
    category: str | None = None, approved_answer: str | None = None,
    merge_target_id: int | None = None,
) -> ReviewResult:
    try:
        return await _review_question_transaction(
            canonical_id, action, request_id, reason=reason, category=category,
            approved_answer=approved_answer, merge_target_id=merge_target_id,
        )
    except IntegrityError:
        # Different canonical rows can race on the same global request ID.
        raise ValueError("request_id conflicts with prior decision") from None
