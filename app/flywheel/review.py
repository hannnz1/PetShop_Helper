"""Transactional human review of canonical questions."""

from dataclasses import dataclass

from sqlalchemy import func, select

import app.db.base as db
from app.db.models import CanonicalOccurrence, CanonicalQuestion, FlywheelReviewAction
from app.flywheel.privacy import mask_sensitive


@dataclass(frozen=True)
class ReviewResult:
    canonical_id: int
    status: str
    merged_into_id: int | None = None


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


async def review_question(
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
        existing = await session.scalar(select(FlywheelReviewAction).where(
            FlywheelReviewAction.request_id == request_id,
        ))
        if existing is not None:
            if (existing.canonical_id, existing.action, existing.reason,
                existing.category, existing.approved_answer) != (
                canonical_id, action, audit_reason, category, approved_answer,
            ):
                raise ValueError("request_id conflicts with prior decision")
            row = await session.get(CanonicalQuestion, canonical_id)
            if row is None:
                raise LookupError("canonical question not found")
            return ReviewResult(row.id, row.status, row.merged_into_id)

        row = await session.scalar(select(CanonicalQuestion).where(
            CanonicalQuestion.id == canonical_id,
        ).with_for_update())
        if row is None:
            raise LookupError("canonical question not found")
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
