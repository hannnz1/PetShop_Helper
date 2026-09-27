"""Customer feedback intake and protected human review routes."""

import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.flywheel import record_unresolved_feedback
from app.flywheel.review import list_review_questions, review_question


router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


def require_review_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> None:
    configured = get_settings().knowledge_review_token
    if configured is None:
        raise HTTPException(status_code=503, detail="Review token not configured")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, configured.get_secret_value(),
    ):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


class UnresolvedFeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int = Field(gt=0)
    assistant_message_id: int = Field(gt=0)


@router.post("/api/feedback/unresolved")
async def unresolved_feedback(body: UnresolvedFeedbackRequest) -> dict[str, int]:
    try:
        row_id = await record_unresolved_feedback(
            body.user_id, body.conversation_id, body.assistant_message_id,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Answer not found") from None
    return {"id": row_id}


class ReviewDecisionRequest(BaseModel):
    action: Literal["reject", "defer", "merge", "approve"]
    request_id: str = Field(min_length=1, max_length=64)
    reason: str | None = None
    category: str | None = None
    approved_answer: str | None = None
    merge_target_id: int | None = Field(default=None, gt=0)


@router.get("/api/review/questions", dependencies=[Depends(require_review_token)])
async def review_queue(offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return await list_review_questions(offset=offset, limit=limit)


@router.post("/api/review/questions/{canonical_id}/decision", dependencies=[Depends(require_review_token)])
async def decide_question(canonical_id: int, body: ReviewDecisionRequest) -> dict:
    try:
        result = await review_question(canonical_id, body.action, body.request_id,
                                       reason=body.reason, category=body.category,
                                       approved_answer=body.approved_answer,
                                       merge_target_id=body.merge_target_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Canonical question not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"canonical_id": result.canonical_id, "status": result.status,
            "merged_into_id": result.merged_into_id}
