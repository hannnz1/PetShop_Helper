"""Customer feedback intake and, later, protected review routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db.flywheel import record_unresolved_feedback


router = APIRouter()


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
