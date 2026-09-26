"""Read-only, owner-scoped conversation index and original history."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.db import repository


router = APIRouter(prefix="/api/conversations")
PageLimit = Annotated[int, Query(ge=1, le=50)]
UserId = Annotated[str, Query(min_length=1)]


def _decode_before(value: str | None) -> tuple[datetime, int] | None:
    if value is None:
        return None
    try:
        stamp, raw_id = value.rsplit("|", 1)
        when = datetime.fromisoformat(stamp)
        identifier = int(raw_id)
        if when.tzinfo is not None or identifier <= 0:
            raise ValueError
        return when, identifier
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid conversation cursor") from None


@router.get("")
async def list_conversations(
    user_id: UserId, limit: PageLimit = 20, before: str | None = None,
) -> dict:
    items, cursor = await repository.list_owned_conversations(user_id, limit, _decode_before(before))
    return {"items": items, "next_cursor": f"{cursor[0].isoformat()}|{cursor[1]}" if cursor else None}


@router.get("/{conversation_id}/messages")
async def list_conversation_messages(
    conversation_id: int, user_id: UserId, limit: PageLimit = 50,
    after: Annotated[int | None, Query(ge=0)] = None,
) -> dict:
    page = await repository.list_owned_visible_messages(conversation_id, user_id, limit, after)
    if page is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages, cursor = page
    return {"items": [
        {"id": message.id, "role": message.role, "content": message.content}
        for message in messages
    ], "next_cursor": cursor}
