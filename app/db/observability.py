"""Minimal usage persistence; model outputs and prompts never enter this table."""

from collections.abc import Sequence

from sqlalchemy.dialects.mysql import insert

from app.db import base as db
from app.db.models import ModelUsageEvent
from app.observability.usage import UsageEvent


async def save_usage_events(events: Sequence[UsageEvent]) -> None:
    if not events:
        return
    async with db.async_session.begin() as session:
        for event in events:
            statement = insert(ModelUsageEvent).values(
                run_id=event.run_id, turn_id=event.turn_id,
                conversation_id=event.conversation_id, intent=event.intent,
                model_name=event.model_name, input_tokens=event.input_tokens,
                output_tokens=event.output_tokens, usage_status=event.usage_status,
                turn_status=event.turn_status, is_estimated=event.is_estimated,
            )
            await session.execute(statement.on_duplicate_key_update(run_id=statement.inserted.run_id))
