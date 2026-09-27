"""Minimal usage persistence; model outputs and prompts never enter this table."""

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta

from sqlalchemy import case, func, select
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


async def usage_by_day(start: date, end: date) -> list[dict]:
    """Aggregate measured usage by calendar day, final intent and model."""
    if start > end:
        raise ValueError("start must be on or before end")

    day = func.date(ModelUsageEvent.created_at).label("day")
    available = ModelUsageEvent.usage_status == "available"
    available_calls = func.sum(case((available, 1), else_=0)).label("available_calls")
    statement = (
        select(
            day, ModelUsageEvent.intent, ModelUsageEvent.model_name,
            func.count().label("calls"), available_calls,
            func.sum(case((available, 0), else_=1)).label("unavailable_calls"),
            func.sum(ModelUsageEvent.input_tokens).label("input_tokens"),
            func.sum(ModelUsageEvent.output_tokens).label("output_tokens"),
        )
        .where(ModelUsageEvent.created_at >= datetime.combine(start, time.min))
        .where(ModelUsageEvent.is_estimated.is_(False))
        .group_by(day, ModelUsageEvent.intent, ModelUsageEvent.model_name)
        .order_by(day, ModelUsageEvent.intent, ModelUsageEvent.model_name)
    )
    if end < date.max:
        statement = statement.where(
            ModelUsageEvent.created_at < datetime.combine(end + timedelta(days=1), time.min)
        )
    async with db.async_session() as session:
        rows = (await session.execute(statement)).mappings().all()
    return [
        {"day": str(row["day"]), "intent": row["intent"],
         "model_name": row["model_name"], "calls": int(row["calls"]),
         "available_calls": int(row["available_calls"]),
         "unavailable_calls": int(row["unavailable_calls"]),
         "input_tokens": int(row["input_tokens"]) if row["input_tokens"] is not None else None,
         "output_tokens": int(row["output_tokens"]) if row["output_tokens"] is not None else None}
        for row in rows
    ]
