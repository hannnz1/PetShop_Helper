"""Minimal usage persistence; model outputs and prompts never enter this table."""

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
import hashlib
import json

from sqlalchemy import case, func, select
from sqlalchemy.dialects.mysql import insert

from app.db import base as db
from app.db.models import EvalRun, ModelUsageEvent
from app.observability.eval_trend import extract_summary
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


async def record_eval_run(run_id: str, dataset_hash: str, strategy: str, report: dict) -> None:
    """Insert once. A duplicate run ID never updates historical measurements."""
    if len(dataset_hash) != 64 or any(ch not in "0123456789abcdef" for ch in dataset_hash):
        raise ValueError("dataset_hash must be a lowercase SHA-256 hex digest")
    meta = report.get("meta") or {}
    summary = extract_summary(report, strategy)
    ended = datetime.fromisoformat(meta["evaluated_at"].replace("Z", "+00:00")).replace(tzinfo=None)
    started = datetime.fromisoformat(meta.get("started_at", meta["evaluated_at"]).replace("Z", "+00:00")).replace(tzinfo=None)
    statement = insert(EvalRun).values(
        run_id=run_id, dataset_hash=dataset_hash, strategy=strategy,
        git_sha=meta["git_sha"], model_name=meta.get("judge_model") or "unavailable",
        started_at=started, ended_at=ended, sample_count=summary["sample_count"],
        status=summary["status"], metrics=summary["metrics"],
        denominators=summary["denominators"], report_path=meta["report_path"],
        reason=meta.get("unavailable_reason") or summary["reason"],
    )
    async with db.async_session.begin() as session:
        await session.execute(statement.on_duplicate_key_update(run_id=statement.inserted.run_id))


async def comparable_trend(dataset_hash: str, strategy: str) -> list[dict]:
    """Return chronological runs, flagging changed metric denominators."""
    async with db.async_session() as session:
        rows = (await session.execute(
            select(EvalRun).where(EvalRun.dataset_hash == dataset_hash,
                                  EvalRun.strategy == strategy)
            .order_by(EvalRun.started_at, EvalRun.run_id)
        )).scalars().all()
    baseline = next((row.denominators for row in rows if row.status == "passed"), None)
    def comparison_group(row: EvalRun) -> str | None:
        if row.status != "passed" or not row.denominators:
            return None
        serialized = json.dumps(row.denominators, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return [
        {"run_id": row.run_id, "dataset_hash": row.dataset_hash,
         "strategy": row.strategy, "git_sha": row.git_sha,
         "model_name": row.model_name, "started_at": row.started_at.isoformat(),
         "ended_at": row.ended_at.isoformat(), "sample_count": row.sample_count,
         "status": row.status, "metrics": row.metrics,
         "denominators": row.denominators, "report_path": row.report_path,
         "reason": row.reason, "comparison_group": comparison_group(row),
         "comparable": row.status == "passed" and row.denominators == baseline}
        for row in rows
    ]
