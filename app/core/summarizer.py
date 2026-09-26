"""Asynchronous, append-only compaction of completed Layer2 turns."""

import asyncio
from dataclasses import dataclass
import logging
from time import monotonic
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.config import get_settings
from app.core.memory import estimate_tokens
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository

# Chapter 7's context log is owned by this logger in graph.nodes.
logger = logging.getLogger("app.graph.nodes")


@dataclass(frozen=True)
class SummaryOutcome:
    status: Literal["skipped", "running", "committed", "failed"]
    from_msg_id: int | None = None
    upto_msg_id: int | None = None


_model: BaseChatModel | None = None
_tasks: set[asyncio.Task] = set()


def configure_summary_model(model: BaseChatModel) -> None:
    global _model
    _model = model


async def schedule_recovery() -> None:
    """Revisit persisted gaps; each task applies the same token gate and CAS."""
    try:
        for conversation_id in await repository.list_summary_candidates():
            schedule_summary(conversation_id)
    except Exception:
        logger.exception("summary fail reason=recovery_scan")


def schedule_summary(conversation_id: int) -> None:
    """Enqueue after response completion; no model call runs in the SSE frame."""
    if _model is None:
        logger.warning("summary fail conversation_id=%s reason=no_model", conversation_id)
        return
    logger.info("summary trigger conversation_id=%s", conversation_id)
    task = asyncio.create_task(summarize_pending(conversation_id, _model))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def close_summary_tasks(timeout: float = 5.0) -> None:
    global _model
    if _tasks:
        done, pending = await asyncio.wait(tuple(_tasks), timeout=timeout)
        for task in done:
            if not task.cancelled():
                task.exception()
        for task in pending:
            logger.warning("summary fail reason=shutdown_timeout task=%s", task.get_name())
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=1.0)
    _tasks.clear()
    _model = None


def _completed_batch(rows):
    pairs = []
    pending = None
    for row in rows:
        if row.role == "user":
            pending = row
        elif row.role == "assistant" and pending is not None:
            pairs.append((pending, row))
            pending = None
    return pairs


def _facts(text: str) -> str:
    value = text.strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    if value.rstrip('。.!！ ') in {'', '无明确事实', '无事实', '无', 'NONE', 'none', 'null'}:
        return ''
    return value[:200]


async def summarize_pending(conversation_id: int, model: BaseChatModel,
                            *, layer2_token_limit: int | None = None) -> SummaryOutcome:
    started = monotonic()
    work = await repository.get_summary_work(conversation_id)
    if work is None:
        logger.info("summary skip conversation_id=%s reason=missing", conversation_id)
        return SummaryOutcome("skipped")
    pairs = _completed_batch(work.messages)
    if not pairs:
        logger.info("summary skip conversation_id=%s reason=no_completed_layer2", conversation_id)
        return SummaryOutcome("skipped")
    settings = get_settings()
    limit = settings.layer2_summary_tokens if layer2_token_limit is None else layer2_token_limit
    compact = []
    for user, assistant in pairs:
        compact.extend((HumanMessage(content=user.content),
                        AIMessage(content=assistant.content[:settings.layer2_assistant_chars])))
    used = estimate_tokens(compact, chars_per_token=settings.cjk_chars_per_token)
    if used <= limit:
        logger.info("summary skip conversation_id=%s layer2_tokens=%s limit=%s", conversation_id, used, limit)
        return SummaryOutcome("skipped")
    start_id, end_id = pairs[0][0].id, pairs[-1][1].id
    batch = "\n".join(f"用户：{user.content}\n客服：{assistant.content}"
                      for user, assistant in pairs)
    logger.info("summary start conversation_id=%s from=%s upto=%s layer2_tokens=%s",
                conversation_id, start_id, end_id, used)
    try:
        response = await model.ainvoke(SUMMARY_PROMPT.format_messages(batch=batch))
        content = _facts(response.content if isinstance(response.content, str) else '')
        committed = await repository.commit_summary_segment(
            conversation_id, work.summary_upto_msg_id, start_id, end_id, content,
        )
    except Exception:
        logger.exception("summary fail conversation_id=%s from=%s upto=%s elapsed_ms=%d",
                         conversation_id, start_id, end_id, int((monotonic() - started) * 1000))
        return SummaryOutcome("failed", start_id, end_id)
    if not committed:
        logger.info("summary skip conversation_id=%s reason=stale_anchor from=%s upto=%s",
                    conversation_id, start_id, end_id)
        return SummaryOutcome("skipped", start_id, end_id)
    status = "committed" if content else "skipped"
    logger.info("summary %s conversation_id=%s from=%s upto=%s elapsed_ms=%d",
                "done" if content else "skip", conversation_id, start_id, end_id,
                int((monotonic() - started) * 1000))
    return SummaryOutcome(status, start_id, end_id)
