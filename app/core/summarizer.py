"""Asynchronous, append-only compaction of completed Layer2 turns."""

import asyncio
from dataclasses import dataclass
import logging
from time import monotonic
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings, get_settings
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
_settings: Settings | None = None
_tasks: set[asyncio.Task] = set()


def configure_summary_model(model: BaseChatModel, settings: Settings | None = None) -> None:
    global _model, _settings
    _model = model
    _settings = settings


async def schedule_recovery(graph_runtime=None) -> None:
    """Revisit persisted gaps; each task applies the same token gate and CAS."""
    try:
        for conversation_id in await repository.list_summary_candidates():
            state = None
            if graph_runtime is not None and graph_runtime.graph is not None:
                checkpoint = await graph_runtime.graph.aget_state(
                    {"configurable": {"thread_id": str(conversation_id)}})
                state = checkpoint.values or None
            schedule_summary(conversation_id, route_state=state)
    except Exception:
        logger.exception("summary fail reason=recovery_scan")


def schedule_summary(conversation_id: int, *, layer2_token_limit: int | None = None,
                     route_state: dict | None = None) -> None:
    """Enqueue after response completion; no model call runs in the SSE frame."""
    if _model is None:
        logger.warning("summary fail conversation_id=%s reason=no_model", conversation_id)
        return
    logger.info("summary trigger conversation_id=%s", conversation_id)
    task = asyncio.create_task(summarize_pending(
        conversation_id, _model, layer2_token_limit=layer2_token_limit,
        settings=_settings, route_state=route_state,
    ))
    _tasks.add(task)
    def finished(completed: asyncio.Task) -> None:
        _tasks.discard(completed)
        try:
            completed.result()
        except asyncio.CancelledError:
            logger.warning("summary fail conversation_id=%s reason=cancelled", conversation_id)
        except Exception:
            logger.exception("summary fail conversation_id=%s reason=unhandled_worker", conversation_id)
    task.add_done_callback(finished)


async def close_summary_tasks(timeout: float = 5.0) -> None:
    global _model, _settings
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
    _settings = None


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


async def _derived_limit(conversation_id: int, settings: Settings,
                         route_state: dict | None) -> int:
    """Recompute a route budget after restart; unknown routes use the safe minimum."""
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise LookupError(f"conversation {conversation_id} is missing")
    snapshot = await repository.get_context_snapshot(conversation_id, conversation.user_id)
    if snapshot is None:
        raise LookupError(f"context for conversation {conversation_id} is missing")
    from app.graph.nodes import _budget
    if route_state and route_state.get("route"):
        return _budget(route_state, snapshot, settings=settings).layer2
    return min(_budget({"route": route}, snapshot, settings=settings).layer2
               for route in ("business", "knowledge", "refund", "chitchat"))


def _batch_prefix(pairs, settings: Settings):
    window = min(settings.model_context_window,
                 settings.token_budget if "token_budget" in settings.model_fields_set
                 else settings.model_context_window)
    capacity = window - max(settings.max_output_tokens, settings.chat_max_tokens) - 250
    selected = []
    for pair in pairs:
        candidate = [*selected, pair]
        batch = "\n".join(f"用户：{user.content}\n客服：{assistant.content}"
                          for user, assistant in candidate)
        prompt = SUMMARY_PROMPT.format_messages(batch=batch)
        if estimate_tokens(prompt, chars_per_token=settings.cjk_chars_per_token) > capacity:
            break
        selected = candidate
    if not selected:
        raise ValueError("oldest completed turn exceeds summary input budget")
    return selected


async def summarize_pending(conversation_id: int, model: BaseChatModel,
                            *, layer2_token_limit: int | None = None,
                            settings: Settings | None = None,
                            route_state: dict | None = None) -> SummaryOutcome:
    started = monotonic()
    start_id = end_id = None
    try:
        work = await repository.get_summary_work(conversation_id)
        if work is None:
            logger.info("summary skip conversation_id=%s reason=missing", conversation_id)
            return SummaryOutcome("skipped")
        pairs = _completed_batch(work.messages)
        if not pairs:
            logger.info("summary skip conversation_id=%s reason=no_completed_layer2", conversation_id)
            return SummaryOutcome("skipped")
        settings = settings or get_settings()
        limit = (layer2_token_limit if layer2_token_limit is not None else
                 await _derived_limit(conversation_id, settings, route_state))
        compact = []
        for user, assistant in pairs:
            compact.extend((HumanMessage(content=user.content),
                            AIMessage(content=assistant.content[:settings.layer2_assistant_chars])))
        used = estimate_tokens(compact, chars_per_token=settings.cjk_chars_per_token)
        if used <= limit:
            logger.info("summary skip conversation_id=%s layer2_tokens=%s limit=%s", conversation_id, used, limit)
            return SummaryOutcome("skipped")
        selected = _batch_prefix(pairs, settings)
        start_id, end_id = selected[0][0].id, selected[-1][1].id
        batch = "\n".join(f"用户：{user.content}\n客服：{assistant.content}"
                          for user, assistant in selected)
        logger.info("summary start conversation_id=%s from=%s upto=%s layer2_tokens=%s limit=%s",
                    conversation_id, start_id, end_id, used, limit)
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
