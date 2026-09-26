"""One tool-planning round with persisted conversations and two answer exits."""

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.memory import estimate_tokens
from app.core.prompts import AGENT_SYSTEM
from app.db import repository
from app.db.models import Message
from app.tools.infra import ToolRun, execute_tool_call
from app.tools.registry import get_all_tools


class ConversationNotFound(Exception):
    """The requested conversation is missing or belongs to another user."""


class ContextBudgetExceeded(ValueError):
    """The current tool exchange cannot fit without losing required results."""


@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]
    citations: list[dict] = field(default_factory=list)


FAQ_REFUSAL = "暂时没有查到足够的知识库信息来回答这个问题，建议您核对平台正式规则或联系官方客服。"


def _faq_result(runs: list[ToolRun]) -> dict | None:
    """Recognize only successful, new-format RAG FAQ tool results."""

    for run in runs:
        if run.name != "query_faq" or not run.ok:
            continue
        try:
            payload = json.loads(run.tool_message.content)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and type(payload.get("sufficient")) is bool:
            return payload
    return None


def _model_tool_message(run: ToolRun) -> ToolMessage:
    """Keep FAQ citations in the SSE payload, but avoid duplicating evidence in the model context."""

    if run.name == "query_faq" and run.ok:
        try:
            payload = json.loads(run.tool_message.content)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("sufficient") is True and isinstance(payload.get("evidence"), str):
            return ToolMessage(
                content=json.dumps({"sufficient": True, "evidence": payload["evidence"]}, ensure_ascii=False),
                tool_call_id=run.tool_call_id,
                name=run.name,
            )
    return run.tool_message


async def _record_faq_refusal(conversation_id: int, message: str, faq: dict) -> None:
    source = faq.get("source")
    if source not in {"retrieval_low_conf", "self_check"}:
        source = "self_check"
    reason = faq.get("reason")
    await repository.insert_low_confidence(
        conversation_id, message, source, reason if isinstance(reason, str) else None,
    )


@dataclass
class _PreparedTurn:
    conversation_id: int
    messages: list[BaseMessage]
    planned: AIMessage


def _text(message: AIMessage) -> str:
    """Extract only user-facing text from string or text-block content."""

    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _complete_turns(rows: list[Message]) -> list[tuple[HumanMessage, AIMessage]]:
    """Only finalized user/assistant pairs survive across requests.

    Tool-calling assistant rows are audit entries; the tool result and any
    orphaned/incomplete rows are deliberately excluded from future prompts.
    """

    pairs: list[tuple[HumanMessage, AIMessage]] = []
    pending_user: HumanMessage | None = None
    for row in rows:
        if row.role == "user":
            pending_user = HumanMessage(content=row.content or "")
        elif row.role == "assistant" and not row.tool_calls and pending_user is not None:
            if row.content:
                pairs.append((pending_user, AIMessage(content=row.content)))
            pending_user = None
    return pairs


def _build_messages(rows: list[Message], current: str, max_tokens: int) -> list[BaseMessage]:
    """Reserve system/current input and retain newest complete old turns."""

    system = SystemMessage(content=AGENT_SYSTEM)
    human = HumanMessage(content=current)
    required = estimate_tokens([system, human])
    if required > max_tokens:
        raise ContextBudgetExceeded("system and current message exceed context budget")
    kept: list[BaseMessage] = []
    for user, assistant in reversed(_complete_turns(rows)):
        candidate = [user, assistant, *kept]
        if estimate_tokens([system, *candidate, human]) > max_tokens:
            break
        kept = candidate
    return [system, *kept, human]


async def _prepare_turn(
    user_id: str, message: str, conversation_id: int | None, model
) -> _PreparedTurn:
    """Authorize, persist input, and ask a tool-bound model to plan once."""

    if conversation_id is None:
        conversation_id = await repository.create_conversation(user_id)
    else:
        conversation = await repository.get_conversation(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise ConversationNotFound(conversation_id)

    # Exclude the just-arrived user row from historical pairing. It remains
    # persisted even if planning fails, but never becomes a completed turn.
    rows = await repository.list_messages(conversation_id)
    messages = _build_messages(rows, message, get_settings().token_budget)
    await repository.append_message(conversation_id, "user", content=message)
    planned: AIMessage = await model.bind_tools(get_all_tools()).ainvoke(messages)
    await repository.append_message(
        conversation_id,
        "assistant",
        content=_text(planned) or None,
        tool_calls=planned.tool_calls or None,
    )
    return _PreparedTurn(conversation_id, messages, planned)


async def _run_tools(turn: _PreparedTurn) -> list[ToolRun]:
    """Execute in model order so a failure cannot leave a late write running."""

    runs: list[ToolRun] = []
    for call in turn.planned.tool_calls:
        run = await execute_tool_call(call, turn.conversation_id)
        await repository.append_message(
            turn.conversation_id,
            "tool",
            content=str(run.tool_message.content),
            tool_call_id=run.tool_call_id,
        )
        runs.append(run)
    return runs


def _convergence_messages(turn: _PreparedTurn, runs: list[ToolRun], max_tokens: int) -> list[BaseMessage]:
    """Fit final-model input by removing only old complete conversation pairs."""

    # _build_messages guarantees system, zero or more complete pairs, current.
    system = turn.messages[0]
    current = turn.messages[-1]
    tool_messages = [_model_tool_message(run) for run in runs]
    required = [system, current, turn.planned, *tool_messages]
    if estimate_tokens(required) > max_tokens:
        raise ContextBudgetExceeded("current tool exchange exceeds context token budget")

    old = list(turn.messages[1:-1])
    while old and estimate_tokens([system, *old, current, turn.planned, *tool_messages]) > max_tokens:
        old = old[2:]
    return [system, *old, current, turn.planned, *tool_messages]


async def run_agent_turn(
    user_id: str, message: str, conversation_id: int | None, model=None
) -> AgentResult:
    """Produce a final JSON-ready answer, with at most one round of tools."""

    model = model if model is not None else get_chat_model()
    turn = await _prepare_turn(user_id, message, conversation_id, model)
    if not turn.planned.tool_calls:
        return AgentResult(turn.conversation_id, _text(turn.planned), [], [])

    runs = await _run_tools(turn)
    faq = _faq_result(runs)
    if faq is not None and not faq["sufficient"]:
        await _record_faq_refusal(turn.conversation_id, message, faq)
        await repository.append_message(turn.conversation_id, "assistant", content=FAQ_REFUSAL)
        return AgentResult(turn.conversation_id, FAQ_REFUSAL, turn.planned.tool_calls, runs)
    convergence = _convergence_messages(turn, runs, get_settings().token_budget)
    final: AIMessage = await model.ainvoke(convergence)
    answer = _text(final)
    await repository.append_message(turn.conversation_id, "assistant", content=answer)
    citations = faq.get("citations", []) if faq is not None else []
    return AgentResult(turn.conversation_id, answer, turn.planned.tool_calls, runs, citations)


async def stream_agent_turn(
    user_id: str, message: str, conversation_id: int | None, model=None
) -> AsyncIterator[dict]:
    """Emit selected tools before running them, then stream the final answer."""

    model = model if model is not None else get_chat_model(streaming=True)
    turn = await _prepare_turn(user_id, message, conversation_id, model)
    if not turn.planned.tool_calls:
        answer = _text(turn.planned)
        if answer:
            yield {"type": "delta", "text": answer}
        yield {"type": "done", "conversation_id": turn.conversation_id}
        return

    # These events indicate selection/ongoing execution, never success.
    for call in turn.planned.tool_calls:
        yield {"type": "tool", "name": call.get("name") or ""}

    runs = await _run_tools(turn)
    faq = _faq_result(runs)
    if faq is not None and not faq["sufficient"]:
        await _record_faq_refusal(turn.conversation_id, message, faq)
        await repository.append_message(turn.conversation_id, "assistant", content=FAQ_REFUSAL)
        yield {"type": "delta", "text": FAQ_REFUSAL}
        yield {"type": "done", "conversation_id": turn.conversation_id}
        return
    if faq is not None:
        yield {"type": "citations", "items": faq.get("citations", [])}
    convergence = _convergence_messages(turn, runs, get_settings().token_budget)
    chunks: list[str] = []
    async for chunk in model.astream(convergence):
        part = _text(chunk)
        if part:
            chunks.append(part)
            yield {"type": "delta", "text": part}
    await repository.append_message(turn.conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": turn.conversation_id}
