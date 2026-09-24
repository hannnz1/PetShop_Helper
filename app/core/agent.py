"""One tool-planning round with persisted conversations and two answer exits."""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

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


@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]


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
        raise ValueError("system and current message exceed context budget")
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
    runs = await asyncio.gather(
        *(execute_tool_call(call, turn.conversation_id) for call in turn.planned.tool_calls)
    )
    for run in runs:
        await repository.append_message(
            turn.conversation_id,
            "tool",
            content=str(run.tool_message.content),
            tool_call_id=run.tool_call_id,
        )
    return runs


async def run_agent_turn(
    user_id: str, message: str, conversation_id: int | None, model=None
) -> AgentResult:
    """Produce a final JSON-ready answer, with at most one round of tools."""

    model = model if model is not None else get_chat_model()
    turn = await _prepare_turn(user_id, message, conversation_id, model)
    if not turn.planned.tool_calls:
        return AgentResult(turn.conversation_id, _text(turn.planned), [], [])

    runs = await _run_tools(turn)
    final: AIMessage = await model.ainvoke(
        [*turn.messages, turn.planned, *(run.tool_message for run in runs)]
    )
    answer = _text(final)
    await repository.append_message(turn.conversation_id, "assistant", content=answer)
    return AgentResult(turn.conversation_id, answer, turn.planned.tool_calls, runs)


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
    chunks: list[str] = []
    async for chunk in model.astream(
        [*turn.messages, turn.planned, *(run.tool_message for run in runs)]
    ):
        part = _text(chunk)
        if part:
            chunks.append(part)
            yield {"type": "delta", "text": part}
    await repository.append_message(turn.conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": turn.conversation_id}
