"""Pure Chapter 7 views of persisted visible turns and the live Graph exchange."""

from copy import deepcopy
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import Settings, get_settings
from app.core.context_budget import ContextBudget, ContextBudgetExceeded
from app.core.memory import estimate_tokens, trim_history
from app.db.repository import ContextSnapshot, VisibleMessage


@dataclass(frozen=True)
class WindowRow:
    message_id: int | None
    layer: int
    role: str
    content: str


@dataclass(frozen=True)
class ModelContext:
    messages: list[BaseMessage]
    injected_summary: str
    token_count: int
    window_rows: tuple[WindowRow, ...]


def _complete_turns(snapshot: ContextSnapshot) -> list[tuple[VisibleMessage, VisibleMessage]]:
    """Use audited visible pairs only; global ID gaps need no special handling."""
    turns: list[tuple[VisibleMessage, VisibleMessage]] = []
    pending: VisibleMessage | None = None
    for row in snapshot.messages:
        if row.id <= snapshot.summary_upto_msg_id:
            continue
        if row.role == "user":
            pending = row
        elif row.role == "assistant" and pending is not None:
            turns.append((pending, row))
            pending = None
    return turns


def _layer_turns(snapshot: ContextSnapshot, layer: int,
                 settings: Settings) -> tuple[list[BaseMessage], list[WindowRow]]:
    messages: list[BaseMessage] = []
    rows: list[WindowRow] = []
    for user, assistant in _complete_turns(snapshot):
        # Anchors move only through a finished assistant row. Never split a turn.
        selected = 2 if assistant.id <= snapshot.layer1_from_msg_id else 1
        if selected != layer or (layer == 2 and user.id > snapshot.layer1_from_msg_id):
            continue
        reply = assistant.content
        if layer == 2:
            limit = settings.layer2_assistant_chars
            reply = reply[:limit] + ("…" if len(reply) > limit else "")
        messages.extend((HumanMessage(content=user.content), AIMessage(content=reply)))
        rows.extend((WindowRow(user.id, layer, "user", user.content),
                     WindowRow(assistant.id, layer, "assistant", reply)))
    return messages, rows


def _trim_layer(messages: list[BaseMessage], rows: list[WindowRow], max_tokens: int,
                settings: Settings) -> tuple[list[BaseMessage], list[WindowRow]]:
    kept = trim_history(messages, max_tokens, chars_per_token=settings.cjk_chars_per_token)
    return kept, rows[len(rows) - len(kept):] if kept else []


def _summary(snapshot: ContextSnapshot) -> str:
    return "\n".join(segment.content for segment in sorted(snapshot.summaries, key=lambda row: row.seq)
                     if segment.upto_msg_id <= snapshot.summary_upto_msg_id and segment.content.strip())


def _history(snapshot: ContextSnapshot, budget: ContextBudget,
             settings: Settings) -> tuple[list[BaseMessage], list[WindowRow]]:
    older, older_rows = _layer_turns(snapshot, 2, settings)
    recent, recent_rows = _layer_turns(snapshot, 1, settings)
    older, older_rows = _trim_layer(older, older_rows, budget.layer2, settings)
    recent, recent_rows = _trim_layer(recent, recent_rows, budget.layer1, settings)
    return [*older, *recent], [*older_rows, *recent_rows]


def build_history_context(snapshot: ContextSnapshot, budget: ContextBudget,
                          *, settings: Settings | None = None) -> str:
    """Summary and a budgeted visible window for coreference and intent."""
    history, _rows = _history(snapshot, budget, settings or get_settings())
    lines = [f"早期摘要：{_summary(snapshot)}"] if _summary(snapshot) else []
    for message in history:
        lines.append(f"{'用户' if isinstance(message, HumanMessage) else '客服'}：{message.content}")
    return "\n".join(lines)


def build_model_context(
    snapshot: ContextSnapshot, graph_messages: list[BaseMessage], current_query: str,
    evidence: str, system_text: str, budget: ContextBudget,
    *, settings: Settings | None = None,
) -> ModelContext:
    """Build an isolated model call, leaving Graph checkpoint messages untouched."""
    current_index = next((index for index in range(len(graph_messages) - 1, -1, -1)
                          if isinstance(graph_messages[index], HumanMessage)), None)
    if current_index is None or graph_messages[current_index].content != current_query:
        raise ValueError("current query is missing from Graph messages")
    settings = settings or get_settings()
    current = deepcopy(graph_messages[current_index:])
    if estimate_tokens(current, chars_per_token=settings.cjk_chars_per_token) > budget.current_peak:
        raise ContextBudgetExceeded("current graph exchange exceeds context token budget")

    history, rows = _history(snapshot, budget, settings)
    summary = _summary(snapshot)
    combined = "\n".join(part for part in (
        f"早期摘要：\n{summary}" if summary else "",
        f"本轮已核验证据：\n{evidence}" if evidence else "",
    ) if part)
    messages: list[BaseMessage] = [SystemMessage(content=system_text), *history, *current]
    if combined:
        messages.append(HumanMessage(content=combined, name="verified_context"))
    # Raw prior checkpoint tool payloads never enter model history. They are
    # represented as one-line audit markers for the caller's window log.
    prior_tool_count = sum(isinstance(message, ToolMessage) for message in graph_messages[:current_index])
    prior_tools = ([WindowRow(None, 2, "tool", f"[旧工具结果已省略: {prior_tool_count} 条]")]
                   if prior_tool_count else [])
    return ModelContext(messages, summary,
                        estimate_tokens(messages, chars_per_token=settings.cjk_chars_per_token),
                        tuple([*rows, *prior_tools]))
