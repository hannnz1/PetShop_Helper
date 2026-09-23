"""Bounded, in-memory chat history for a single process."""

from copy import deepcopy

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


class SessionStore:
    """Keep successful conversation turns by session ID in process memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        """Return an isolated snapshot; an unknown ID does not create a session."""
        return deepcopy(self._sessions.get(session_id, []))

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(deepcopy(messages))

    def replace(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Commit bounded history after a successful turn."""
        self._sessions[session_id] = deepcopy(messages)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Estimate a conservative CJK-friendly budget, not provider-exact usage."""
    return count_tokens_approximately(messages, chars_per_token=1.0)


def _validate_complete_turns(messages: list[BaseMessage]) -> None:
    if len(messages) % 2:
        raise ValueError("history must contain complete Human/AI turns")
    for index in range(0, len(messages), 2):
        if not isinstance(messages[index], HumanMessage) or not isinstance(messages[index + 1], AIMessage):
            raise ValueError("history must contain complete Human/AI turns")


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Keep newest whole turns that fit; never retain half a turn."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    _validate_complete_turns(messages)

    start = len(messages)
    for index in range(len(messages) - 2, -1, -2):
        if estimate_tokens(messages[index:]) > max_tokens:
            break
        start = index

    if start == len(messages):
        return []
    # Use LangChain's last-message trimmer as the final budget gate. The
    # candidate is already a sequence of complete turns, so no half-turn can
    # enter the result even when the budget is exactly at the boundary.
    candidate = deepcopy(messages[start:])
    trimmed = trim_messages(
        candidate,
        strategy="last",
        token_counter=estimate_tokens,
        max_tokens=max_tokens,
        start_on="human",
        end_on="ai",
        allow_partial=False,
    )
    if len(trimmed) != len(candidate):
        raise RuntimeError("LangChain unexpectedly trimmed a pre-budgeted history")
    return trimmed


def build_context(
    system: SystemMessage,
    history: list[BaseMessage],
    current: HumanMessage,
    max_tokens: int,
) -> list[BaseMessage]:
    """Reserve the system and current input, then add fitting old turns."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    required = estimate_tokens([system, current])
    if required > max_tokens:
        raise ValueError("system and current message exceed the context budget")
    kept = trim_history(history, max_tokens - required)
    return [deepcopy(system), *kept, deepcopy(current)]
