"""Shared state carried between workflow nodes and conversation turns."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    conversation_id: int
    query: str
    resolved_query: str
    intent: str
    route: str
    evidence: str
    citations: list[dict]
    sufficient: bool
    source: str
    reason: str
    answer: str
    planned_tool_calls: list[dict]
    tool_calls: list[dict]
    tool_results: list[dict]
    steps: int
    tokens_used: int
    suggested_actions: list[dict]
    trace: dict


def new_turn(user_id: str, conversation_id: int, message: str) -> ConversationState:
    """Reset per-turn fields while the reducer keeps previous messages."""
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id, "conversation_id": conversation_id, "query": message,
        "resolved_query": "",
        "intent": "", "route": "", "evidence": "", "citations": [],
        "sufficient": False, "answer": "", "planned_tool_calls": [],
        "tool_calls": [], "tool_results": [], "steps": 0, "tokens_used": 0,
        "suggested_actions": [], "trace": {},
    }
