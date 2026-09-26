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
    order_id: str
    order_data: dict
    no_orders: bool
    intent: str
    route: str
    evidence: str
    history_ctx: str
    model_ctx: dict
    summary_layer2_budget: int
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
    """Reset turn fields; the reducer keeps messages and trace keeps the audit marker."""
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id, "conversation_id": conversation_id, "query": message,
        "resolved_query": "",
        "order_id": "", "order_data": {}, "no_orders": False,
        "intent": "", "route": "", "evidence": "", "citations": [],
        "history_ctx": "", "model_ctx": {}, "summary_layer2_budget": 0,
        "sufficient": False, "answer": "", "planned_tool_calls": [],
        "tool_calls": [], "tool_results": [], "steps": 0, "tokens_used": 0,
        "suggested_actions": [],
    }
