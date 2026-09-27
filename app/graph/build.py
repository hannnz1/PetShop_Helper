"""Compose the deterministic exits around the bounded tool loop."""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.core.intent import IntentClassifier
from app.config import get_settings
from app.config import Settings
from app.db.repository import ContextSnapshot
from app.core.context_budget import ContextBudget
from app.graph import nodes
from app.graph.routing import route_by_intent, should_continue
from app.graph.state import ConversationState


class GraphContext(TypedDict, total=False):
    model: BaseChatModel
    classifier: IntentClassifier
    snapshot: ContextSnapshot
    budget: ContextBudget
    settings: Settings
    turn_id: str


def _route_agent(state: ConversationState, runtime: Runtime[GraphContext]) -> str:
    settings = runtime.context.get("settings") if runtime.context else None
    return should_continue(state, max_steps=(settings or get_settings()).max_agent_steps)


def build_graph(checkpointer=None):
    builder = StateGraph(ConversationState, context_schema=GraphContext)
    for name in (
        "coref", "classify_intent_node", "fetch_order", "retrieve_policy",
        "forced_rag", "agent_llm",
        "agent_tools", "final_answer", "chitchat_reply", "complaint_reply",
        "fallback_reply", "log_turn",
    ):
        builder.add_node(name, getattr(nodes, name))
    builder.add_edge(START, "coref")
    builder.add_edge("coref", "classify_intent_node")
    builder.add_conditional_edges(
        "classify_intent_node",
        lambda state: route_by_intent(state["intent"]),
        {"business": "agent_llm", "knowledge": "forced_rag", "refund": "fetch_order",
         "complaint": "complaint_reply", "chitchat": "chitchat_reply",
         "fallback": "fallback_reply"},
    )
    builder.add_conditional_edges(
        "fetch_order", lambda state: "empty" if state.get("no_orders") else "found",
        {"empty": "fallback_reply", "found": "retrieve_policy"},
    )
    builder.add_conditional_edges(
        "retrieve_policy", nodes.confidence_gate,
        {"agent": "agent_llm", "fallback": "fallback_reply"},
    )
    builder.add_conditional_edges(
        "forced_rag", nodes.confidence_gate,
        {"agent": "agent_llm", "fallback": "fallback_reply"},
    )
    builder.add_conditional_edges(
        "agent_llm",
        _route_agent,
        {"tools": "agent_tools", "final": "final_answer", "fallback": "fallback_reply"},
    )
    builder.add_edge("agent_tools", "agent_llm")
    for name in ("final_answer", "chitchat_reply", "complaint_reply", "fallback_reply"):
        builder.add_edge(name, "log_turn")
    builder.add_edge("log_turn", END)
    return builder.compile(checkpointer=checkpointer)
