"""Deterministic Chapter 5 state and route decisions."""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.config import Settings


@pytest.mark.parametrize(
    ("intent", "route"),
    [
        ("物流", "business"), ("订单", "business"), ("售后", "business"),
        ("商品咨询", "knowledge"), ("退款退货", "knowledge"),
        ("投诉", "complaint"), ("闲聊", "chitchat"),
        ("不存在", "fallback"), ("", "fallback"),
    ],
)
def test_seven_intents_have_fixed_safe_routes(intent, route):
    from app.graph.routing import route_by_intent

    assert route_by_intent(intent) == route


@pytest.mark.parametrize(
    ("steps", "calls", "expected"),
    [(5, [{"name": "query_order"}], "tools"),
     (6, [{"name": "query_order"}], "fallback"),
     (2, [], "final")],
)
def test_react_decision_is_bounded(steps, calls, expected):
    from app.graph.routing import should_continue

    assert should_continue({"steps": steps, "planned_tool_calls": calls}, max_steps=6) == expected


def test_new_turn_resets_transient_fields_but_appends_only_current_human_message():
    from app.graph.state import new_turn

    state = new_turn("owner", 17, "还有货吗")
    assert state["user_id"] == "owner" and state["conversation_id"] == 17
    assert state["steps"] == 0 and state["citations"] == []
    assert state["tool_calls"] == [] and state["suggested_actions"] == []
    assert state["intent"] == "" and state["evidence"] == ""
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "还有货吗"


@pytest.mark.asyncio
async def test_unknown_or_failed_classifier_never_routes_to_tools():
    from app.core.intent import safe_classify
    from app.graph.routing import route_by_intent

    class Unknown:
        async def classify(self, query):
            return "business"

    class Broken:
        async def classify(self, query):
            raise TimeoutError()

    assert await safe_classify(Unknown(), "test") == "unknown"
    assert await safe_classify(Broken(), "test") == "unknown"
    assert route_by_intent("unknown") == "fallback"


def test_max_agent_steps_default_and_validation():
    values = {"chat_model": "test", "chat_base_url": "https://example.test/v1",
              "chat_api_key": "test-key", "_env_file": None}
    assert Settings(**values).max_agent_steps == 6
    with pytest.raises(ValidationError):
        Settings(**values, max_agent_steps=0)
