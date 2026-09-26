"""A model can suggest a refund form, but cannot persist an application."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from app.graph import nodes
from app.tools.infra import execute_tool_call
from app.tools.registry import get_chat_tools


async def _seed(factory):
    async with factory.begin() as session:
        await session.execute(text("""
            INSERT INTO sample_orders (order_id,user_id,status,product,amount)
            VALUES ('1001','alice','已签收','猫粮',88.00),
                   ('2001','bob','已签收','猫砂',39.00)
        """))


def _state(order_id="1001", *, sufficient=True, call_order_id="1001", actions=None):
    return {
        "route": "refund", "user_id": "alice", "conversation_id": 1,
        "order_id": order_id, "order_data": {"order_id": order_id, "status": "已签收"},
        "sufficient": sufficient, "evidence": "[1] 退款政策" if sufficient else "",
        "citations": [{"n": 1, "section_path": "售后", "answer": "退款政策"}] if sufficient else [],
        "planned_tool_calls": [{"name": "submit_refund", "args": {"order_id": call_order_id, "reason": "质量问题"},
                                "id": "refund-call-1"}],
        "tool_results": [], "suggested_actions": actions or [],
    }


@pytest.mark.asyncio
async def test_refund_route_only_binds_suggestion_and_dispatcher_refuses_write(db_session_factory, db_clean):
    assert {tool.name for tool in get_chat_tools("refund")} == {"submit_refund"}
    result = await execute_tool_call(
        {"name": "submit_refund", "args": {"order_id": "1001"}, "id": "forged"}, 1,
        allowed_names={"submit_refund"},
    )
    assert result.ok is False
    assert "用户确认" in result.tool_message.content
    async with db_session_factory() as session:
        assert (await session.execute(text("SELECT COUNT(*) FROM refund_requests"))).scalar_one() == 0


@pytest.mark.asyncio
async def test_owned_order_and_strong_evidence_yield_form_without_db_write(db_session_factory, db_clean):
    await _seed(db_session_factory)
    result = await nodes.agent_tools(_state())
    assert result["suggested_actions"] == [
        {"type": "refund_form", "draft": {"order_id": "1001", "reason": "质量问题"}},
    ]
    assert result["tool_results"][0]["ok"] is True
    assert "待用户确认" in result["messages"][0].content
    async with db_session_factory() as session:
        assert (await session.execute(text("SELECT COUNT(*) FROM refund_requests"))).scalar_one() == 0


@pytest.mark.asyncio
async def test_weak_evidence_or_wrong_order_never_yields_refund_form(db_session_factory, db_clean):
    await _seed(db_session_factory)
    weak = await nodes.agent_tools(_state(sufficient=False))
    wrong = await nodes.agent_tools(_state(call_order_id="2001"))
    assert not any(action["type"] == "refund_form" for action in weak.get("suggested_actions", []))
    assert not any(action["type"] == "refund_form" for action in wrong.get("suggested_actions", []))
    assert wrong["tool_results"][0]["ok"] is False
    assert [order["order_id"] for order in wrong["suggested_actions"][0]["orders"]] == ["1001"]


@pytest.mark.asyncio
async def test_repeat_suggestion_does_not_duplicate_form(db_session_factory, db_clean):
    await _seed(db_session_factory)
    existing = [{"type": "refund_form", "draft": {"order_id": "1001", "reason": "质量问题"}}]
    result = await nodes.agent_tools(_state(actions=existing))
    assert result["suggested_actions"] == existing


@pytest.mark.asyncio
async def test_forged_refund_tool_on_non_refund_route_shows_no_order_selector(db_session_factory, db_clean):
    await _seed(db_session_factory)
    state = _state(call_order_id="2001")
    state["route"] = "business"
    result = await nodes.agent_tools(state)
    assert result["suggested_actions"] == []
    assert result["tool_results"][0]["ok"] is False


@pytest.mark.asyncio
async def test_suggestion_tool_itself_checks_order_ownership(db_session_factory, db_clean):
    from app.tools.refunds import submit_refund

    await _seed(db_session_factory)
    owned = await submit_refund.ainvoke({"order_id": "1001", "user_id": "alice"})
    foreign = await submit_refund.ainvoke({"order_id": "2001", "user_id": "alice"})
    assert owned["status"] == "待用户确认"
    assert foreign["status"] == "订单不属于当前用户"


def test_refund_agent_prompt_contains_verified_order_snapshot():
    state = _state()
    state["messages"] = [HumanMessage(content="订单1001能退吗")]
    prompt = nodes._fit_messages(state, "客服规则")
    system_text = "\n".join(message.content for message in prompt if isinstance(message, SystemMessage))
    assert "1001" in system_text and "已签收" in system_text
