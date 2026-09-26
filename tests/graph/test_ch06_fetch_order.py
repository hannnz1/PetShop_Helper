"""Owned demo order selection must pause and resume through a compiled graph."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from sqlalchemy import text

from app.graph import nodes
from app.graph.routing import route_by_intent
from app.graph.state import ConversationState


def _order_graph():
    builder = StateGraph(ConversationState)
    builder.add_node("fetch_order", nodes.fetch_order)
    builder.add_edge(START, "fetch_order")
    return builder.compile(checkpointer=InMemorySaver())


async def _seed(factory):
    async with factory.begin() as session:
        await session.execute(text("""
            INSERT INTO sample_orders (order_id,user_id,status,product,amount)
            VALUES ('1001','alice','已签收','猫粮',88.00),
                   ('1002','alice','已发货','饮水机',139.00),
                   ('2001','bob','已签收','猫砂',39.00)
        """))


def test_after_sales_and_refunds_route_to_refund_flow():
    assert route_by_intent("退款退货") == "refund"
    assert route_by_intent("售后") == "refund"


@pytest.mark.asyncio
async def test_chinese_adjacent_order_number_fetches_owned_snapshot(db_session_factory, db_clean):
    await _seed(db_session_factory)
    graph = _order_graph()
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="订单1001能退吗")], "query": "订单1001能退吗",
         "resolved_query": "订单1001能退吗", "user_id": "alice"},
        {"configurable": {"thread_id": "owned-1"}},
    )
    assert state["order_id"] == "1001"
    assert state["order_data"]["product"] == "猫粮"
    assert "__interrupt__" not in state


@pytest.mark.asyncio
async def test_missing_and_foreign_order_reinterrupt_then_accept_owned(db_session_factory, db_clean):
    await _seed(db_session_factory)
    graph = _order_graph()
    config = {"configurable": {"thread_id": "choose-1"}}
    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="能退吗")], "query": "能退吗",
         "resolved_query": "能退吗", "user_id": "alice"}, config,
    )
    payload = first["__interrupt__"][0].value
    assert payload["type"] == "select_order"
    assert [row["order_id"] for row in payload["orders"]] == ["1001", "1002"]

    wrong = await graph.ainvoke(Command(resume="2001"), config)
    assert [row["order_id"] for row in wrong["__interrupt__"][0].value["orders"]] == ["1001", "1002"]

    chosen = await graph.ainvoke(Command(resume="1002"), config)
    assert chosen["order_id"] == "1002"
    assert chosen["order_data"]["status"] == "已发货"


@pytest.mark.asyncio
async def test_foreign_spoken_order_never_exposes_foreign_snapshot(db_session_factory, db_clean):
    await _seed(db_session_factory)
    graph = _order_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="订单2001能退吗")], "query": "订单2001能退吗",
         "resolved_query": "订单2001能退吗", "user_id": "alice"},
        {"configurable": {"thread_id": "foreign-1"}},
    )
    assert "order_data" not in result
    assert all(row["order_id"] != "2001" for row in result["__interrupt__"][0].value["orders"])


@pytest.mark.asyncio
async def test_user_with_no_demo_orders_gets_terminal_fallback(db_session_factory, db_clean):
    graph = _order_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="能退吗")], "query": "能退吗",
         "resolved_query": "能退吗", "user_id": "nobody"},
        {"configurable": {"thread_id": "none-1"}},
    )
    assert result["no_orders"] is True
    assert "__interrupt__" not in result
