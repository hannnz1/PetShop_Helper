"""The business loop can use a prior tool result without exposing writes."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.graph.state import new_turn


class Classifier:
    def __init__(self, intent):
        self.intent = intent

    async def classify(self, query):
        return self.intent


class Model:
    def __init__(self, replies, final="这是演示结果"):
        self.replies = list(replies)
        self.final = final
        self.bound_names = []
        self.plans = []
        self.final_prompts = []

    def bind_tools(self, tools):
        self.bound_names.append({tool.name for tool in tools})
        return self

    async def ainvoke(self, messages):
        self.plans.append(list(messages))
        return self.replies.pop(0)

    async def astream(self, messages):
        self.final_prompts.append(list(messages))
        for part in (self.final[:3], self.final[3:]):
            yield AIMessageChunk(content=part)


async def _no_log(*args, **kwargs):
    return 1


@pytest.mark.asyncio
async def test_second_tool_decision_sees_first_result_and_final_answer_is_separate(monkeypatch):
    from app.graph.build import build_graph
    from app.graph import nodes
    from app.tools import registry
    from app.tools.business import query_order
    from langchain_core.tools import tool

    monkeypatch.setattr(nodes.repository, "append_turn_messages", _no_log)
    monkeypatch.setattr(nodes.engine.repository, "insert_tool_audit", _no_log)

    @tool
    async def query_logistics(tracking_no: str) -> dict:
        """查询演示物流轨迹。"""
        return {"tracking_no": tracking_no, "status": "运输中"}

    async def discover():
        return [*registry.builtin_specs(), registry.spec_from_langchain_tool(
            query_logistics, source="mcp", mcp_server="logistics")]

    monkeypatch.setattr(nodes.registry, "get_all_specs", discover)
    tracking_no = (await query_order.ainvoke({"order_id": "1001"}))["tracking_no"]
    model = Model([
        AIMessage(content="先查订单", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "order-1"}]),
        AIMessage(content="再查物流", tool_calls=[
            {"name": "query_logistics", "args": {"tracking_no": tracking_no}, "id": "logistics-2"}]),
        AIMessage(content="不再调用工具"),
    ])
    graph = build_graph()
    result = await graph.ainvoke(
        new_turn("u", 1, "订单1001物流怎样"),
        context={"model": model, "classifier": Classifier("物流")},
    )
    assert result["answer"] == "这是演示结果"
    assert result["steps"] == 3
    assert [call["name"] for call in result["tool_calls"]] == ["query_order", "query_logistics"]
    assert all(names == {"query_order", "query_product", "query_logistics"}
               for names in model.bound_names)
    assert any(isinstance(message, ToolMessage) and "1001" in str(message.content)
               for message in model.plans[1])
    assert len(model.final_prompts) == 1


@pytest.mark.asyncio
async def test_model_forged_ticket_call_is_rejected_without_write(monkeypatch):
    from app.graph.build import build_graph
    from app.graph import nodes

    monkeypatch.setattr(nodes.repository, "append_turn_messages", _no_log)

    async def forbidden(*args):
        raise AssertionError("model must never create a ticket")

    monkeypatch.setattr(nodes.repository, "create_ticket", forbidden)
    model = Model([
        AIMessage(content="", tool_calls=[
            {"name": "create_ticket", "args": {"description": "投诉", "ticket_type": "投诉"},
             "id": "forged"}]),
        AIMessage(content="不能直接建单"),
    ])
    result = await build_graph().ainvoke(
        new_turn("u", 1, "查订单1001"),
        context={"model": model, "classifier": Classifier("订单")},
    )
    assert result["tool_results"][0]["ok"] is False
    assert "create_ticket" not in model.bound_names[0]


@pytest.mark.asyncio
async def test_sixth_planning_step_stops_without_executing_sixth_tool(monkeypatch):
    from app.graph.build import build_graph
    from app.graph import nodes

    monkeypatch.setattr(nodes.repository, "append_turn_messages", _no_log)
    replies = [AIMessage(content="", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1001"}, "id": f"call-{index}"}
    ]) for index in range(6)]
    model = Model(replies)
    result = await build_graph().ainvoke(
        new_turn("u", 1, "一直查询"),
        context={"model": model, "classifier": Classifier("订单")},
    )
    assert result["steps"] == 6
    assert len(result["tool_results"]) == 5
    assert "暂时" in result["answer"]
    assert model.final_prompts == []


@pytest.mark.asyncio
async def test_knowledge_route_binds_only_order_tool_after_valid_evidence(monkeypatch):
    from app.graph.build import build_graph
    from app.graph import nodes

    monkeypatch.setattr(nodes.repository, "append_turn_messages", _no_log)

    async def rag(state):
        return {"sufficient": True, "evidence": "[1] 商品参数", "citations": [
            {"n": 1, "id": 1, "section_path": "商品", "answer": "商品参数"}]}

    monkeypatch.setattr(nodes, "forced_rag", rag)
    model = Model([AIMessage(content="无需工具")])
    result = await build_graph().ainvoke(
        new_turn("u", 1, "商品参数"),
        context={"model": model, "classifier": Classifier("商品咨询")},
    )
    assert model.bound_names == [{"query_order"}]
    assert result["citations"][0]["n"] == 1
