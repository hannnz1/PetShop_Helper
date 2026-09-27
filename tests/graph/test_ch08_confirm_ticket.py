"""Ticket writes require an explicit, resumable preview confirmation."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import nodes
from app.graph.build import build_graph
from app.graph.state import new_turn
from app.tools import engine, registry


class HumanClassifier:
    async def classify(self, query):
        return "人工"


class TicketModel:
    def __init__(self, args=None):
        self.args = args or {"description": "猫砂盆漏电", "ticket_type": "售后"}
        self.calls = 0
        self.bound_names = []

    def bind_tools(self, tools):
        self.bound_names.append({tool.name for tool in tools})
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket", "args": self.args, "id": "ticket-1",
            }])
        return AIMessage(content="工单操作已处理")

    async def astream(self, messages):
        yield AIMessageChunk(content="工单操作已处理")


@pytest.fixture()
def wired(monkeypatch):
    created, audits = [], []

    async def discover():
        return registry.builtin_specs()

    async def create(cid, description, ticket_type, request_id):
        created.append((cid, description, ticket_type, request_id))
        return "T-001"

    async def audit(**kwargs):
        audits.append(kwargs)

    async def no_log(*args, **kwargs):
        return 1

    monkeypatch.setattr(nodes.registry, "get_all_specs", discover)
    monkeypatch.setattr(nodes.repository, "create_ticket_only", create)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
    monkeypatch.setattr(nodes.repository, "append_turn_messages", no_log)
    return created, audits


@pytest.mark.asyncio
async def test_confirm_true_creates_ticket_after_interrupt(wired):
    created, audits = wired
    model = TicketModel()
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "ticket-yes"}}
    first = await graph.ainvoke(new_turn("u1", 1, "帮我建工单，猫砂盆漏电"), config,
                               context={"model": model, "classifier": HumanClassifier()})
    assert first["__interrupt__"][0].value == {
        "type": "confirm_ticket", "conversation_id": 1,
        "preview": {"description": "猫砂盆漏电", "ticket_type": "售后"},
    }
    assert created == []
    assert "create_ticket" in model.bound_names[0]

    finished = await graph.ainvoke(Command(resume={"confirmed": True}), config,
                                   context={"model": model, "classifier": HumanClassifier()})
    assert len(created) == 1
    assert created[0][:3] == (1, "猫砂盆漏电", "售后")
    assert created[0][3].startswith("graph-")
    assert any(a["status"] == "成功" for a in audits)
    assert "T-001" in str(finished["messages"])


@pytest.mark.asyncio
async def test_cancel_never_writes_and_is_audited(wired):
    created, audits = wired
    model = TicketModel()
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "ticket-no"}}
    await graph.ainvoke(new_turn("u1", 1, "帮我建工单，猫砂盆漏电"), config,
                        context={"model": model, "classifier": HumanClassifier()})
    await graph.ainvoke(Command(resume={"confirmed": False}), config,
                        context={"model": model, "classifier": HumanClassifier()})
    assert created == []
    assert any(a["status"] == "权限拒绝" for a in audits)


@pytest.mark.asyncio
async def test_missing_description_is_rejected_without_preview(wired):
    created, audits = wired
    model = TicketModel({"ticket_type": "咨询"})
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "ticket-missing"}}
    result = await graph.ainvoke(new_turn("u1", 1, "帮我建工单"), config,
                                 context={"model": model, "classifier": HumanClassifier()})
    assert "__interrupt__" not in result
    assert created == []
    assert any(a["status"] == "校验拦下" for a in audits)


@pytest.mark.asyncio
async def test_runtime_rejects_new_turn_and_wrong_resume_then_accepts_confirmation(
    wired, tmp_path, db_session_factory, db_clean,
):
    from app.db import repository
    from app.graph.runtime import ConversationPending, GraphDivergence, GraphRuntime

    cid = await repository.create_conversation("u1")
    model = TicketModel()
    async with GraphRuntime(tmp_path / "ticket.sqlite", build_graph,
                            classifier=HumanClassifier()) as runtime:
        first = await runtime.ainvoke_turn("u1", "帮我建工单，猫砂盆漏电", cid,
                                           model=model)
        assert first["__interrupt__"][0].value["type"] == "confirm_ticket"
        with pytest.raises(ConversationPending):
            await runtime.ainvoke_turn("u1", "换个话题", cid, model=model)
        with pytest.raises(GraphDivergence):
            await runtime.prepare_resume_turn("u1", cid, "1001", model=model)
        stream = await runtime.prepare_resume_turn("u1", cid, {"confirmed": False},
                                                   model=model)
        events = [event async for event in stream]
        assert any(mode == "updates" and "log_turn" in update
                   for mode, update in events)
        assert not (await runtime.graph.aget_state(
            {"configurable": {"thread_id": str(cid)}})).next
