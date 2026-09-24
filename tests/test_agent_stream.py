import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage, SystemMessage

from app.core import agent
from app.core.memory import estimate_tokens
from app.core.prompts import AGENT_SYSTEM
from app.db import repository
from app.tools.infra import ToolRun
from tests.test_agent_orchestration import FakeModel


@pytest.mark.asyncio
async def test_stream_no_tool_returns_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="您好")])
    events = [event async for event in agent.stream_agent_turn("u1", "你好", None, model=model)]
    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["text"] == "您好"
    assert model.streams == []
    assert [row.role for row in await repository.list_messages(events[-1]["conversation_id"])] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_status_emitted_before_tool_execution_and_streams_final(db_session_factory, db_clean, monkeypatch):
    planned = AIMessage(content="马上查到真实物流", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}
    ])
    model = FakeModel([planned], chunks=["演示", "运输中"])
    entered = asyncio.Event()
    release = asyncio.Event()
    original = agent.execute_tool_call

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent, "execute_tool_call", gated)
    stream = agent.stream_agent_turn("u1", "物流1001", None, model=model)
    first = await asyncio.wait_for(anext(stream), timeout=5)
    assert first == {"type": "tool", "name": "query_logistics"}
    assert not entered.is_set()  # first status precedes even starting the tool
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), timeout=5)
    assert not pending.done()
    release.set()
    rest = [await asyncio.wait_for(pending, timeout=5)]
    rest.extend([event async for event in stream])
    assert [event["type"] for event in rest] == ["delta", "delta", "done"]
    assert "".join(event["text"] for event in rest if event["type"] == "delta") == "演示运输中"
    assert "马上查到真实物流" not in str(rest)
    assert model.bind_calls == 1
    assert len(model.streams) == 1
    assert any(isinstance(message, ToolMessage) for message in model.streams[0])
    assert [row.role for row in await repository.list_messages(rest[-1]["conversation_id"])] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_stream_cross_user_rejected_without_done(db_session_factory, db_clean):
    cid = await repository.create_conversation("owner")
    model = FakeModel([AIMessage(content="不会调用")])
    with pytest.raises(agent.ConversationNotFound):
        [event async for event in agent.stream_agent_turn("other", "你好", cid, model=model)]
    assert await repository.list_messages(cid) == []


@pytest.mark.asyncio
async def test_stream_failure_does_not_emit_done(db_session_factory, db_clean, monkeypatch):
    model = FakeModel([AIMessage(content="马上提交", tool_calls=[
        {"name": "create_ticket", "args": {"description": "投诉", "ticket_type": "投诉"}, "id": "c1"}
    ])], chunks=["不应出现"])

    async def failed(*args, **kwargs):
        raise ConnectionError("db unavailable")

    monkeypatch.setattr(agent, "execute_tool_call", failed)
    stream = agent.stream_agent_turn("u1", "投诉", None, model=model)
    assert (await anext(stream))["type"] == "tool"
    with pytest.raises(ConnectionError):
        [event async for event in stream]
    assert model.streams == []


@pytest.mark.asyncio
async def test_stream_convergence_overflow_after_tool_status_has_no_done(db_session_factory, db_clean, monkeypatch):
    from app.config import get_settings

    baseline = estimate_tokens([SystemMessage(AGENT_SYSTEM), HumanMessage("查订单")])
    monkeypatch.setattr(agent, "get_settings", lambda: get_settings().model_copy(update={"token_budget": baseline + 100}))
    model = FakeModel([AIMessage(content="审计", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
    ])], chunks=["不应发送"])

    async def huge_tool(*args, **kwargs):
        return ToolRun("c1", "query_order", True, ToolMessage(content="X" * 1000, tool_call_id="c1"))

    monkeypatch.setattr(agent, "execute_tool_call", huge_tool)
    stream = agent.stream_agent_turn("u1", "查订单", None, model=model)
    assert (await anext(stream))["type"] == "tool"
    with pytest.raises(agent.ContextBudgetExceeded):
        [event async for event in stream]
    assert model.streams == []
