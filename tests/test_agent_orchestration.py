"""Task 11: orchestration against a real isolated MySQL repository."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from app.core import agent
from app.db import repository


class FakeModel:
    def __init__(self, replies, chunks=()):
        self.replies = list(replies)
        self.chunks = list(chunks)
        self.bind_calls = 0
        self.bound_tools = []
        self.invocations = []
        self.streams = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        self.invocations.append(list(messages))
        return self.replies.pop(0)

    async def astream(self, messages):
        self.streams.append(list(messages))
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk)


@pytest.mark.asyncio
async def test_first_turn_without_tool_is_persisted(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="您好")])
    result = await agent.run_agent_turn("u1", "你好", None, model=model)
    assert result.answer == "您好"
    assert result.tool_calls == result.tool_runs == []
    assert model.bind_calls == 1
    assert isinstance(model.invocations[0][0], SystemMessage)
    assert isinstance(model.invocations[0][-1], HumanMessage)
    assert [row.role for row in await repository.list_messages(result.conversation_id)] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_tool_result_converges_on_unbound_model_and_hides_preamble(db_session_factory, db_clean):
    planned = AIMessage(content="订单已查到", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call-1"}
    ])
    model = FakeModel([planned, AIMessage(content="演示记录显示物流运输中")])
    result = await agent.run_agent_turn("u1", "订单1001到哪了", None, model=model)
    assert result.answer == "演示记录显示物流运输中"
    assert result.tool_runs[0].ok
    assert model.bind_calls == 1
    assert any(isinstance(message, ToolMessage) for message in model.invocations[1])
    rows = await repository.list_messages(result.conversation_id)
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].content == "订单已查到"  # audit only
    assert rows[2].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_continuation_replays_complete_user_final_pairs_only(db_session_factory, db_clean):
    first = FakeModel([
        AIMessage(content="我已提交", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="这是模拟订单记录"),
    ])
    cid = (await agent.run_agent_turn("u1", "查订单1001", None, model=first)).conversation_id
    second = FakeModel([AIMessage(content="不客气")])
    await agent.run_agent_turn("u1", "谢谢", cid, model=second)
    contents = [message.content for message in second.invocations[0]]
    assert "查订单1001" in contents
    assert "这是模拟订单记录" in contents
    assert "我已提交" not in contents
    assert not any(isinstance(message, ToolMessage) for message in second.invocations[0])


@pytest.mark.asyncio
async def test_incomplete_persisted_turn_is_not_replayed(db_session_factory, db_clean):
    cid = await repository.create_conversation("u1")
    await repository.append_message(cid, "user", "旧的未完成问题")
    await repository.append_message(cid, "assistant", "仅审计", tool_calls=[{"name": "query_order", "args": {}, "id": "old"}])
    model = FakeModel([AIMessage(content="您好")])
    await agent.run_agent_turn("u1", "新问题", cid, model=model)
    contents = [message.content for message in model.invocations[0]]
    assert "旧的未完成问题" not in contents
    assert "仅审计" not in contents


@pytest.mark.asyncio
async def test_cross_user_and_missing_conversation_rejected_before_write(db_session_factory, db_clean):
    cid = await repository.create_conversation("owner")
    model = FakeModel([AIMessage(content="泄露")])
    for forbidden in (cid, 999999999):
        with pytest.raises(agent.ConversationNotFound):
            await agent.run_agent_turn("other", "试图续接", forbidden, model=model)
    assert await repository.list_messages(cid) == []
    assert model.invocations == []


@pytest.mark.asyncio
async def test_system_and_current_count_against_budget(db_session_factory, db_clean, monkeypatch):
    from app.config import get_settings

    settings = get_settings().model_copy(update={"token_budget": 1})
    monkeypatch.setattr(agent, "get_settings", lambda: settings)
    model = FakeModel([AIMessage(content="不会调用")])
    with pytest.raises(ValueError, match="budget"):
        await agent.run_agent_turn("u1", "你好", None, model=model)
    assert model.invocations == []


@pytest.mark.asyncio
async def test_database_failure_propagates_instead_of_normal_result(db_session_factory, db_clean, monkeypatch):
    async def failed(*args, **kwargs):
        raise ConnectionError("db unavailable")

    monkeypatch.setattr(agent.repository, "append_message", failed)
    model = FakeModel([AIMessage(content="不会调用")])
    with pytest.raises(ConnectionError):
        await agent.run_agent_turn("u1", "你好", None, model=model)
    assert model.invocations == []


@pytest.mark.asyncio
async def test_business_tool_failure_is_fed_back_for_final_answer(db_session_factory, db_clean):
    planned = AIMessage(content="已经找到订单", tool_calls=[
        {"name": "unknown_tool", "args": {}, "id": "bad-call"}
    ])
    model = FakeModel([planned, AIMessage(content="目前无法查到，您可以核对订单号")])
    result = await agent.run_agent_turn("u1", "查一下", None, model=model)
    assert result.answer == "目前无法查到，您可以核对订单号"
    assert not result.tool_runs[0].ok
    assert model.invocations[1][-1].status == "error"
    assert "已经找到订单" not in result.answer


@pytest.mark.asyncio
async def test_budget_trims_old_whole_pairs(db_session_factory, db_clean, monkeypatch):
    from app.config import get_settings

    cid = await repository.create_conversation("u1")
    await repository.append_message(cid, "user", "旧问题" * 200)
    await repository.append_message(cid, "assistant", "旧答案" * 200)
    await repository.append_message(cid, "user", "最近问题")
    await repository.append_message(cid, "assistant", "最近答案")
    settings = get_settings().model_copy(update={"token_budget": 1000})
    monkeypatch.setattr(agent, "get_settings", lambda: settings)
    model = FakeModel([AIMessage(content="新答案")])
    await agent.run_agent_turn("u1", "新问题", cid, model=model)
    contents = [message.content for message in model.invocations[0]]
    assert "最近问题" in contents and "最近答案" in contents
    assert "旧问题" * 200 not in contents
    assert len(model.invocations[0]) == 4
