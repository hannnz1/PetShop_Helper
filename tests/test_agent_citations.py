"""RAG FAQ citations and low-confidence answers at the agent boundary."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select

from app.core import agent
from app.db import repository
from app.db.models import LowConfidenceQuestion
from app.tools.infra import ToolRun
from tests.test_agent_orchestration import FakeModel


def _run(payload, *, ok=True, name="query_faq"):
    return ToolRun("faq1", name, ok, ToolMessage(
        content=json.dumps(payload, ensure_ascii=False), tool_call_id="faq1",
    ))


def _planned():
    return AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "包邮吗"}, "id": "faq1"}
    ])


def test_faq_result_only_accepts_successful_rag_shape():
    citation = {"n": 1, "id": 5, "section_path": "运费政策"}
    faq = agent._faq_result([_run({"sufficient": True, "citations": [citation]})])
    assert faq["citations"] == [citation]
    assert agent._faq_result([_run({"hits": []})]) is None
    assert agent._faq_result([_run({"sufficient": False}, ok=False)]) is None
    assert agent._faq_result([_run({"sufficient": "false"})]) is None
    assert agent._faq_result([_run({"sufficient": False}, name="query_order")]) is None


@pytest.mark.asyncio
async def test_run_agent_refuses_and_pools_original_question(db_session_factory, db_clean, monkeypatch):
    async def fake_execute(*args, **kwargs):
        return _run({"sufficient": False, "source": "self_check", "reason": "没有退货期限证据", "citations": []})

    monkeypatch.setattr(agent, "execute_tool_call", fake_execute)
    model = FakeModel([_planned()])
    result = await agent.run_agent_turn("u1", "退货期限是多少？", None, model=model)
    assert "没有查到" in result.answer
    assert len(model.invocations) == 1  # no ungrounded second model answer
    assert result.citations == []
    async with db_session_factory() as session:
        row = (await session.execute(select(LowConfidenceQuestion))).scalar_one()
    assert row.raw_question == "退货期限是多少？"
    assert row.source == "self_check"
    assert row.conversation_id == result.conversation_id
    assert [m.role for m in await repository.list_messages(result.conversation_id)] == [
        "user", "assistant", "tool", "assistant",
    ]


@pytest.mark.asyncio
async def test_stream_citations_before_deltas(db_session_factory, db_clean, monkeypatch):
    citation = {"n": 1, "id": 5, "section_path": "运费政策"}

    async def fake_execute(*args, **kwargs):
        return _run({"sufficient": True, "evidence": "[1] 满99包邮", "citations": [citation]})

    monkeypatch.setattr(agent, "execute_tool_call", fake_execute)
    model = FakeModel([_planned()], chunks=["满99元", "包邮[1]"])
    events = [e async for e in agent.stream_agent_turn("u1", "包邮吗", None, model=model)]
    assert [e["type"] for e in events] == ["tool", "citations", "delta", "delta", "done"]
    assert events[1]["items"] == [citation]
    assert len(model.streams) == 1


@pytest.mark.asyncio
async def test_stream_refusal_has_no_citations(db_session_factory, db_clean, monkeypatch):
    async def fake_execute(*args, **kwargs):
        return _run({"sufficient": False, "source": "retrieval_low_conf", "reason": "无命中"})

    monkeypatch.setattr(agent, "execute_tool_call", fake_execute)
    model = FakeModel([_planned()], chunks=["不能出现"])
    events = [e async for e in agent.stream_agent_turn("u1", "货到付款吗", None, model=model)]
    assert [e["type"] for e in events] == ["tool", "delta", "done"]
    assert "没有查到" in events[1]["text"]
    assert model.streams == []
