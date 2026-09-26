"""Persisted graph turns respect MySQL ownership and per-thread serialization."""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph

from app.db import repository
from app.graph.state import ConversationState


def _build_graph(saver, *, entered=None, release=None):
    async def answer(state):
        if entered is not None:
            entered.set()
            await release.wait()
        text = str(len(state["messages"]))
        return {"answer": text, "messages": [AIMessage(content=text)]}

    builder = StateGraph(ConversationState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_runtime_allocates_mysql_conversation_and_restores_sqlite_history(
    tmp_path, db_session_factory, db_clean,
):
    from app.graph.runtime import GraphRuntime

    path = tmp_path / "checkpoint.sqlite"
    model = FakeListChatModel(responses=["unused"])
    async with GraphRuntime(path, _build_graph, enforce_audit=False) as runtime:
        first = await runtime.ainvoke_turn("owner", "第一轮", None, model=model)
        conversation_id = first["conversation_id"]
        assert first["answer"] == "1"
        assert (await repository.get_conversation(conversation_id)).user_id == "owner"

    async with GraphRuntime(path, _build_graph, enforce_audit=False) as runtime:
        second = await runtime.ainvoke_turn("owner", "第二轮", conversation_id, model=model)
    assert second["answer"] == "3"
    assert len(second["messages"]) == 4


@pytest.mark.asyncio
async def test_runtime_rejects_other_user_before_checkpoint_read(
    tmp_path, db_session_factory, db_clean,
):
    from app.graph.runtime import ConversationNotFound, GraphRuntime

    conversation_id = await repository.create_conversation("owner")
    model = FakeListChatModel(responses=["unused"])
    async with GraphRuntime(tmp_path / "checkpoint.sqlite", _build_graph, enforce_audit=False) as runtime:
        with pytest.raises(ConversationNotFound):
            await runtime.ainvoke_turn("intruder", "偷看", conversation_id, model=model)
        assert runtime.graph.get_state is not None


@pytest.mark.asyncio
async def test_same_conversation_returns_busy_instead_of_interleaving(
    tmp_path, db_session_factory, db_clean,
):
    from app.graph.runtime import ConversationBusy, GraphRuntime

    conversation_id = await repository.create_conversation("owner")
    entered, release = asyncio.Event(), asyncio.Event()
    model = FakeListChatModel(responses=["unused"])

    def factory(saver):
        return _build_graph(saver, entered=entered, release=release)

    async with GraphRuntime(tmp_path / "checkpoint.sqlite", factory, enforce_audit=False) as runtime:
        first = asyncio.create_task(runtime.ainvoke_turn("owner", "先来", conversation_id, model=model))
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(ConversationBusy):
            await runtime.ainvoke_turn("owner", "并发", conversation_id, model=model)
        release.set()
        assert (await first)["answer"] == "1"
