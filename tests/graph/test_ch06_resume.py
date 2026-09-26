"""Only an owned, pending select_order interrupt may resume a graph thread."""

import pytest
from sqlalchemy import text

from app.db import repository


class RefundClassifier:
    async def classify(self, query):
        return "退款退货"


async def _seed(factory):
    async with factory.begin() as session:
        await session.execute(text("""
            INSERT INTO sample_orders (order_id,user_id,status,product,amount)
            VALUES ('1001','alice','已签收','猫粮',88.00),
                   ('2001','bob','已签收','猫砂',39.00)
        """))


@pytest.mark.asyncio
async def test_pending_select_order_rejects_new_turn_and_resumes_after_reopen(
    tmp_path, db_session_factory, db_clean, monkeypatch,
):
    from app.graph import nodes
    from app.graph.build import build_graph
    from app.graph.runtime import ConversationPending, GraphRuntime

    await _seed(db_session_factory)
    cid = await repository.create_conversation("alice")

    async def weak_policy(state, runtime):
        return {"sufficient": False, "evidence": "", "citations": [], "reason": "无政策"}

    monkeypatch.setattr(nodes, "retrieve_policy", weak_policy)
    path = tmp_path / "refund.sqlite"
    async with GraphRuntime(path, build_graph, classifier=RefundClassifier()) as runtime:
        first = await runtime.ainvoke_turn("alice", "能退吗", cid, model=object())
        assert first["__interrupt__"][0].value["type"] == "select_order"
        assert await repository.last_message_id(cid) is None
        with pytest.raises(ConversationPending):
            await runtime.ainvoke_turn("alice", "换个问题", cid, model=object())

    async with GraphRuntime(path, build_graph, classifier=RefundClassifier()) as runtime:
        stream = await runtime.prepare_resume_turn("alice", cid, "1001", model=object())
        events = [event async for event in stream]
        assert any(mode == "updates" and "log_turn" in payload for mode, payload in events)
        assert await repository.last_message_id(cid) is not None
        assert not (await runtime.graph.aget_state({"configurable": {"thread_id": str(cid)}})).next


@pytest.mark.asyncio
async def test_wrong_resume_value_reinterrupts_without_foreign_data(tmp_path, db_session_factory, db_clean):
    from app.graph.build import build_graph
    from app.graph.runtime import GraphRuntime

    await _seed(db_session_factory)
    cid = await repository.create_conversation("alice")
    async with GraphRuntime(tmp_path / "wrong.sqlite", build_graph, classifier=RefundClassifier()) as runtime:
        await runtime.ainvoke_turn("alice", "能退吗", cid, model=object())
        stream = await runtime.prepare_resume_turn("alice", cid, "2001", model=object())
        events = [event async for event in stream]
    interrupts = [payload["__interrupt__"][0].value for mode, payload in events
                  if mode == "updates" and "__interrupt__" in payload]
    assert len(interrupts) == 1
    assert [order["order_id"] for order in interrupts[0]["orders"]] == ["1001"]
    assert await repository.last_message_id(cid) is None


@pytest.mark.asyncio
async def test_other_pending_interrupt_cannot_be_resumed_as_order(tmp_path, db_session_factory, db_clean):
    from langgraph.graph import START, StateGraph
    from langgraph.types import interrupt
    from app.graph.runtime import GraphDivergence, GraphRuntime
    from app.graph.state import ConversationState

    cid = await repository.create_conversation("alice")

    def factory(saver):
        def other(state):
            interrupt({"type": "confirm_ticket"})

        builder = StateGraph(ConversationState)
        builder.add_node("other", other)
        builder.add_edge(START, "other")
        return builder.compile(checkpointer=saver)

    async with GraphRuntime(tmp_path / "other.sqlite", factory) as runtime:
        await runtime.ainvoke_turn("alice", "请确认", cid, model=object())
        with pytest.raises(GraphDivergence):
            await runtime.prepare_resume_turn("alice", cid, "1001", model=object())


@pytest.mark.asyncio
async def test_busy_resume_is_rejected_before_stream_starts(tmp_path, db_session_factory, db_clean):
    from app.graph.build import build_graph
    from app.graph.runtime import ConversationBusy, GraphRuntime

    cid = await repository.create_conversation("alice")
    async with GraphRuntime(tmp_path / "busy.sqlite", build_graph, classifier=RefundClassifier()) as runtime:
        runtime._claim(cid)
        try:
            with pytest.raises(ConversationBusy):
                await runtime.prepare_resume_turn("alice", cid, "1001", model=object())
        finally:
            runtime._active.remove(cid)
