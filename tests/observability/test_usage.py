"""Real model usage is settled once per Graph turn without storing text."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, LLMResult
from sqlalchemy import select, text

from app.observability.usage import UsageCollector
from app.db.observability import save_usage_events
from app.db.models import ModelUsageEvent


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_cancelled_usage_settlement_releases_stream_claim(resume):
    from app.graph.runtime import GraphRuntime

    entered_settlement = asyncio.Event()

    class FinishedGraph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"intent": "refund"})

        async def astream(self, value, config, *, context, stream_mode):
            yield ("updates", {"final_answer": {"answer": "ok"}})

    async def resolved(*_):
        return 101

    async def nothing(*_, **__):
        return None

    async def snapshot(*_):
        return object()

    async def settlement(*_):
        entered_settlement.set()
        await asyncio.Event().wait()

    runtime = GraphRuntime("unused.sqlite", lambda _: None, enforce_audit=False)
    runtime.graph = FinishedGraph()
    runtime._conversation_id = resolved
    runtime._check_audit = nothing
    runtime._prepare_context_snapshot = snapshot
    runtime._settle_usage = settlement
    if resume:
        stream = await runtime.prepare_resume_turn("user", 101, "1001", model=object())
    else:
        stream = await runtime.prepare_stream_turn("user", "prompt", 101, model=object())

    async def consume():
        return [event async for event in stream]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered_settlement.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 101 not in runtime._active
    runtime._claim(101)
    assert 101 in runtime._active


def _result(usage, model="offline-test"):
    message = AIMessage(content="private reply", usage_metadata=usage,
                        response_metadata={"model_name": model})
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_same_run_id_is_counted_once():
    collector = UsageCollector("turn-1", 101)
    run_id = uuid4()
    result = _result({"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})
    collector.on_llm_end(result, run_id=run_id)
    collector.on_llm_end(result, run_id=run_id)
    events = collector.finalize("business")
    assert len(events) == 1
    assert (events[0].run_id, events[0].input_tokens, events[0].output_tokens) == (str(run_id), 11, 7)
    assert events[0].usage_status == "available"


def test_missing_usage_is_unavailable():
    collector = UsageCollector("turn-1", 101)
    collector.on_llm_end(_result(None), run_id=uuid4())
    event = collector.finalize("knowledge")[0]
    assert event.usage_status == "unavailable"
    assert event.input_tokens is None and event.output_tokens is None


def test_final_intent_applies_to_early_model_calls():
    collector = UsageCollector("turn-1", 101)
    collector.on_llm_end(_result({"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}),
                         run_id=uuid4())
    events = collector.finalize("refund")
    assert events[0].intent == "refund"
    assert events[0].turn_id == "turn-1" and events[0].conversation_id == 101


def test_failed_turn_has_unknown_intent_and_explicit_outcome():
    collector = UsageCollector("turn-1", 101)
    collector.on_llm_end(_result(None), run_id=uuid4())
    event = collector.finalize("refund", outcome="failed")[0]
    assert event.intent == "unknown" and event.turn_status == "failed"


def test_fake_stream_terminal_usage_settles_once_without_changing_tokens():
    class FakeModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "offline-test"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hi"))])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="h"))
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="i", usage_metadata={"input_tokens": 2, "output_tokens": 1,
                                             "total_tokens": 3}))

    collector = UsageCollector("turn-1", 101)
    chunks = list(FakeModel().stream("private prompt", config={"callbacks": [collector]}))
    assert "".join(chunk.content for chunk in chunks) == "hi"
    events = collector.finalize("business")
    assert len(events) == 1
    assert (events[0].input_tokens, events[0].output_tokens) == (2, 1)


@pytest.mark.asyncio
async def test_mysql_usage_is_idempotent_and_contains_no_text(db_session_factory, db_clean):
    collector = UsageCollector("turn-1", 101)
    run_id = uuid4()
    collector.on_llm_end(_result({"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}),
                         run_id=run_id)
    events = collector.finalize("business")
    await save_usage_events(events)
    await save_usage_events(events)
    async with db_session_factory() as session:
        rows = (await session.execute(select(ModelUsageEvent))).scalars().all()
        columns = {row[0] for row in (await session.execute(
            text("SHOW COLUMNS FROM model_usage_events"))).all()}
    assert len(rows) == 1
    assert rows[0].run_id == str(run_id)
    assert (rows[0].input_tokens, rows[0].output_tokens) == (4, 6)
    assert "content" not in ModelUsageEvent.__table__.columns
    assert "prompt" not in ModelUsageEvent.__table__.columns
    assert "api_key" not in ModelUsageEvent.__table__.columns
    assert columns == {
        "run_id", "conversation_id", "turn_id", "intent", "model_name",
        "input_tokens", "output_tokens", "usage_status", "turn_status",
        "is_estimated", "created_at",
    }


@pytest.mark.asyncio
async def test_graph_settles_final_intent_and_preserves_stream_events(monkeypatch):
    from app.graph.runtime import GraphRuntime

    saved = []

    async def save(events):
        saved.append(events)

    monkeypatch.setattr("app.graph.runtime.save_usage_events", save)
    monkeypatch.setattr("app.graph.runtime.repository.last_message_id", lambda *_: _none())

    class RecordingGraph:
        async def ainvoke(self, value, config, *, context):
            assert context["turn_id"] == config["metadata"]["turn_id"]
            collector = config["callbacks"][-1]
            collector.on_llm_end(_result({"input_tokens": 3, "output_tokens": 4,
                                          "total_tokens": 7}), run_id=uuid4())
            return {"intent": "business", "trace": {}}

        async def astream(self, value, config, *, context, stream_mode):
            assert context["turn_id"] == config["metadata"]["turn_id"]
            yield ("updates", {"classify_intent_node": {"intent": "knowledge"}})
            config["callbacks"][-1].on_llm_end(_result(None), run_id=uuid4())
            yield ("messages", ("token", {"node": "final_answer"}))

    async def _none():
        return None

    runtime = GraphRuntime("unused.sqlite", lambda _: None, enforce_audit=False)
    runtime.graph = RecordingGraph()
    runtime._conversation_id = lambda *_: _resolved()
    runtime._prepare_context_snapshot = lambda *_: _snapshot()

    async def _resolved():
        return 101

    async def _snapshot():
        return object()

    await runtime.ainvoke_turn("user", "private prompt", 101, model=object())
    stream = await runtime.prepare_stream_turn("user", "private prompt", 101, model=object())
    assert [event async for event in stream] == [
        ("updates", {"classify_intent_node": {"intent": "knowledge"}}),
        ("messages", ("token", {"node": "final_answer"})),
    ]
    assert [batch[0].intent for batch in saved] == ["business", "knowledge"]
    assert saved[0][0].turn_status == saved[1][0].turn_status == "completed"


@pytest.mark.asyncio
async def test_resume_uses_intent_from_pending_checkpoint(monkeypatch):
    from types import SimpleNamespace
    from app.graph.runtime import GraphRuntime

    saved = []
    async def save(events):
        saved.extend(events)
    monkeypatch.setattr("app.graph.runtime.save_usage_events", save)

    class ResumeGraph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"intent": "refund"})

        async def astream(self, value, config, *, context, stream_mode):
            config["callbacks"][-1].on_llm_end(_result(None), run_id=uuid4())
            yield ("updates", {"final_answer": {"answer": "private reply"}})

    async def resolved(*_):
        return 101

    async def nothing(*_, **__):
        return None

    async def snapshot(*_):
        return object()

    runtime = GraphRuntime("unused.sqlite", lambda _: None, enforce_audit=False)
    runtime.graph = ResumeGraph()
    runtime._conversation_id = resolved
    runtime._check_audit = nothing
    runtime._prepare_context_snapshot = snapshot
    stream = await runtime.prepare_resume_turn("user", 101, "1001", model=object())
    assert [event async for event in stream] == [
        ("updates", {"final_answer": {"answer": "private reply"}})]
    assert len(saved) == 1 and saved[0].intent == "refund"


@pytest.mark.asyncio
async def test_invoke_interrupt_settles_unknown(monkeypatch):
    from app.graph.runtime import GraphRuntime

    saved = []
    async def save(events):
        saved.extend(events)
    monkeypatch.setattr("app.graph.runtime.save_usage_events", save)
    async def nothing(*_, **__):
        return None
    async def resolved(*_):
        return 101
    async def snapshot(*_):
        return object()

    class InterruptGraph:
        async def ainvoke(self, value, config, *, context):
            config["callbacks"][-1].on_llm_end(_result(None), run_id=uuid4())
            return {"intent": "refund", "__interrupt__": ("select_order",), "trace": {}}

    runtime = GraphRuntime("unused.sqlite", lambda _: None, enforce_audit=False)
    runtime.graph = InterruptGraph()
    runtime._conversation_id = resolved
    runtime._check_audit = nothing
    runtime._prepare_context_snapshot = snapshot
    monkeypatch.setattr("app.graph.runtime.repository.last_message_id", nothing)
    await runtime.ainvoke_turn("user", "private prompt", 101, model=object())
    assert len(saved) == 1
    assert saved[0].intent == "unknown" and saved[0].turn_status == "interrupted"
