"""Chapter 7 async summary boundaries against the isolated MySQL schema."""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.api.chat import graph_event_stream
from app.core.summarizer import summarize_pending
from app.db import repository
from app.graph.runtime import GraphRuntime


class SummaryModel:
    def __init__(self, answer='订单 1001 的猫粮退款申请尚未解决'):
        self.answer = answer
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return AIMessage(content=self.answer)


async def seeded_layer2(conversation_id, text_content='订单 1001 的猫粮申请退款，仍未解决'):
    first = await repository.append_message(conversation_id, 'user', text_content)
    last = await repository.append_message(conversation_id, 'assistant', '已收到诉求')
    assert await repository.advance_layer1(conversation_id, last)
    return first, last


@pytest.mark.asyncio
async def test_token_threshold_and_append_only_batch(db_session_factory, db_clean):
    cid = await repository.create_conversation('owner')
    first, last = await seeded_layer2(cid, '订单 1001 的猫粮申请退款，仍未解决。' * 15)
    model = SummaryModel()
    outcome = await summarize_pending(cid, model, layer2_token_limit=20)
    assert outcome.status == 'committed'
    assert (outcome.from_msg_id, outcome.upto_msg_id) == (first, last)
    snapshot = await repository.get_context_snapshot(cid, 'owner')
    assert len(snapshot.summaries) == 1
    assert snapshot.summary_upto_msg_id == last
    assert '1001' in snapshot.summaries[0].content
    await repository.append_message(cid, 'user', '订单 2002 再问')
    later = await repository.append_message(cid, 'assistant', '明白')
    assert await repository.advance_layer1(cid, later)
    await summarize_pending(cid, model, layer2_token_limit=1)
    assert len(model.calls) == 2
    assert '1001' not in model.calls[1][-1].content
    second = await repository.get_context_snapshot(cid, 'owner')
    assert len(second.summaries) == 2
    assert second.summaries[0] == snapshot.summaries[0]


@pytest.mark.asyncio
async def test_threshold_counts_tokens_not_rows(db_session_factory, db_clean):
    cid = await repository.create_conversation('owner')
    await seeded_layer2(cid, '长' * 500)
    outcome = await summarize_pending(cid, SummaryModel(), layer2_token_limit=30)
    assert outcome.status == 'committed'


@pytest.mark.asyncio
async def test_failure_keeps_anchor_and_retry_succeeds(db_session_factory, db_clean):
    cid = await repository.create_conversation('owner')
    await seeded_layer2(cid)
    result = await summarize_pending(cid, SummaryModel(RuntimeError('offline')), layer2_token_limit=1)
    assert result.status == 'failed'
    assert (await repository.get_context_snapshot(cid, 'owner')).summary_upto_msg_id == 0
    assert (await summarize_pending(cid, SummaryModel(), layer2_token_limit=1)).status == 'committed'


@pytest.mark.asyncio
async def test_empty_facts_advance_with_skip_segment(db_session_factory, db_clean):
    cid = await repository.create_conversation('owner')
    await seeded_layer2(cid, '你好')
    result = await summarize_pending(cid, SummaryModel('无明确事实。'), layer2_token_limit=1)
    assert result.status == 'skipped'
    snapshot = await repository.get_context_snapshot(cid, 'owner')
    assert snapshot.summary_upto_msg_id == result.upto_msg_id
    assert snapshot.summaries[0].content == ''
    assert (await summarize_pending(cid, SummaryModel(), layer2_token_limit=1)).status == 'skipped'


@pytest.mark.asyncio
async def test_concurrent_tasks_commit_range_once(db_session_factory, db_clean):
    cid = await repository.create_conversation('owner')
    await seeded_layer2(cid)
    results = await asyncio.gather(*(summarize_pending(cid, SummaryModel(), layer2_token_limit=1)
                                     for _ in range(2)))
    snapshot = await repository.get_context_snapshot(cid, 'owner')
    assert len(snapshot.summaries) == 1
    assert sum(result.status == 'committed' for result in results) == 1


@pytest.mark.asyncio
async def test_restart_candidate_scan_finds_unprocessed_layer2(db_session_factory, db_clean):
    pending = await repository.create_conversation('owner')
    await seeded_layer2(pending)
    untouched = await repository.create_conversation('owner')
    await repository.append_message(untouched, 'user', '尚未结束')
    assert pending in await repository.list_summary_candidates()
    assert untouched not in await repository.list_summary_candidates()


@pytest.mark.asyncio
async def test_db_rejects_anchor_rewind_and_segment_mutation(_test_engine, db_session_factory, db_clean):
    cid = await repository.create_conversation('owner')
    first, last = await seeded_layer2(cid)
    await summarize_pending(cid, SummaryModel(), layer2_token_limit=1)
    with pytest.raises(DBAPIError):
        async with _test_engine.begin() as conn:
            await conn.execute(text('UPDATE conversations SET summary_upto_msg_id=0 WHERE id=:id'), {'id': cid})
    with pytest.raises(DBAPIError):
        async with _test_engine.begin() as conn:
            await conn.execute(text("UPDATE conversation_summaries SET content='tampered' WHERE conversation_id=:id"), {'id': cid})
    with pytest.raises(DBAPIError):
        async with _test_engine.begin() as conn:
            await conn.execute(text('DELETE FROM conversation_summaries WHERE conversation_id=:id'), {'id': cid})


@pytest.mark.asyncio
async def test_schedule_only_after_done_frame_and_not_pending(monkeypatch):
    scheduled = []
    monkeypatch.setattr('app.api.chat.schedule_summary', scheduled.append)

    async def completed():
        yield 'updates', {'log_turn': {'conversation_id': 7}}

    stream = graph_event_stream(completed(), 'owner')
    assert '"event": "done"' in await anext(stream)
    assert scheduled == []
    assert '[DONE]' in await anext(stream)
    assert scheduled == []
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert scheduled == [7]

    async def pending():
        yield 'updates', {'__interrupt__': [type('I', (), {'value': {'type': 'select_order', 'conversation_id': 7}})()]}

    assert not any('[DONE]' in frame for frame in [frame async for frame in graph_event_stream(pending(), 'owner')])
    assert scheduled == [7]


@pytest.mark.asyncio
async def test_disconnect_before_completion_does_not_schedule(monkeypatch):
    scheduled = []
    monkeypatch.setattr('app.api.chat.schedule_summary', scheduled.append)

    async def events():
        yield 'messages', (AIMessage(content='partial'), {'langgraph_node': 'final_answer'})
        yield 'updates', {'log_turn': {'conversation_id': 7}}

    stream = graph_event_stream(events(), 'owner')
    assert 'partial' in await anext(stream)
    await stream.aclose()
    assert scheduled == []


@pytest.mark.asyncio
async def test_nonstream_invoke_schedules_only_new_completed_audit(monkeypatch):
    scheduled = []
    monkeypatch.setattr('app.graph.runtime.schedule_summary', scheduled.append)
    monkeypatch.setattr(repository, 'get_context_snapshot', lambda *_: asyncio.sleep(0, result=object()))
    monkeypatch.setattr(repository, 'last_message_id', lambda *_: asyncio.sleep(0, result=3))
    runtime = GraphRuntime('unused.sqlite', lambda _: None, enforce_audit=False)
    runtime._conversation_id = lambda *_: asyncio.sleep(0, result=7)

    class Graph:
        result = {'trace': {'audit_message_id': 4}}
        next = ()

        async def ainvoke(self, *_args, **_kwargs):
            return self.result

        async def aget_state(self, *_args):
            return type('State', (), {'next': self.next})()

    runtime.graph = Graph()
    await runtime.ainvoke_turn('owner', 'hi', 7, model=SummaryModel())
    assert scheduled == [7]
    runtime.graph.result = {'trace': {'audit_message_id': 3}}
    runtime.graph.next = ('fetch_order',)
    await runtime.ainvoke_turn('owner', 'hi', 7, model=SummaryModel())
    assert scheduled == [7]
