"""Feedback must target the answer actually committed by this stream."""
import json
from types import SimpleNamespace

import pytest

from app.api.chat import graph_event_stream


async def collect(updates, monkeypatch):
    monkeypatch.setattr('app.api.chat.schedule_summary', lambda *a, **kw: None)
    async def stream():
        for update in updates:
            if isinstance(update, Exception):
                raise update
            yield 'updates', update
    frames = [frame async for frame in graph_event_stream(stream(), 'owner')]
    return [json.loads(frame[6:]) for frame in frames if frame.startswith('data: {')]


@pytest.mark.asyncio
async def test_done_uses_committed_audit_id(monkeypatch):
    frames = await collect([{'log_turn': {'conversation_id': 7, 'trace': {'audit_message_id': 42}}}], monkeypatch)
    assert frames[-1] == {'event': 'done', 'conversation_id': 7, 'assistant_message_id': 42}


@pytest.mark.asyncio
async def test_failure_has_no_feedback_completion(monkeypatch):
    frames = await collect([ConnectionError('save failed')], monkeypatch)
    assert not any(frame.get('event') == 'done' for frame in frames)


@pytest.mark.asyncio
async def test_legacy_done_remains_compatible(monkeypatch):
    frames = await collect([{'log_turn': {'conversation_id': 7}}], monkeypatch)
    assert frames[-1] == {'event': 'done', 'conversation_id': 7}


@pytest.mark.asyncio
async def test_interrupted_turn_has_no_feedback_id(monkeypatch):
    frames = await collect([
        {'__interrupt__': [SimpleNamespace(value={'type': 'select_order', 'conversation_id': 7})]},
        {'log_turn': {'conversation_id': 7, 'trace': {'audit_message_id': 42}}},
    ], monkeypatch)
    assert frames[-1]['event'] == 'interrupt'
    assert not any('assistant_message_id' in frame or frame.get('event') == 'done' for frame in frames)
