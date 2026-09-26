"""Chapter-two streaming HTTP contract."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.core import agent
from app.main import create_app
from app.tools.infra import ToolInfrastructureError


def _client(monkeypatch, producer):
    monkeypatch.setattr(agent, "stream_agent_turn", producer)
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test")
    return TestClient(create_app(settings=settings, model=object()))


def _frames(response):
    return [frame for frame in response.text.split("\n\n") if frame]


def test_stream_tools_deltas_and_done(monkeypatch):
    async def producer(user_id, message, conversation_id, model=None):
        assert (user_id, message, conversation_id) == ("u1", "查物流", None)
        yield {"type": "tool", "name": "query_logistics"}
        yield {"type": "delta", "text": "演示"}
        yield {"type": "delta", "text": "运输中"}
        yield {"type": "done", "conversation_id": 12}

    with _client(monkeypatch, producer) as client:
        response = client.post("/api/chat", json={"user_id": "u1", "message": "查物流"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert _frames(response) == [
        'data: {"event": "tool", "name": "query_logistics"}',
        'data: {"delta": "演示"}', 'data: {"delta": "运输中"}',
        'data: {"event": "done", "conversation_id": 12}', 'data: [DONE]',
    ]


def test_stream_forwards_citations_before_answer(monkeypatch):
    citation = {"n": 1, "id": 5, "section_path": "运费政策"}

    async def producer(*args, **kwargs):
        yield {"type": "citations", "items": [citation]}
        yield {"type": "delta", "text": "满99元包邮[1]"}
        yield {"type": "done", "conversation_id": 12}

    with _client(monkeypatch, producer) as client:
        response = client.post("/api/chat", json={"user_id": "u1", "message": "包邮吗"})
    assert [json.loads(frame[6:]) for frame in _frames(response)[:-1]] == [
        {"event": "citations", "items": [citation]},
        {"delta": "满99元包邮[1]"},
        {"event": "done", "conversation_id": 12},
    ]


@pytest.mark.parametrize("failure,expected", [
    (agent.ConversationNotFound(99), "会话不存在"),
    (agent.ContextBudgetExceeded("too big"), "上下文预算"),
    (ValueError("other problem"), "上游模型暂时不可用"),
    (ToolInfrastructureError("timeout"), "数据库暂时不可用"),
    (OperationalError("select", {}, Exception("secret")), "数据库暂时不可用"),
    (ConnectionError("secret"), "数据库暂时不可用"),
    (OSError("secret"), "数据库暂时不可用"),
    (RuntimeError("secret"), "上游模型暂时不可用"),
])
def test_stream_failure_after_tool_status_has_no_done(monkeypatch, failure, expected):
    async def producer(*args, **kwargs):
        yield {"type": "tool", "name": "create_ticket"}
        raise failure

    with _client(monkeypatch, producer) as client:
        response = client.post("/api/chat", json={"user_id": "u1", "message": "投诉"})
    frames = _frames(response)
    assert json.loads(frames[0][6:]) == {"event": "tool", "name": "create_ticket"}
    assert frames[1].startswith("event: error\ndata: ")
    assert expected in frames[1]
    assert "[DONE]" not in response.text
    assert "secret" not in response.text


def test_chat_invalid_request_422(monkeypatch):
    async def never(*args, **kwargs):
        raise AssertionError("invalid request reached producer")
        yield

    with _client(monkeypatch, never) as client:
        assert client.post("/api/chat", json={"user_id": "u1", "message": " "}).status_code == 422
        assert client.post("/api/chat", json={"message": "hello"}).status_code == 422
        assert client.post("/api/chat", json={"user_id": "u1", "message": "hi", "conversation_id": -1}).status_code == 422
        assert client.post("/api/chat", json={"user_id": "u" * 65, "message": "hi"}).status_code == 422
