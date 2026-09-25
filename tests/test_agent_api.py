"""Programmatic agent endpoint contract and failure classification."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.core import agent
from app.main import create_app
from app.tools.infra import ToolInfrastructureError, ToolRun


def _client():
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test")
    return TestClient(create_app(settings=settings, model=object()))


def test_agent_returns_tool_trace(monkeypatch):
    async def run(user_id, message, conversation_id, model=None):
        assert (user_id, message, conversation_id) == ("u1", "订单 1001 到哪了", None)
        tool = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
        return agent.AgentResult(12, "演示数据：运输中", [{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}], [ToolRun("c1", "query_logistics", True, tool)])

    monkeypatch.setattr(agent, "run_agent_turn", run)
    with _client() as client:
        response = client.post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": 12, "answer": "演示数据：运输中",
        "tool_calls": [{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}],
        "tool_results": [{"tool_call_id": "c1", "name": "query_logistics", "ok": True, "content": '{"status":"运输中"}'}],
    }


@pytest.mark.parametrize("failure,code,detail", [
    (agent.ConversationNotFound(999), 404, "会话不存在"),
    (agent.ContextBudgetExceeded("too big"), 422, "上下文预算"),
    (ValueError("other problem"), 502, "上游模型暂时不可用"),
    (ToolInfrastructureError("timeout"), 503, "数据库暂时不可用"),
    (OperationalError("select", {}, Exception("secret")), 503, "数据库暂时不可用"),
    (ConnectionError("secret"), 503, "数据库暂时不可用"),
    (OSError("secret"), 503, "数据库暂时不可用"),
    (RuntimeError("secret"), 502, "上游模型暂时不可用"),
])
def test_agent_error_mapping(monkeypatch, failure, code, detail):
    async def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(agent, "run_agent_turn", run)
    with _client() as client:
        response = client.post("/api/agent", json={"user_id": "u1", "message": "hello", "conversation_id": 999})
    assert response.status_code == code
    assert detail in response.json()["detail"]
    assert "secret" not in response.text


def test_agent_invalid_request_422():
    with _client() as client:
        assert client.post("/api/agent", json={"message": "hi"}).status_code == 422
        assert client.post("/api/agent", json={"user_id": "u" * 65, "message": "hi"}).status_code == 422
