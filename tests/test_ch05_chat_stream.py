"""Only final user-facing text may leave the graph as SSE deltas."""

import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.config import Settings
from app.graph.runtime import ConversationBusy, ConversationNotFound, GraphDivergence
from app.main import create_app


def _install_stream(client, producer):
    async def prepare(*args, **kwargs):
        return producer(*args, **kwargs)
    client.app.state.graph.prepare_stream_turn = prepare


def test_stream_filters_internal_chunks_and_sends_actions(tmp_path):
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", token_budget=32768, graph_checkpoint_path=str(tmp_path / "graph.sqlite"))
    with TestClient(create_app(settings=settings, model=object())) as client:
        async def stream(*args, **kwargs):
            yield "messages", (AIMessageChunk(content="隐秘分类"), {"langgraph_node": "classify_intent_node"})
            yield "messages", (AIMessageChunk(content="内部规划"), {"langgraph_node": "agent_llm"})
            yield "updates", {"agent_tools": {"tool_results": [{"name": "query_order", "ok": True}]}}
            yield "messages", (AIMessageChunk(content="您好"), {"langgraph_node": "final_answer"})
            yield "messages", (AIMessageChunk(content="！"), {"langgraph_node": "final_answer"})
            yield "updates", {"complaint_reply": {"suggested_actions": [{"type": "create_ticket"}]}}
            yield "updates", {"log_turn": {"conversation_id": 7}}

        _install_stream(client, stream)
        response = client.post("/api/chat", json={"user_id": "u1", "message": "帮我查订单"})
    payloads = [json.loads(frame[6:]) for frame in response.text.split("\n\n") if frame.startswith("data: {")]
    assert payloads == [
        {"event": "tool", "name": "query_order"},
        {"delta": "您好"}, {"delta": "！"},
        {"event": "actions", "items": [{"type": "create_ticket"}]},
        {"event": "done", "conversation_id": 7},
    ]
    assert "内部规划" not in response.text and "隐秘分类" not in response.text


def test_stream_rejects_owner_and_busy_before_http_starts(tmp_path):
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", token_budget=32768, graph_checkpoint_path=str(tmp_path / "graph.sqlite"))
    with TestClient(create_app(settings=settings, model=object())) as client:
        async def missing(*args, **kwargs):
            raise ConversationNotFound()

        client.app.state.graph.prepare_stream_turn = missing
        payload = {"user_id": "owner", "message": "继续", "conversation_id": 7}
        assert client.post("/api/chat", json=payload).status_code == 404

        async def busy(*args, **kwargs):
            raise ConversationBusy()

        client.app.state.graph.prepare_stream_turn = busy
        assert client.post("/api/chat", json=payload).status_code == 409

        async def divergent(*args, **kwargs):
            raise GraphDivergence()

        client.app.state.graph.prepare_stream_turn = divergent
        assert client.post("/api/chat", json=payload).status_code == 503


def test_multistep_tool_status_is_not_repeated(tmp_path):
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", token_budget=32768, graph_checkpoint_path=str(tmp_path / "graph.sqlite"))
    with TestClient(create_app(settings=settings, model=object())) as client:
        async def stream(*args, **kwargs):
            yield "updates", {"agent_tools": {"tool_results": [{"tool_call_id": "c1", "name": "query_order"}]}}
            yield "updates", {"agent_tools": {"tool_results": [
                {"tool_call_id": "c1", "name": "query_order"},
                {"tool_call_id": "c2", "name": "query_logistics"}]}}
            yield "updates", {"log_turn": {"conversation_id": 9}}

        _install_stream(client, stream)
        response = client.post("/api/chat", json={"user_id": "u", "message": "查订单"})
    payloads = [json.loads(frame[6:]) for frame in response.text.split("\n\n") if frame.startswith("data: {")]
    assert [frame["name"] for frame in payloads if frame.get("event") == "tool"] == [
        "query_order", "query_logistics",
    ]
