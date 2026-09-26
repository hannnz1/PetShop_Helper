"""HTTP routes through the real orchestration and isolated MySQL."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.config import Settings
from app.db import repository
from app.main import create_app


class FakeModel:
    def __init__(self):
        self.replies = [
            AIMessage(content="已经查到了", tool_calls=[
                {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call-1"}
            ]),
            AIMessage(content="好的，继续帮您处理"),
        ]
        self.chunks = ["演示记录", "显示物流运输中"]

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return self.replies.pop(0)

    async def astream(self, messages):
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk)


class FakeClassifier:
    async def classify(self, query):
        return "物流" if "订单" in query else "闲聊"


@pytest.mark.asyncio
async def test_chat_and_agent_http_use_real_core_and_mysql(db_session_factory, db_clean, tmp_path):
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test",
                        graph_checkpoint_path=str(tmp_path / "graph.sqlite"))
    with TestClient(create_app(settings=settings, model=FakeModel())) as client:
        client.app.state.graph.classifier = FakeClassifier()
        chat = client.post("/api/chat", json={"user_id": "owner", "message": "订单1001到哪了"})
        assert chat.status_code == 200
        frames = [frame for frame in chat.text.split("\n\n") if frame]
        payloads = [json.loads(frame[6:]) for frame in frames if frame.startswith("data: {")]
        assert payloads[0] == {"event": "tool", "name": "query_logistics"}
        assert "".join(frame["delta"] for frame in payloads if "delta" in frame) == "演示记录显示物流运输中"
        conversation_id = payloads[-1]["conversation_id"]
        assert payloads[-1]["event"] == "done"
        assert frames[-1] == "data: [DONE]"

        forbidden = client.post("/api/agent", json={"user_id": "other", "message": "看历史", "conversation_id": conversation_id})
        assert forbidden.status_code == 404
        continued = client.post("/api/agent", json={"user_id": "owner", "message": "谢谢", "conversation_id": conversation_id})
        assert continued.status_code == 200
        assert continued.json()["conversation_id"] == conversation_id
        assert "您好" in continued.json()["answer"]

    rows = await repository.list_messages(conversation_id)
    assert [row.role for row in rows] == ["user", "tool", "assistant", "user", "assistant"]
    assert rows[2].content == "演示记录显示物流运输中"
