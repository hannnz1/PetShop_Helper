"""SSE interrupt and resume contract on the shared chat graph."""

import asyncio
import json

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db import repository
from app.main import create_app


class RefundClassifier:
    async def classify(self, query):
        return "退款退货"


def _frames(response):
    return [json.loads(chunk[6:]) for chunk in response.text.split("\n\n")
            if chunk.startswith("data: {")]


async def _seed(factory):
    async with factory.begin() as session:
        await session.execute(text("""
            INSERT INTO sample_orders (order_id,user_id,status,product,amount)
            VALUES ('1001','alice','已签收','猫粮',88.00),
                   ('2001','bob','已签收','猫砂',39.00)
        """))


def test_chat_interrupt_then_resume_streams_and_preflights(tmp_path, db_session_factory, db_clean, monkeypatch):
    from app.graph import nodes
    import app.main as main_module

    async def weak_policy(state, runtime):
        return {"sufficient": False, "evidence": "", "citations": [], "reason": "无政策"}

    monkeypatch.setattr(nodes, "retrieve_policy", weak_policy)
    monkeypatch.setattr(main_module, "ModelIntentClassifier", lambda *args, **kwargs: RefundClassifier())
    asyncio.run(_seed(db_session_factory))
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", graph_checkpoint_path=str(tmp_path / "api.sqlite"))
    with TestClient(create_app(settings=settings, model=object())) as client:
        first = client.post("/api/chat", json={"user_id": "alice", "message": "能退吗"})
        assert first.status_code == 200
        interrupt = next(item for item in _frames(first) if item.get("event") == "interrupt")
        cid = interrupt["conversation_id"]
        assert interrupt["kind"] == "select_order"
        assert [row["order_id"] for row in interrupt["orders"]] == ["1001"]
        assert not any(item.get("event") == "done" for item in _frames(first))

        pending = client.post("/api/chat", json={"user_id": "alice", "conversation_id": cid,
                                                  "message": "新问题"})
        assert pending.status_code == 409
        foreign = client.post("/api/actions/resume", json={"user_id": "intruder",
                                                           "conversation_id": cid, "order_id": "1001"})
        assert foreign.status_code == 404
        resumed = client.post("/api/actions/resume", json={"user_id": "alice",
                                                           "conversation_id": cid, "order_id": "1001"})
        assert resumed.status_code == 200
        assert any(item.get("event") == "done" for item in _frames(resumed))
        repeated = client.post("/api/actions/resume", json={"user_id": "alice",
                                                            "conversation_id": cid, "order_id": "1001"})
        assert repeated.status_code == 409


def test_resume_without_checkpoint_does_not_replay_old_messages(tmp_path, db_session_factory, db_clean):
    cid = asyncio.run(repository.create_conversation("alice"))
    asyncio.run(repository.append_message(cid, "user", content="旧记录"))
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", graph_checkpoint_path=str(tmp_path / "lost.sqlite"))
    with TestClient(create_app(settings=settings, model=object())) as client:
        response = client.post("/api/actions/resume", json={"user_id": "alice",
                                                           "conversation_id": cid, "order_id": "1001"})
    assert response.status_code == 503


def test_json_agent_exposes_order_choice_for_interrupt(tmp_path, db_session_factory, db_clean, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "ModelIntentClassifier", lambda *args, **kwargs: RefundClassifier())
    asyncio.run(_seed(db_session_factory))
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", graph_checkpoint_path=str(tmp_path / "agent.sqlite"))
    with TestClient(create_app(settings=settings, model=object())) as client:
        response = client.post("/api/agent", json={"user_id": "alice", "message": "能退吗"})
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] > 0
    assert data["answer"] == ""
    assert data["suggested_actions"][0]["type"] == "select_order"
    assert [row["order_id"] for row in data["suggested_actions"][0]["orders"]] == ["1001"]
