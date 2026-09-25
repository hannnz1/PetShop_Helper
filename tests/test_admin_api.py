"""The admin landing page isolates knowledge dependency outages."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin


def test_admin_keeps_chat_and_agent_cards_when_knowledge_is_offline(monkeypatch):
    async def offline(_request):
        return {
            "mysql": {"available": False, "stats": None},
            "milvus": {"available": False, "count": None},
            "consistent": None, "staging": None, "sources": [], "jobs": [],
        }

    monkeypatch.setattr(admin.kb, "overview", offline)
    app = FastAPI()
    app.include_router(admin.router)
    with TestClient(app) as client:
        response = client.get("/api/admin/overview")
    assert response.status_code == 200
    cards = {item["id"]: item for item in response.json()["modules"]}
    assert {"chat", "agent", "kb"} <= set(cards)
    assert cards["kb"]["status"] == "unavailable"
    assert cards["chat"]["status"] != "unavailable"
    assert cards["kb"]["href"] == "/kb"
