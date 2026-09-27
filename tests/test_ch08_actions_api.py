"""The resume API passes a confirmed decision without changing order selection."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_resume_api_requires_one_action_and_forwards_ticket_decision():
    settings = Settings(_env_file=None, chat_model="offline-test",
                        chat_base_url="http://127.0.0.1:9/v1", chat_api_key="offline-test",
                        token_budget=32768)
    with TestClient(create_app(settings=settings, model=object())) as client:
        seen = []

        async def prepare(user_id, conversation_id, value, *, model):
            seen.append((user_id, conversation_id, value))

            async def events():
                yield "updates", {"__interrupt__": [type("Interrupt", (), {
                    "value": {"type": "confirm_ticket", "conversation_id": conversation_id,
                              "preview": {"ticket_type": "售后", "description": "漏电"}},
                })()]}

            return events()

        client.app.state.graph.prepare_resume_turn = prepare
        common = {"user_id": "u1", "conversation_id": 1}
        assert client.post("/api/actions/resume", json=common).status_code == 400
        assert client.post("/api/actions/resume", json={**common, "order_id": "1001",
                                                        "confirmed": True}).status_code == 400
        response = client.post("/api/actions/resume", json={**common, "confirmed": False})
        assert response.status_code == 200
        assert '"kind": "confirm_ticket"' in response.text
        assert '"preview"' in response.text
        assert seen == [("u1", 1, {"confirmed": False})]

        response = client.post("/api/actions/resume", json={**common, "order_id": "1001"})
        assert response.status_code == 200
        assert seen[-1] == ("u1", 1, "1001")
