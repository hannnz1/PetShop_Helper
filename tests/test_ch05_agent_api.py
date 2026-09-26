"""The JSON endpoint exposes graph results and never performs a write action."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.graph.runtime import ConversationBusy, ConversationNotFound, GraphDivergence
from app.main import create_app


def _client(tmp_path):
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test", graph_checkpoint_path=str(tmp_path / "graph.sqlite"))
    return TestClient(create_app(settings=settings, model=object()))


def test_agent_reads_graph_result_and_suggests_action(tmp_path):
    with _client(tmp_path) as client:
        async def result(user_id, message, conversation_id, *, model):
            assert (user_id, message, conversation_id) == ("u1", "我要投诉", None)
            return {"conversation_id": 12, "answer": "很抱歉", "tool_calls": [],
                    "tool_results": [], "suggested_actions": [{"type": "create_ticket"}]}

        client.app.state.graph.ainvoke_turn = result
        response = client.post("/api/agent", json={"user_id": "u1", "message": "我要投诉"})
    assert response.status_code == 200
    assert response.json() == {"conversation_id": 12, "answer": "很抱歉", "tool_calls": [],
                              "tool_results": [], "suggested_actions": [{"type": "create_ticket"}]}


def test_agent_maps_graph_ownership_and_busy(tmp_path):
    with _client(tmp_path) as client:
        async def missing(*args, **kwargs):
            raise ConversationNotFound()

        client.app.state.graph.ainvoke_turn = missing
        payload = {"user_id": "u1", "message": "继续", "conversation_id": 10}
        assert client.post("/api/agent", json=payload).status_code == 404

        async def busy(*args, **kwargs):
            raise ConversationBusy()

        client.app.state.graph.ainvoke_turn = busy
        assert client.post("/api/agent", json=payload).status_code == 409

        async def divergent(*args, **kwargs):
            raise GraphDivergence()

        client.app.state.graph.ainvoke_turn = divergent
        assert client.post("/api/agent", json=payload).status_code == 503
