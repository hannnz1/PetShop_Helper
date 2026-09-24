import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.config import Settings
from app.main import create_app


def config(token_budget: int = 2000) -> Settings:
    return Settings(
        _env_file=None,
        chat_model="test-model",
        chat_base_url="https://example.test/v1",
        chat_api_key="private-key",
        token_budget=token_budget,
    )


class StreamingFake:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.calls = []

    async def astream(self, messages):
        self.calls.append(deepcopy(messages))
        for chunk in next(self.turns):
            if isinstance(chunk, Exception):
                raise chunk
            yield AIMessageChunk(content=chunk)


def frames(response):
    return [frame for frame in response.text.split("\n\n") if frame]


def deltas(response):
    return [json.loads(frame[6:])["delta"] for frame in frames(response) if frame.startswith("data: {")]


def test_sse_forwards_each_chunk_and_done():
    model = StreamingFake([["你", "好", "呀"]])
    app = create_app(settings=config(), model=model)
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"session_id": "s1", "message": "在吗"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert frames(response) == [
        'data: {"delta": "你"}',
        'data: {"delta": "好"}',
        'data: {"delta": "呀"}',
        "data: [DONE]",
    ]


def test_second_turn_sends_actual_prior_messages_and_isolates_sessions():
    model = StreamingFake([["第一轮答"], ["第二轮答"], ["另一会话"]])
    app = create_app(settings=config(), model=model)
    with TestClient(app) as client:
        for session, message in [("s", "第一问"), ("s", "第二问"), ("other", "独立问")]:
            response = client.post("/api/chat", json={"session_id": session, "message": message})
            assert response.status_code == 200
            assert frames(response)[-1] == "data: [DONE]"
    assert [m.content for m in model.calls[1][1:]] == ["第一问", "第一轮答", "第二问"]
    assert [m.content for m in model.calls[2][1:]] == ["独立问"]
    assert [m.content for m in app.state.store.get("s")] == [
        "第一问", "第一轮答", "第二问", "第二轮答"
    ]
    assert [m.content for m in app.state.store.get("other")] == ["独立问", "另一会话"]


def test_validation_and_budget_reject_before_model_call():
    model = StreamingFake([["unused"]])
    app = create_app(settings=config(token_budget=500), model=model)
    with TestClient(app) as client:
        assert client.post("/api/chat", json={"session_id": "s", "message": " "}).status_code == 422
        assert client.post("/api/chat", json={"session_id": "s", "message": "长" * 600}).status_code == 422
    assert model.calls == []
    assert app.state.store.get("s") == []


def test_partial_upstream_error_is_atomic_and_does_not_commit_history():
    model = StreamingFake([["部分", RuntimeError("private-key upstream body")], ["恢复"]])
    app = create_app(settings=config(), model=model)
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"session_id": "s", "message": "问"})
        assert response.status_code == 200
        assert frames(response)[0] == 'data: {"delta": "部分"}'
        assert len(frames(response)) == 2
        assert frames(response)[1].startswith("event: error\ndata: ")
        assert "private-key" not in response.text
        assert "[DONE]" not in response.text
        assert app.state.store.get("s") == []
        retry = client.post("/api/chat", json={"session_id": "s", "message": "再问"})
        assert frames(retry)[-1] == "data: [DONE]"
    assert [m.content for m in app.state.store.get("s")] == ["再问", "恢复"]


def test_empty_upstream_response_is_error_without_history():
    app = create_app(settings=config(), model=StreamingFake([["", ""]]))
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"session_id": "s", "message": "问"})
    assert len(frames(response)) == 1
    assert frames(response)[0].startswith("event: error\ndata: ")
    assert app.state.store.get("s") == []


def test_content_blocks_forward_text_without_empty_frames():
    model = StreamingFake([[[{"type": "text", "text": "文字"}], ""]])
    app = create_app(settings=config(), model=model)
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"session_id": "s", "message": "问"})
    assert deltas(response) == ["文字"]
    assert frames(response)[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_same_session_conflict_and_cancellation_release():
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingModel:
        async def astream(self, messages):
            entered.set()
            await release.wait()
            yield AIMessageChunk(content="完成")

    app = create_app(settings=config(), model=BlockingModel())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/chat", json={"session_id": "s", "message": "一"}))
            await asyncio.wait_for(entered.wait(), 2)
            busy = await client.post("/api/chat", json={"session_id": "s", "message": "二"})
            assert busy.status_code == 409
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            await asyncio.sleep(0)
            assert app.state.active_sessions == set()
            release.set()
            retry = await client.post("/api/chat", json={"session_id": "s", "message": "三"})
            assert retry.status_code == 200
            assert app.state.active_sessions == set()
            assert [m.content for m in app.state.store.get("s")] == ["三", "完成"]


def test_lifespan_isolated_state_root_health_and_injected_model_not_closed():
    model = StreamingFake([["ok"]])
    one = create_app(settings=config(), model=model)
    two = create_app(settings=config(), model=StreamingFake([["else"]]))
    with TestClient(one) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "text/html" in client.get("/").headers["content-type"]
        assert client.post("/api/chat", json={"session_id": "x", "message": "hi"}).status_code == 200
    with TestClient(two):
        assert two.state.store.get("x") == []


def test_missing_settings_fails_lifespan_with_safe_message(monkeypatch):
    import app.main as main

    def missing():
        raise RuntimeError("private-key should never escape")

    monkeypatch.setattr(main, "get_settings", missing)
    with pytest.raises(RuntimeError, match="CHAT_MODEL") as error:
        with TestClient(create_app()):
            pass
    assert "private-key" not in str(error.value)


def test_lifespan_closes_only_owned_model_clients(monkeypatch):
    import app.main as main

    class AsyncClient:
        closed = False

        async def close(self):
            self.closed = True

    class SyncClient:
        closed = False

        def close(self):
            self.closed = True

    class Owned:
        root_async_client = AsyncClient()
        root_client = SyncClient()

        async def astream(self, messages):
            raise AssertionError("startup must not call model")
            yield

    owned = Owned()
    monkeypatch.setattr(main, "get_chat_model", lambda **kwargs: owned)
    with TestClient(create_app(settings=config())) as client:
        assert client.get("/health").status_code == 200
    assert owned.root_async_client.closed
    assert owned.root_client.closed
