"""Feedback API is scoped to the caller's conversation."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.db import repository


@pytest.mark.asyncio
async def test_unresolved_feedback_endpoint_returns_same_id_on_retry(db_session_factory, db_clean):
    from app.api.flywheel import router

    conversation_id = await repository.create_conversation("alice")
    await repository.append_message(conversation_id, "user", "没解决的运费问题")
    answer_id = await repository.append_message(conversation_id, "assistant", "抱歉")
    app = FastAPI()
    app.include_router(router)
    payload = {
        "user_id": "alice", "conversation_id": conversation_id,
        "assistant_message_id": answer_id,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/api/feedback/unresolved", json=payload)
        second = await client.post("/api/feedback/unresolved", json=payload)
        foreign = await client.post(
            "/api/feedback/unresolved", json={**payload, "user_id": "bob"},
        )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert foreign.status_code == 404
