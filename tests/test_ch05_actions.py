"""Chapter five ticket action contract, using the isolated MySQL test schema."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings
from app.db import repository
from app.db.models import Conversation, Ticket
from app.main import create_app


@pytest.mark.asyncio
async def test_ticket_retry_is_idempotent_and_does_not_handoff(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("owner")
    first = await repository.create_ticket_only(conversation_id, "请处理退款", "售后", "client-1")
    again = await repository.create_ticket_only(conversation_id, "请处理退款", "售后", "client-1")
    assert first == again
    async with db_session_factory() as session:
        tickets = (await session.scalars(select(Ticket))).all()
        conversation = await session.get(Conversation, conversation_id)
    assert len(tickets) == 1
    assert conversation.status == "进行中"


@pytest.mark.asyncio
async def test_ticket_retry_changed_payload_conflicts(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("owner")
    await repository.create_ticket_only(conversation_id, "原请求", "投诉", "client-2")
    with pytest.raises(repository.TicketRequestConflict):
        await repository.create_ticket_only(conversation_id, "变更请求", "投诉", "client-2")


@pytest.mark.asyncio
async def test_concurrent_ticket_retry_creates_one_row(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("owner")
    results = await asyncio.gather(*[
        repository.create_ticket_only(conversation_id, "并发提交", "咨询", "client-3")
        for _ in range(4)
    ])
    assert len(set(results)) == 1
    async with db_session_factory() as session:
        assert len((await session.scalars(select(Ticket))).all()) == 1


def test_action_rejects_foreign_owner_and_blank_content(monkeypatch):
    from app.api import actions

    async def foreign(_conversation_id):
        return Conversation(id=10, user_id="another")

    monkeypatch.setattr(actions.repository, "get_conversation", foreign)
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test")
    with TestClient(create_app(settings=settings, model=object())) as client:
        payload = {"user_id": "owner", "conversation_id": 10, "ticket_type": "投诉",
                   "description": "问题", "request_id": "client-4"}
        assert client.post("/api/actions/create-ticket", json=payload).status_code == 404
        assert client.post("/api/actions/create-ticket", json={**payload, "description": "   "}).status_code == 422


@pytest.mark.asyncio
async def test_action_returns_same_number_then_conflict(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("owner")
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1", chat_api_key="test")
    payload = {"user_id": "owner", "conversation_id": conversation_id,
               "ticket_type": "售后", "description": "请处理退款", "request_id": "client-api"}
    with TestClient(create_app(settings=settings, model=object())) as client:
        first = client.post("/api/actions/create-ticket", json=payload)
        retry = client.post("/api/actions/create-ticket", json=payload)
        changed = client.post("/api/actions/create-ticket", json={**payload, "description": "其他问题"})
    assert first.status_code == retry.status_code == 200
    assert first.json() == retry.json()
    assert changed.status_code == 409
