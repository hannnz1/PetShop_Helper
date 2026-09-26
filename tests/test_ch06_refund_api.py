"""Only an explicit user confirmation creates a pending refund application."""

import asyncio

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db import repository
from app.main import create_app


async def _seed(factory):
    async with factory.begin() as session:
        await session.execute(text("""
            INSERT INTO sample_orders (order_id,user_id,status,product,amount)
            VALUES ('1001','alice','已签收','猫粮',88.00),
                   ('1002','alice','已发货','饮水机',139.00),
                   ('2001','bob','已签收','猫砂',39.00)
        """))


def _client():
    settings = Settings(_env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
                        chat_api_key="test")
    return TestClient(create_app(settings=settings, model=object()))


def test_confirm_refund_is_idempotent_and_changes_no_other_state(db_session_factory, db_clean):
    asyncio.run(_seed(db_session_factory))
    cid = asyncio.run(repository.create_conversation("alice"))
    payload = {"user_id": "alice", "conversation_id": cid, "order_id": "1001",
               "reason": "质量问题", "request_id": "confirm-1"}
    with _client() as client:
        first = client.post("/api/actions/create-refund", json=payload)
        retry = client.post("/api/actions/create-refund", json=payload)
        changed_reason = client.post("/api/actions/create-refund", json={**payload, "reason": "其他"})
        changed_order = client.post("/api/actions/create-refund", json={**payload, "order_id": "1002"})
    assert first.status_code == retry.status_code == 200
    assert first.json() == retry.json()
    assert first.json()["status"] == "待人工审核"
    assert first.json()["refund_no"].startswith("R")
    assert changed_reason.status_code == changed_order.status_code == 409

    async def rows():
        async with db_session_factory() as session:
            refunds = (await session.execute(text("SELECT refund_no,status FROM refund_requests"))).all()
            orders = (await session.execute(text("SELECT order_id,status FROM sample_orders ORDER BY order_id"))).all()
            tickets = (await session.execute(text("SELECT COUNT(*) FROM tickets"))).scalar_one()
        return refunds, orders, tickets

    refunds, orders, tickets = asyncio.run(rows())
    assert refunds == [(first.json()["refund_no"], "待人工审核")]
    assert orders == [("1001", "已签收"), ("1002", "已发货"), ("2001", "已签收")]
    assert tickets == 0


def test_refund_rejects_foreign_conversation_order_and_bad_reason(db_session_factory, db_clean):
    asyncio.run(_seed(db_session_factory))
    cid = asyncio.run(repository.create_conversation("alice"))
    payload = {"user_id": "alice", "conversation_id": cid, "order_id": "1001",
               "reason": "质量问题", "request_id": "confirm-2"}
    with _client() as client:
        assert client.post("/api/actions/create-refund", json={**payload, "user_id": "bob"}).status_code == 404
        assert client.post("/api/actions/create-refund", json={**payload, "order_id": "2001"}).status_code == 404
        assert client.post("/api/actions/create-refund", json={**payload, "order_id": "9999"}).status_code == 404
        for reason in ("", "   ", "随便退款", "质量问题 "):
            response = client.post("/api/actions/create-refund", json={**payload, "reason": reason})
            assert response.status_code == 422

    async def count():
        async with db_session_factory() as session:
            return (await session.execute(text("SELECT COUNT(*) FROM refund_requests"))).scalar_one()

    assert asyncio.run(count()) == 0
