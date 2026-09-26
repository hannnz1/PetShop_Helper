"""Persistent demo order ownership and confirmed refund request semantics."""

import asyncio

import pytest
from sqlalchemy import text

from app.db import repository


async def _seed_orders(factory):
    async with factory.begin() as session:
        await session.execute(text("""
            INSERT INTO sample_orders (order_id, user_id, status, product, amount)
            VALUES ('1001', 'alice', '已签收', '猫粮', 88.00),
                   ('1002', 'alice', '已发货', '饮水机', 139.00),
                   ('2001', 'bob', '已签收', '猫砂', 39.00)
        """))


@pytest.mark.asyncio
async def test_sample_orders_only_expose_owners_rows(db_session_factory, db_clean):
    await _seed_orders(db_session_factory)
    orders = await repository.list_sample_orders("alice")
    assert [row["order_id"] for row in orders] == ["1001", "1002"]
    assert await repository.get_owned_sample_order("alice", "2001") is None
    assert (await repository.get_owned_sample_order("bob", "2001"))["product"] == "猫砂"
    assert await repository.list_sample_orders("nobody") == []


@pytest.mark.asyncio
async def test_refund_request_is_idempotent_and_keeps_order_unchanged(db_session_factory, db_clean):
    await _seed_orders(db_session_factory)
    conversation_id = await repository.create_conversation("alice")
    first = await repository.create_refund_request(
        conversation_id, "alice", "1001", "质量问题", "refund-key-1",
    )
    again = await repository.create_refund_request(
        conversation_id, "alice", "1001", "质量问题", "refund-key-1",
    )
    assert first == again
    async with db_session_factory() as session:
        rows = (await session.execute(text("SELECT refund_no, status FROM refund_requests"))).all()
        order = (await session.execute(text("SELECT status FROM sample_orders WHERE order_id='1001'"))).scalar_one()
    assert rows == [(first, "待人工审核")]
    assert order == "已签收"


@pytest.mark.asyncio
async def test_changed_idempotency_payload_conflicts(db_session_factory, db_clean):
    await _seed_orders(db_session_factory)
    conversation_id = await repository.create_conversation("alice")
    await repository.create_refund_request(conversation_id, "alice", "1001", "质量问题", "refund-key-2")
    with pytest.raises(repository.RefundRequestConflict):
        await repository.create_refund_request(conversation_id, "alice", "1002", "质量问题", "refund-key-2")


@pytest.mark.asyncio
async def test_foreign_order_cannot_create_refund(db_session_factory, db_clean):
    await _seed_orders(db_session_factory)
    conversation_id = await repository.create_conversation("alice")
    with pytest.raises(repository.OrderNotOwned):
        await repository.create_refund_request(conversation_id, "alice", "2001", "其他", "refund-key-3")
    async with db_session_factory() as session:
        assert (await session.execute(text("SELECT COUNT(*) FROM refund_requests"))).scalar_one() == 0


@pytest.mark.asyncio
async def test_concurrent_retry_creates_one_request(db_session_factory, db_clean):
    await _seed_orders(db_session_factory)
    conversation_id = await repository.create_conversation("alice")
    values = await asyncio.gather(*[
        repository.create_refund_request(conversation_id, "alice", "1001", "其他", "refund-key-4")
        for _ in range(3)
    ])
    assert len(set(values)) == 1


@pytest.mark.asyncio
async def test_migration_recognizes_existing_chapter_tables(_test_engine):
    from scripts.migrate_ch06 import apply_migration

    assert await apply_migration(_test_engine) == "already applied"


@pytest.mark.asyncio
async def test_demo_seed_is_explicit_and_idempotent(db_session_factory, db_clean):
    from scripts.seed_ch06_demo import seed_demo_orders

    assert await repository.list_sample_orders("demo-user") == []
    await seed_demo_orders(db_session_factory)
    await seed_demo_orders(db_session_factory)
    assert [row["order_id"] for row in await repository.list_sample_orders("demo-user")] == ["1001", "1002"]
