"""Repository integration checks against the isolated Docker MySQL schema."""

import pytest
from sqlalchemy.exc import DataError, IntegrityError

from app.db import repository as repo
from app.db.models import Conversation, Faq, Ticket


@pytest.mark.asyncio
async def test_create_and_get_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1" and conv.status == "进行中"
    assert await repo.get_conversation(cid + 999999) is None


@pytest.mark.asyncio
async def test_append_and_list_messages_keep_order_and_tool_ids(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    calls = [{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]
    ids = [
        await repo.append_message(cid, "user", content="订单 1001 到哪了"),
        await repo.append_message(cid, "assistant", tool_calls=calls),
        await repo.append_message(cid, "tool", content='{"status":"运输中"}', tool_call_id="c1"),
    ]
    msgs = await repo.list_messages(cid)
    assert [msg.id for msg in msgs] == ids
    assert [msg.role for msg in msgs] == ["user", "assistant", "tool"]
    assert msgs[1].tool_calls == calls and msgs[1].content is None
    assert msgs[2].tool_call_id == "c1" and msgs[2].content == '{"status":"运输中"}'


@pytest.mark.asyncio
async def test_search_faq_literal_like_hit_and_miss(db_session_factory, db_clean):
    async with db_session_factory() as session:
        session.add_all(
            [
                Faq(question="退货政策", answer="7 天无理由退货", category="售后"),
                Faq(question="运费怎么算", answer="按地址计算", category="物流"),
                Faq(question="折扣 20%_优惠", answer="仅活动期", category="活动"),
            ]
        )
        await session.commit()
    assert [faq.question for faq in await repo.search_faq("退货政策")] == ["退货政策"]
    assert await repo.search_faq("邮费") == []
    assert [faq.question for faq in await repo.search_faq("20%_")] == ["折扣 20%_优惠"]
    assert await repo.search_faq("%无匹配") == []


@pytest.mark.asyncio
async def test_search_faq_caps_matching_rows(db_session_factory, db_clean):
    async with db_session_factory() as session:
        session.add_all(
            [Faq(question=f"退货问题 {number}", answer="联系客户服务", category="售后") for number in range(15)]
        )
        await session.commit()
    rows = await repo.search_faq("退货")
    assert len(rows) == 10
    assert [row.question for row in rows] == [f"退货问题 {number}" for number in range(10)]


@pytest.mark.asyncio
async def test_create_ticket_writes_and_flips_conversation_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    no = await repo.create_ticket(cid, "要退货", "售后")
    assert no.startswith("T") and len(no) <= 32
    async with db_session_factory() as session:
        ticket = await session.get(Ticket, no)
        conv = await session.get(Conversation, cid)
    assert ticket is not None and ticket.ticket_type == "售后" and ticket.status == "待处理"
    assert conv is not None and conv.status == "已转人工"


@pytest.mark.asyncio
async def test_ticket_numbers_are_distinct_without_process_sequence(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    first = await repo.create_ticket(cid, "问题一", "咨询")
    second = await repo.create_ticket(cid, "问题二", "投诉")
    assert first != second
    async with db_session_factory() as session:
        assert await session.get(Ticket, first) is not None
        assert await session.get(Ticket, second) is not None


@pytest.mark.asyncio
async def test_create_ticket_missing_conversation_writes_nothing(db_session_factory, db_clean):
    with pytest.raises(ValueError, match="conversation"):
        await repo.create_ticket(999999, "没有会话", "咨询")
    async with db_session_factory() as session:
        assert await session.get(Ticket, "Tmissing") is None
        assert (await session.execute(Ticket.__table__.select())).all() == []


@pytest.mark.asyncio
async def test_create_ticket_invalid_type_rolls_back_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    with pytest.raises((DataError, IntegrityError)):
        await repo.create_ticket(cid, "问题", "不合法类型")
    async with db_session_factory() as session:
        conv = await session.get(Conversation, cid)
        tickets = (await session.execute(Ticket.__table__.select())).all()
    assert conv is not None and conv.status == "进行中"
    assert tickets == []
