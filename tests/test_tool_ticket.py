"""Ticket tool hides trusted context from the model and persists the handoff."""

import pytest
from langchain_openai import ChatOpenAI
from sqlalchemy import select

from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket


def test_conversation_id_hidden_from_model_but_required_at_execution():
    model_props = create_ticket.tool_call_schema.model_json_schema()["properties"]
    runtime_props = create_ticket.get_input_schema().model_json_schema()["properties"]
    assert set(model_props) == {"description", "ticket_type"}
    assert "conversation_id" not in model_props
    assert "conversation_id" in runtime_props
    assert "conversation_id" in create_ticket.get_input_schema().model_fields


def test_bind_tools_request_omits_conversation_id():
    model = ChatOpenAI(model="test-only", api_key="test-only", base_url="http://localhost:9999/v1")
    bound = model.bind_tools([create_ticket])
    assert len(bound.kwargs["tools"]) == 1
    props = bound.kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert set(props) == {"description", "ticket_type"}


@pytest.mark.asyncio
async def test_create_ticket_injects_conversation_id_and_flips_status(db_session_factory, db_clean):
    async with db_session_factory() as session:
        conversation = Conversation(user_id="ticket-user")
        session.add(conversation)
        await session.commit()
        cid = conversation.id

    result = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )

    assert result["ticket_no"].startswith("T")
    assert result["status"] == "已转人工"
    async with db_session_factory() as session:
        ticket = await session.get(Ticket, result["ticket_no"])
        conversation = await session.get(Conversation, cid)
    assert ticket is not None
    assert ticket.conversation_id == cid
    assert ticket.description == "商品损坏要退货"
    assert ticket.ticket_type == "售后"
    assert conversation.status == "已转人工"


@pytest.mark.asyncio
async def test_create_ticket_missing_conversation_does_not_write(db_session_factory, db_clean):
    with pytest.raises(ValueError, match="does not exist"):
        await create_ticket.ainvoke(
            {"description": "无法关联", "ticket_type": "咨询", "conversation_id": 999999}
        )

    async with db_session_factory() as session:
        assert list((await session.scalars(select(Ticket))).all()) == []
