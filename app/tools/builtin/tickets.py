"""Graph ticket write tool with a hidden idempotency key."""

from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from app.tools import registry
from app.db import repository


@tool("create_ticket")
async def create_ticket_confirmed(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
    request_id: Annotated[str, InjectedToolArg],
) -> dict:
    """Only after the user confirms the preview, save a support ticket for follow-up."""

    ticket_no = await repository.create_ticket_only(
        conversation_id, description, ticket_type, request_id,
    )
    return {"ticket_no": ticket_no, "status": "已建单，待人工处理"}

registry.register(registry.spec_from_langchain_tool(
    create_ticket_confirmed, source="builtin", inject_conversation=True,
    inject_request_id=True,
))
