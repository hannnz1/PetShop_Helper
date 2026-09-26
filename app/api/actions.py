"""Explicit, user-confirmed write actions."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.db import repository

router = APIRouter()


class CreateTicketRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int = Field(gt=0)
    ticket_type: str = Field(pattern="^(售后|投诉|咨询)$")
    description: str = Field(min_length=1, max_length=20_000)
    request_id: str = Field(min_length=1, max_length=64)


class CreateTicketResponse(BaseModel):
    ticket_no: str


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket(req: CreateTicketRequest) -> CreateTicketResponse:
    conversation = await repository.get_conversation(req.conversation_id)
    if conversation is None or conversation.user_id != req.user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    try:
        ticket_no = await repository.create_ticket_only(
            req.conversation_id, req.description, req.ticket_type, req.request_id,
        )
    except repository.TicketRequestConflict:
        raise HTTPException(status_code=409, detail="请求标识已用于其他工单") from None
    return CreateTicketResponse(ticket_no=ticket_no)
