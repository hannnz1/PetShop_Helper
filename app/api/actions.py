"""Explicit, user-confirmed write actions."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from app.api.chat import get_model, graph_event_stream
from app.db import repository
from app.graph.runtime import (
    ConversationBusy, ConversationNotFound, ConversationPending,
    GraphDivergence, ResumeNotPending,
)
from app.tools.infra import ToolInfrastructureError

router = APIRouter()


class ResumeOrderRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int = Field(gt=0)
    order_id: str = Field(min_length=1, max_length=64)


@router.post("/api/actions/resume")
async def resume_order(req: ResumeOrderRequest, request: Request,
                       model: BaseChatModel = Depends(get_model)) -> StreamingResponse:
    try:
        graph_stream = await request.app.state.graph.prepare_resume_turn(
            req.user_id, req.conversation_id, req.order_id, model=model,
        )
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在") from None
    except ConversationBusy:
        raise HTTPException(status_code=409, detail="会话正在处理上一项操作") from None
    except (ConversationPending, ResumeNotPending):
        raise HTTPException(status_code=409, detail="当前会话不在等待选单") from None
    except GraphDivergence:
        raise HTTPException(status_code=503, detail="会话状态需恢复，请开启新对话") from None
    except (ToolInfrastructureError, SQLAlchemyError, ConnectionError, OSError):
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试") from None
    return StreamingResponse(
        graph_event_stream(graph_stream, req.user_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
