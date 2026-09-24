"""Streaming customer-service conversation endpoint."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from starlette.types import Receive, Scope, Send

from app.config import Settings
from app.core.memory import build_context, estimate_tokens, trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()
_ERROR_FRAME = 'event: error\ndata: {"message": "上游模型暂时不可用，请稍后重试"}\n\n'


class _SessionStreamingResponse(StreamingResponse):
    def __init__(self, *args, active_sessions: set[str], session_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_sessions = active_sessions
        self._session_id = session_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Response startup can fail before its body generator is entered.
            self._active_sessions.discard(self._session_id)


def get_model(request: Request) -> BaseChatModel:
    """Return the shared model, overridable through FastAPI dependencies."""
    return request.app.state.model


def get_config(request: Request) -> Settings:
    return request.app.state.settings


def _chunk_text(chunk: object) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    text = getattr(chunk, "text", "")
    if isinstance(text, str):
        return text
    if callable(text):
        text = text()
    return text if isinstance(text, str) else ""


@router.post("/api/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    model: BaseChatModel = Depends(get_model),
    config: Settings = Depends(get_config),
) -> StreamingResponse:
    system = CUSTOMER_SERVICE_PROMPT.format_messages(history=[])[0]
    previous = request.app.state.store.get(req.session_id)
    try:
        messages = build_context(
            system,
            previous,
            HumanMessage(content=req.message),
            config.token_budget,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="消息超出上下文预算") from None

    active = request.app.state.active_sessions
    if req.session_id in active:
        raise HTTPException(status_code=409, detail="会话正在生成回复")
    active.add(req.session_id)

    async def event_stream() -> AsyncIterator[str]:
        chunks: list[str] = []
        try:
            async for chunk in model.astream(messages):
                text = _chunk_text(chunk)
                if text:
                    chunks.append(text)
                    yield f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"
            if not chunks:
                raise ValueError("empty upstream response")
        except Exception:
            # An upstream exception may contain credentials or response bodies.
            logger.warning("Chat upstream stream failed")
            yield _ERROR_FRAME
            return

        complete = [*messages[1:], AIMessage(content="".join(chunks))]
        history_budget = config.token_budget + config.chat_max_tokens - estimate_tokens([system])
        yield "data: [DONE]\n\n"
        request.app.state.store.replace(
            req.session_id,
            trim_history(complete, max_tokens=max(history_budget, 0)),
        )

    return _SessionStreamingResponse(
        event_stream(),
        active_sessions=active,
        session_id=req.session_id,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
