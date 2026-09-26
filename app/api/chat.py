"""SSE customer-service entry point backed by the shared agent orchestration."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.schemas.chat import ChatRequest
from app.tools.infra import ToolInfrastructureError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_model(request: Request) -> BaseChatModel:
    return request.app.state.model


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _error(message: str) -> str:
    return f"event: error\ndata: {json.dumps({'message': message}, ensure_ascii=False)}\n\n"


@router.post("/api/chat")
async def chat(req: ChatRequest, model: BaseChatModel = Depends(get_model)) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        completed = False
        try:
            async for event in agent.stream_agent_turn(
                req.user_id, req.message, req.conversation_id, model=model
            ):
                if event["type"] == "tool":
                    yield _sse({"event": "tool", "name": event["name"]})
                elif event["type"] == "citations":
                    yield _sse({"event": "citations", "items": event["items"]})
                elif event["type"] == "delta":
                    yield _sse({"delta": event["text"]})
                elif event["type"] == "done":
                    completed = True
                    yield _sse({"event": "done", "conversation_id": event["conversation_id"]})
        except agent.ConversationNotFound:
            yield _error("会话不存在")
            return
        except agent.ContextBudgetExceeded:
            yield _error("消息超出上下文预算")
            return
        except (ToolInfrastructureError, SQLAlchemyError, ConnectionError, OSError) as exc:
            logger.warning("Chat infrastructure failure type=%s user_id=%s", type(exc).__name__, req.user_id)
            yield _error("数据库暂时不可用，请稍后重试")
            return
        except Exception:
            # Upstream failures may include secrets or response bodies.
            logger.warning("Chat model or orchestration failure user_id=%s", req.user_id)
            yield _error("上游模型暂时不可用，请稍后重试")
            return
        if completed:
            yield "data: [DONE]\n\n"
        else:
            yield _error("上游模型暂时不可用，请稍后重试")

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
