"""SSE customer-service entry point backed by the shared agent orchestration."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.core.context_budget import ContextBudgetExceeded
from app.core.summarizer import schedule_summary
from app.graph.runtime import (
    ConversationBusy, ConversationNotFound, ConversationPending,
    GraphDivergence, ResumeNotPending,
)
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


async def graph_event_stream(graph_stream, user_id: str) -> AsyncIterator[str]:
    """Translate one prepared graph stream for chat and resume endpoints."""
    completed = False
    completed_conversation_id = None
    completed_layer2_budget = None
    interrupted = False
    interrupt_event = None
    seen_tools: set[str] = set()
    seen_actions: set[str] = set()
    try:
        async for mode, payload in graph_stream:
            if interrupted:
                continue
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") == "final_answer" and isinstance(chunk.content, str) and chunk.content:
                    yield _sse({"delta": chunk.content})
                continue
            if mode != "updates":
                continue
            if "__interrupt__" in payload:
                interrupt_value = payload["__interrupt__"][0].value
                interrupt_event = {"event": "interrupt", "kind": interrupt_value.get("type", ""),
                                   "conversation_id": interrupt_value.get("conversation_id")}
                for field in ("orders", "preview"):
                    if field in interrupt_value:
                        interrupt_event[field] = interrupt_value[field]
                interrupted = True
                continue
            for node, update in payload.items():
                if node == "agent_tools":
                    for index, run in enumerate(update.get("tool_results", [])):
                        identifier = run.get("tool_call_id") or f"index:{index}"
                        if identifier in seen_tools:
                            continue
                        seen_tools.add(identifier)
                        yield _sse({"event": "tool", "name": run["name"]})
                    for action in update.get("suggested_actions", []):
                        marker = json.dumps(action, ensure_ascii=False, sort_keys=True)
                        if marker not in seen_actions:
                            seen_actions.add(marker)
                            yield _sse({"event": "actions", "items": [action]})
                elif node in {"forced_rag", "retrieve_policy"} and update.get("citations"):
                    yield _sse({"event": "citations", "items": update["citations"]})
                elif node in {"chitchat_reply", "complaint_reply", "fallback_reply"}:
                    if update.get("answer"):
                        yield _sse({"delta": update["answer"]})
                    if update.get("suggested_actions"):
                        yield _sse({"event": "actions", "items": update["suggested_actions"]})
                elif node == "log_turn":
                    completed = True
                    completed_conversation_id = update["conversation_id"]
                    completed_layer2_budget = (update.get("trace") or {}).get("summary_layer2_budget")
    except ConversationNotFound:
        yield _error("会话不存在")
        return
    except ConversationBusy:
        yield _error("会话正在处理")
        return
    except (ConversationPending, ResumeNotPending):
        yield _error("会话正在等待上一项操作")
        return
    except GraphDivergence:
        yield _error("会话状态需恢复，请开启新对话")
        return
    except ContextBudgetExceeded:
        yield _error("上下文预算不足")
        return
    except agent.ContextBudgetExceeded:
        yield _error("消息超出上下文预算")
        return
    except (ToolInfrastructureError, SQLAlchemyError, ConnectionError, OSError) as exc:
        logger.warning("Chat infrastructure failure type=%s user_id=%s", type(exc).__name__, user_id)
        yield _error("数据库暂时不可用，请稍后重试")
        return
    except Exception:
        logger.warning("Chat model or orchestration failure user_id=%s", user_id)
        yield _error("上游模型暂时不可用，请稍后重试")
        return
    finally:
        await graph_stream.aclose()
    if interrupted:
        yield _sse(interrupt_event)
        return
    if completed:
        yield _sse({"event": "done", "conversation_id": completed_conversation_id})
        yield "data: [DONE]\n\n"
        if completed_layer2_budget is not None:
            schedule_summary(completed_conversation_id,
                             layer2_token_limit=completed_layer2_budget)
        else:
            schedule_summary(completed_conversation_id)
    else:
        yield _error("上游模型暂时不可用，请稍后重试")


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request,
               model: BaseChatModel = Depends(get_model)) -> StreamingResponse:
    try:
        graph_stream = await request.app.state.graph.prepare_stream_turn(
            req.user_id, req.message, req.conversation_id, model=model,
        )
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在") from None
    except ConversationBusy:
        raise HTTPException(status_code=409, detail="会话正在处理上一条消息") from None
    except ConversationPending:
        raise HTTPException(status_code=409, detail="请先完成当前确认") from None
    except GraphDivergence:
        raise HTTPException(status_code=503, detail="会话状态需恢复，请开启新对话") from None
    except (ToolInfrastructureError, SQLAlchemyError, ConnectionError, OSError):
        logger.warning("Chat preflight database failure user_id=%s", req.user_id)
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试") from None

    return StreamingResponse(
        graph_event_stream(graph_stream, req.user_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
