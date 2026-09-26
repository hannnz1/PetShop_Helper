"""JSON agent endpoint for clients needing a complete tool trace."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.language_models import BaseChatModel
from sqlalchemy.exc import SQLAlchemyError

from app.api.chat import get_model
from app.core import agent
from app.graph.runtime import ConversationBusy, ConversationNotFound
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView
from app.tools.infra import ToolInfrastructureError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest, request: Request,
                    model: BaseChatModel = Depends(get_model)) -> AgentResponse:
    try:
        result = await request.app.state.graph.ainvoke_turn(
            req.user_id, req.message, req.conversation_id, model=model
        )
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在") from None
    except ConversationBusy:
        raise HTTPException(status_code=409, detail="会话正在处理上一条消息") from None
    except agent.ContextBudgetExceeded:
        raise HTTPException(status_code=422, detail="消息超出上下文预算") from None
    except (ToolInfrastructureError, SQLAlchemyError, ConnectionError, OSError):
        logger.warning("Agent database service failure user_id=%s", req.user_id)
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试") from None
    except Exception:
        logger.warning("Agent model or orchestration failure user_id=%s", req.user_id)
        raise HTTPException(status_code=502, detail="上游模型暂时不可用，请稍后重试") from None

    return AgentResponse(
        conversation_id=result["conversation_id"],
        answer=result["answer"],
        tool_calls=[ToolCallView(id=call["id"], name=call["name"], args=call["args"])
                    for call in result.get("tool_calls", [])],
        tool_results=[ToolResultView(**run) for run in result.get("tool_results", [])],
        suggested_actions=result.get("suggested_actions", []),
    )
