"""Validate model tool calls, add trusted context, and execute with bounded retries."""

import asyncio
import json
import logging
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.tools import registry

logger = logging.getLogger(__name__)


class ToolInfrastructureError(RuntimeError):
    """A database-backed tool timed out; the API should report service failure."""


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage


def _error_run(tool_call_id: str, name: str, reason: str) -> ToolRun:
    return ToolRun(
        tool_call_id=tool_call_id,
        name=name,
        ok=False,
        tool_message=ToolMessage(
            content=f"工具执行失败:{reason}",
            tool_call_id=tool_call_id,
            name=name,
            status="error",
        ),
    )


async def execute_tool_call(
    tool_call: dict,
    conversation_id: int,
    timeout: float | None = None,
    max_retries: int = 2,
    allowed_names: set[str] | None = None,
) -> ToolRun:
    """Run one model call, preserving DB failures for the API error boundary."""

    if not isinstance(tool_call, dict):
        return _error_run("invalid-tool-call", "unknown", "工具调用格式错误")
    name = tool_call.get("name")
    call_id = tool_call.get("id")
    safe_name = name if isinstance(name, str) and name else "unknown"
    safe_id = call_id if isinstance(call_id, str) and call_id else "invalid-tool-call"
    if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
        return _error_run(safe_id, safe_name, "工具调用缺少名称或 ID")
    if name == "create_ticket":
        return _error_run(safe_id, name, "建工单仅能由用户确认动作执行")
    if allowed_names is not None and name not in allowed_names:
        return _error_run(safe_id, name, "当前路径不允许此工具")
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return _error_run(safe_id, safe_name, "工具参数格式错误")
    tool = registry.get_tool(name)
    if tool is None:
        return _error_run(safe_id, safe_name, f"未知工具 {safe_name}")

    args = dict(args)
    if name in registry.INJECT_CONVERSATION:
        # InjectedToolArg hides this field from the model schema, but LangChain
        # still accepts it at execution. Never trust a model-supplied value.
        args["conversation_id"] = conversation_id
    elif "conversation_id" in args:
        return _error_run(safe_id, name, "不允许模型提供会话编号")
    if name == "query_faq":
        keyword = args.get("keyword")
        if not isinstance(keyword, str) or not keyword.strip():
            return _error_run(safe_id, name, "FAQ 关键词不能为空")

    # The RAG FAQ path includes query understanding, embedding, reranking and
    # a sufficiency check. Live runs take ~16 s, well beyond a simple mock tool.
    effective_timeout = timeout if timeout is not None else (45.0 if name == "query_faq" else 5.0)

    retries = 0 if name in registry.NO_RETRY else max(0, max_retries)
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=effective_timeout)
            return ToolRun(
                tool_call_id=safe_id,
                name=name,
                ok=True,
                tool_message=ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=safe_id,
                    name=name,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - classify before model-facing error
            if attempt < retries:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            if isinstance(exc, TimeoutError) and name in registry.DATABASE_TOOLS:
                raise ToolInfrastructureError(f"{name} timed out") from exc
            if isinstance(exc, (SQLAlchemyError, ConnectionError, OSError)) and not isinstance(exc, TimeoutError):
                raise
            logger.warning("Tool execution failed: name=%s type=%s", name, type(exc).__name__)
            return _error_run(safe_id, name, type(exc).__name__)

    raise AssertionError("unreachable")
