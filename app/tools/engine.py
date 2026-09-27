"""Single execution boundary for validation, permission, retries and audit."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage

from app.config import get_settings
from app.db import repository
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)
_TRANSIENT = (TimeoutError, ConnectionError, httpx.NetworkError, httpx.TimeoutException)


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str
    retry_count: int = 0
    duration_ms: int = 0


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """Return a model-facing JSON Schema error, or None when valid."""

    error = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if error is None:
        return None
    return f"{error.message} ({error.json_path})"


def _format_result(spec: ToolSpec, result: object) -> str:
    if (isinstance(result, list) and result
            and all(isinstance(block, dict) and block.get("type") == "text"
                    and isinstance(block.get("text"), str) for block in result)):
        result = "\n".join(block["text"] for block in result)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result
    if spec.format_result and isinstance(result, dict):
        result = spec.format_result(result)
    return json.dumps(result, ensure_ascii=False, default=str)


async def _audit(
    conversation_id: int, call_id: str, name: str, spec: ToolSpec | None,
    args: dict | None, content: str | None, status: str, error: str | None,
    retries: int, duration_ms: int,
) -> None:
    try:
        await repository.insert_tool_audit(
            conversation_id=conversation_id, tool_call_id=call_id,
            tool_name=name, tool_source=spec.source if spec else "builtin",
            mcp_server=spec.mcp_server if spec else None,
            arguments=args, result_summary=content[:500] if content else None,
            status=status, error_message=error[:512] if error else None,
            retry_count=retries, duration_ms=duration_ms,
        )
    except Exception:  # Audit is deliberately best effort.
        logger.exception("tool audit failed: name=%s status=%s", name, status)


async def execute_tool_call(
    tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec],
    *, confirmed: bool = False, deny_note: str | None = None,
) -> ToolRun:
    started = time.monotonic()
    call = tool_call if isinstance(tool_call, dict) else {}
    name = call.get("name") if isinstance(call.get("name"), str) else "unknown"
    call_id = call.get("id") if isinstance(call.get("id"), str) else "invalid-tool-call"
    args = call.get("args")
    args = dict(args) if isinstance(args, dict) else None
    spec = specs.get(name)

    def result(ok: bool, status: str, content: str, retries: int = 0) -> ToolRun:
        return ToolRun(
            tool_call_id=call_id, name=name, ok=ok, status=status,
            retry_count=retries, duration_ms=int((time.monotonic() - started) * 1000),
            tool_message=ToolMessage(
                content=content, tool_call_id=call_id, name=name,
                status="success" if ok else "error",
            ),
        )

    if spec is None:
        run = result(False, "失败", f"工具执行失败:未知工具 {name}")
        await _audit(conversation_id, call_id, name, None, args, None, run.status,
                     "未知工具", 0, run.duration_ms)
        return run

    if args is None:
        error = "参数必须是 JSON 对象"
    else:
        error = validate_args(spec, args)
    if error:
        run = result(False, "校验拦下", f"参数校验未通过:{error}。请修正参数或向用户追问。")
        await _audit(conversation_id, call_id, name, spec, args, None, run.status,
                     error, 0, run.duration_ms)
        return run

    if spec.permission == "write" and not confirmed:
        note = deny_note or "未收到用户确认"
        run = result(False, "权限拒绝", f"写操作未执行:{note}")
        await _audit(conversation_id, call_id, name, spec, args, None, run.status,
                     note, 0, run.duration_ms)
        return run

    execution_args = dict(args)
    if spec.inject_conversation:
        execution_args["conversation_id"] = conversation_id
    settings = get_settings()
    timeout = spec.timeout or (settings.mcp_tool_timeout if spec.source == "mcp"
                               else settings.tool_default_timeout)
    max_retries = 0 if spec.permission == "write" else settings.tool_max_retries

    for attempt in range(max_retries + 1):
        try:
            output = await asyncio.wait_for(spec.tool.ainvoke(execution_args), timeout=timeout)
            content = _format_result(spec, output)
            run = result(True, "成功", content, attempt)
            await _audit(conversation_id, call_id, name, spec, args, content, run.status,
                         None, attempt, run.duration_ms)
            return run
        except _TRANSIENT as exc:
            if attempt < max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            timed_out = isinstance(exc, (TimeoutError, httpx.TimeoutException))
            status = "超时" if timed_out else "失败"
            run = result(False, status, f"工具暂时不可用:{'执行超时' if timed_out else type(exc).__name__}", attempt)
            await _audit(conversation_id, call_id, name, spec, args, None, status,
                         f"{type(exc).__name__}: {exc}", attempt, run.duration_ms)
            return run
        except Exception as exc:
            logger.exception("tool execution failed: name=%s", name)
            run = result(False, "失败", f"工具暂时不可用:{type(exc).__name__}", attempt)
            await _audit(conversation_id, call_id, name, spec, args, None, run.status,
                         f"{type(exc).__name__}: {exc}", attempt, run.duration_ms)
            return run

    raise AssertionError("unreachable")
