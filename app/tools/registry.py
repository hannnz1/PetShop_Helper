"""Local tool registration and model-visible JSON Schema metadata."""

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)
WRITE_TOOLS = {"create_ticket"}
APPROVED_MCP_SERVERS = {"logistics", "aftersales"}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict
    tool: BaseTool
    permission: str
    source: str
    mcp_server: str | None = None
    timeout: float | None = None
    inject_conversation: bool = False
    format_result: Callable[[dict], dict] | None = None


_BUILTIN: dict[str, ToolSpec] = {}
_scanned = False


def permission_for(name: str) -> str:
    return "write" if name in WRITE_TOOLS else "read"


def spec_from_langchain_tool(
    tool: BaseTool, *, source: str, mcp_server: str | None = None,
    timeout: float | None = None, inject_conversation: bool = False,
    format_result: Callable[[dict], dict] | None = None,
) -> ToolSpec:
    """Extract the schema shown to the model, excluding injected arguments."""

    if source == "mcp" and mcp_server not in APPROVED_MCP_SERVERS:
        raise ValueError("MCP server is not locally approved")
    if source == "mcp" and tool.name in WRITE_TOOLS:
        raise ValueError("MCP server cannot claim an application write tool")
    raw = getattr(tool, "args_schema", None)
    schema = raw if isinstance(raw, dict) else tool.tool_call_schema.model_json_schema()
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        raise ValueError(f"tool {tool.name} has no JSON Schema properties")
    return ToolSpec(
        name=tool.name, description=tool.description or "", json_schema=schema,
        tool=tool, permission=permission_for(tool.name), source=source,
        mcp_server=mcp_server, timeout=timeout,
        inject_conversation=inject_conversation, format_result=format_result,
    )


def register(spec: ToolSpec) -> None:
    if spec.name in _BUILTIN:
        logger.warning("工具重名,丢弃后注册者 name=%s", spec.name)
        return
    _BUILTIN[spec.name] = spec


def scan_builtin() -> None:
    global _scanned
    if _scanned:
        return
    from app.tools import builtin

    for module in pkgutil.iter_modules(builtin.__path__):
        importlib.import_module(f"{builtin.__name__}.{module.name}")
    _scanned = True
    logger.info("内置工具注册完成:%s", sorted(_BUILTIN))


def builtin_specs() -> list[ToolSpec]:
    scan_builtin()
    return list(_BUILTIN.values())


def get_builtin_spec(name: str) -> ToolSpec | None:
    scan_builtin()
    return _BUILTIN.get(name)


async def get_all_specs() -> list[ToolSpec]:
    """Merge each fresh MCP discovery with immutable builtin registrations."""

    from app.tools import mcp_client

    merged = {spec.name: spec for spec in builtin_specs()}
    for spec in await mcp_client.fetch_mcp_specs():
        if (spec.source != "mcp" or spec.mcp_server not in APPROVED_MCP_SERVERS
                or spec.name in WRITE_TOOLS):
            logger.warning("MCP source rejected: name=%s server=%s", spec.name, spec.mcp_server)
            continue
        if spec.name in merged:
            logger.warning("MCP 工具重名,保留内置 name=%s server=%s", spec.name, spec.mcp_server)
            continue
        merged[spec.name] = spec
    return list(merged.values())


# The Ch02 HTTP agent remains available alongside the Ch05 Graph endpoint.
# Keep its public registry API until that endpoint is migrated separately.
NO_RETRY = {"create_ticket", "submit_refund"}
INJECT_CONVERSATION = {"create_ticket"}
DATABASE_TOOLS = {"query_faq", "create_ticket"}
TOOL_TIMEOUTS = {"query_faq": 45.0}


def get_all_tools() -> list[BaseTool]:
    return [spec.tool for spec in builtin_specs() if spec.permission == "read"]


def get_tool(name: str) -> BaseTool | None:
    spec = get_builtin_spec(name)
    return spec.tool if spec else None


def get_chat_tools(route: str) -> list[BaseTool]:
    """Conservative static schemas used at startup before MCP discovery."""

    if route == "knowledge":
        names = ("query_order",)
    elif route == "refund":
        names = ("submit_refund",)
    elif route == "business":
        names = ("query_order", "query_product")
    else:
        names = ()
    tools = [spec.tool for name in names if (spec := get_builtin_spec(name))]
    return tools
