"""Fresh discovery of approved mock MCP tools via LangChain's official adapter."""

import logging
import asyncio
from urllib.parse import urlsplit

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import get_settings
from app.tools import registry
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)


def _translate(values: dict[str, str], code: str | None) -> str | None:
    return values.get(code, code)


def _format_logistics(value: dict) -> dict:
    public_trace = [item for item in value.get("trace", [])
                    if isinstance(item, str) and "内部状态码" not in item]
    return {
        "tracking_no": value.get("tracking_no"),
        "status": _translate({"PICKED_UP": "已揽件", "IN_TRANSIT": "运输中",
                              "DELIVERING": "派送中", "DELIVERED": "已签收"}, value.get("status_code")),
        "current_city": value.get("current_city"), "trace": public_trace,
    }


def _format_warranty(value: dict) -> dict:
    return {
        "order_id": value.get("order_id"),
        "warranty": _translate({"IN_WARRANTY": "在保", "EXPIRED": "已过保"},
                                value.get("warranty_code")),
        "warranty_until": value.get("warranty_until"),
    }


def _format_return(value: dict) -> dict:
    return {
        "order_id": value.get("order_id"),
        "return_status": _translate({"AUDITING": "审核中", "RETURNING": "退货中",
                                     "REFUNDED": "已退款", "NONE": "无退货记录"},
                                    value.get("return_code")),
        "updated_at": value.get("updated_at"),
    }


FORMATTERS = {
    "query_logistics": _format_logistics,
    "query_warranty": _format_warranty,
    "query_return_status": _format_return,
}


def _connections() -> dict:
    settings = get_settings()
    timeout = settings.mcp_tool_timeout
    return {
        "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url,
                      "timeout": timeout, "sse_read_timeout": timeout},
        "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url,
                       "timeout": timeout, "sse_read_timeout": timeout},
    }


async def _get_tools_of(*, server_name: str):
    connections = _connections()
    if server_name not in registry.APPROVED_MCP_SERVERS or server_name not in connections:
        raise ValueError("MCP server is not locally approved")
    endpoint = urlsplit(connections[server_name]["url"])
    if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.port is None or endpoint.path != "/mcp" or endpoint.username
            or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("MCP endpoint must be an approved local HTTP /mcp service")
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(endpoint.hostname, endpoint.port), timeout=0.3,
        )
    except (OSError, TimeoutError) as exc:
        raise ConnectionError(f"local MCP server {server_name} is unavailable") from exc
    writer.close()
    await writer.wait_closed()
    client = MultiServerMCPClient(connections, handle_tool_errors=False)
    return await client.get_tools(server_name=server_name)


async def fetch_mcp_specs() -> list[ToolSpec]:
    specs = []
    for server in registry.APPROVED_MCP_SERVERS:
        try:
            tools = await _get_tools_of(server_name=server)
        except Exception as exc:
            logger.warning("MCP Server %s 不可达,本轮跳过:%s", server, type(exc).__name__)
            continue
        for tool in tools:
            try:
                specs.append(registry.spec_from_langchain_tool(
                    tool, source="mcp", mcp_server=server,
                    format_result=FORMATTERS.get(tool.name),
                ))
            except ValueError as exc:
                logger.warning("MCP tool rejected: server=%s name=%s reason=%s", server, tool.name, exc)
    return specs
