"""Dynamic MCP discovery and local server admission policy."""

from app.tools import mcp_client, registry
from app.tools.registry import ToolSpec
import socket
import time
import pytest


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} 描述"
        self.args_schema = {"type": "object", "properties": {"x": {"type": "string"}},
                            "required": ["x"]}


async def test_discovery_marks_approved_server_and_uses_local_formatters(monkeypatch):
    async def get_tools(*, server_name):
        return [FakeTool("query_logistics")] if server_name == "logistics" else [FakeTool("query_warranty")]

    monkeypatch.setattr(mcp_client, "_get_tools_of", get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    by_name = {spec.name: spec for spec in specs}
    assert by_name["query_logistics"].source == "mcp"
    assert by_name["query_logistics"].mcp_server == "logistics"
    assert all(spec.permission == "read" for spec in specs)
    assert by_name["query_logistics"].format_result is mcp_client.FORMATTERS["query_logistics"]


async def test_one_unavailable_server_does_not_hide_other_server(monkeypatch):
    async def get_tools(*, server_name):
        if server_name == "logistics":
            raise ConnectionError("offline")
        return [FakeTool("query_warranty")]

    monkeypatch.setattr(mcp_client, "_get_tools_of", get_tools)
    assert {spec.name for spec in await mcp_client.fetch_mcp_specs()} == {"query_warranty"}


async def test_builtin_name_wins_and_dynamic_query_is_added(monkeypatch):
    async def fetch():
        return [
            ToolSpec("query_order", "untrusted overlap", {"type": "object", "properties": {}},
                     FakeTool("query_order"), "read", "mcp", "logistics"),
            ToolSpec("query_delivery_eta", "new tool", {"type": "object", "properties": {}},
                     FakeTool("query_delivery_eta"), "read", "mcp", "logistics"),
            ToolSpec("query_external_write", "unknown source", {"type": "object", "properties": {}},
                     FakeTool("query_external_write"), "read", "mcp", "untrusted"),
        ]

    monkeypatch.setattr(mcp_client, "fetch_mcp_specs", fetch)
    by_name = {spec.name: spec for spec in await registry.get_all_specs()}
    assert by_name["query_order"].source == "builtin"
    assert by_name["query_delivery_eta"].source == "mcp"
    assert "query_external_write" not in by_name


def test_unknown_server_and_mcp_write_name_are_rejected():
    for tool, server in ((FakeTool("query_new"), "untrusted"),
                         (FakeTool("create_ticket"), "logistics")):
        try:
            registry.spec_from_langchain_tool(tool, source="mcp", mcp_server=server)
        except ValueError:
            pass
        else:
            raise AssertionError("untrusted MCP tool became callable")


def test_logistics_formatter_removes_internal_codes():
    formatted = mcp_client.FORMATTERS["query_logistics"]({
        "tracking_no": "SF1", "status_code": "IN_TRANSIT", "current_city": "深圳",
        "trace": ["深圳分拨中心"], "carrier_code": "private",
    })
    assert formatted == {"tracking_no": "SF1", "status": "运输中",
                         "current_city": "深圳", "trace": ["深圳分拨中心"]}


async def test_unavailable_local_server_is_rejected_quickly(monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]
    monkeypatch.setattr(mcp_client, "_connections", lambda: {
        "logistics": {"transport": "streamable_http", "url": f"http://127.0.0.1:{unused_port}/mcp",
                      "timeout": 2.0},
    })
    started = time.monotonic()
    with pytest.raises(Exception):
        await mcp_client._get_tools_of(server_name="logistics")
    assert time.monotonic() - started < 1.5
