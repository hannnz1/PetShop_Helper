"""Start both official-SDK MCP servers and discover/call over real HTTP."""

import os
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


def test_both_server_entrypoints_exist():
    assert Path("mcp_servers/logistics_server.py").is_file()
    assert Path("mcp_servers/aftersales_server.py").is_file()


@pytest.fixture(scope="module")
def mcp_servers():
    children = []
    try:
        for filename, port in (("logistics_server.py", 18101),
                               ("aftersales_server.py", 18102)):
            env = {**os.environ, "PORT": str(port), "MOCK_DELAY_SECONDS": "0"}
            child = subprocess.Popen(
                [sys.executable, f"mcp_servers/{filename}"], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            children.append(child)
        for port in (18101, 18102):
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                with socket.socket() as sock:
                    sock.settimeout(0.2)
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        break
                if any(child.poll() is not None for child in children):
                    raise AssertionError("MCP server exited before readiness")
                time.sleep(0.1)
            else:
                raise AssertionError(f"MCP server :{port} did not start")
        yield
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=5)


def client():
    return MultiServerMCPClient({
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18101/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18102/mcp"},
    }, handle_tool_errors=False)


async def test_both_servers_discover_three_described_schema_tools(mcp_servers):
    tools = await client().get_tools()
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"query_logistics", "query_warranty", "query_return_status"}
    for tool in by_name.values():
        schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
        assert tool.description and schema.get("properties")


async def test_logistics_mock_is_stable_over_real_mcp_http(mcp_servers):
    tools = {tool.name: tool for tool in await client().get_tools(server_name="logistics")}
    first = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    second = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    first_payload = json.loads(first[0]["text"])
    second_payload = json.loads(second[0]["text"])
    assert first_payload == second_payload
    assert first_payload["tracking_no"] == "SF123"
    assert "status_code" in first_payload


async def test_registry_and_engine_use_real_logistics_mcp(mcp_servers, monkeypatch):
    from app.tools import engine, mcp_client, registry
    from app.tools.business import query_order

    monkeypatch.setattr(mcp_client, "_connections", lambda: {
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18101/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18102/mcp"},
    })
    audits = []

    async def audit(**fields):
        audits.append(fields)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
    order = await query_order.ainvoke({"order_id": "1001"})
    specs = {spec.name: spec for spec in await registry.get_all_specs()}
    assert specs["query_logistics"].source == "mcp"
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": order["tracking_no"]}, "id": "live-mcp"},
        7, specs,
    )
    assert run.ok and run.status == "成功"
    payload = json.loads(run.tool_message.content)
    assert payload["tracking_no"] == order["tracking_no"]
    assert payload["status"] in {"已揽件", "运输中", "派送中", "已签收"}
    assert "carrier_code" not in payload
    assert audits[-1]["tool_source"] == "mcp"
