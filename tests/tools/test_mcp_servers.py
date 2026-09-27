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
    assert "内部状态码" not in str(payload)
    assert audits[-1]["tool_source"] == "mcp"


async def test_delayed_mcp_records_timeout_and_retries(monkeypatch):
    from app.tools import engine, registry

    child = subprocess.Popen(
        [sys.executable, "mcp_servers/logistics_server.py"],
        env={**os.environ, "PORT": "18121", "MOCK_DELAY_SECONDS": "12"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(("127.0.0.1", 18121)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise AssertionError("slow MCP server did not start")
        tools = await MultiServerMCPClient({
            "logistics": {"transport": "streamable_http",
                          "url": "http://127.0.0.1:18121/mcp", "timeout": 2.0,
                          "sse_read_timeout": 2.0},
        }, handle_tool_errors=False).get_tools(server_name="logistics")
        spec = registry.spec_from_langchain_tool(
            tools[0], source="mcp", mcp_server="logistics",
        )
        audits = []

        async def audit(**fields):
            audits.append(fields)

        monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
        run = await engine.execute_tool_call(
            {"name": "query_logistics", "args": {"tracking_no": "SF123"}, "id": "slow-1"},
            1, {"query_logistics": spec},
        )
        assert run.status == "超时" and run.retry_count == 2
        assert audits[-1]["status"] == "超时" and audits[-1]["retry_count"] == 2
    finally:
        child.terminate()
        try:
            child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=5)


async def test_new_mcp_tool_appears_after_only_server_restart(tmp_path, monkeypatch):
    from app.tools import mcp_client, registry

    monkeypatch.setattr(mcp_client, "_connections", lambda: {
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18131/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18132/mcp"},
    })

    def start(path):
        child = subprocess.Popen(
            [sys.executable, str(path)], env={**os.environ, "PORT": "18131"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(("127.0.0.1", 18131)) == 0:
                    return child
            if child.poll() is not None:
                raise AssertionError("MCP server exited before readiness")
            time.sleep(0.1)
        child.terminate()
        raise AssertionError("MCP server did not listen")

    child = start(Path("mcp_servers/logistics_server.py"))
    try:
        before = {spec.name for spec in await registry.get_all_specs()}
        assert "query_logistics" in before and "query_delivery_eta" not in before
    finally:
        child.terminate()
        child.wait(timeout=5)

    source = Path("mcp_servers/logistics_server.py").read_text(encoding="utf-8")
    extra = '''@mcp.tool()
async def query_delivery_eta(tracking_no: str) -> dict:
    """查询演示预计送达日。"""
    return {"tracking_no": tracking_no, "eta": "演示预计明日"}


'''
    alternate = tmp_path / "logistics_server_extra.py"
    alternate.write_text(source.replace('if __name__ == "__main__":',
                                        extra + 'if __name__ == "__main__":'), encoding="utf-8")
    child = start(alternate)
    try:
        after = {spec.name for spec in await registry.get_all_specs()}
        assert "query_delivery_eta" in after
    finally:
        child.terminate()
        child.wait(timeout=5)
