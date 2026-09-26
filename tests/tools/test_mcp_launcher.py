"""Windows launcher starts both isolated mock servers and stops only its PIDs."""

import subprocess
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


@pytest.mark.skipif(not Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").exists(),
                    reason="Windows PowerShell launcher test")
async def test_windows_launcher_starts_and_stops_servers():
    assert Path("scripts/mcp.ps1").is_file()
    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               "scripts/mcp.ps1"]
    args = ["-LogisticsPort", "18111", "-AftersalesPort", "18112"]
    try:
        start = subprocess.run(command + ["-Action", "Start"] + args,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        assert start.returncode == 0
        client = MultiServerMCPClient({
            "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18111/mcp"},
            "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18112/mcp"},
        })
        assert {tool.name for tool in await client.get_tools()} == {
            "query_logistics", "query_warranty", "query_return_status",
        }
    finally:
        stop = subprocess.run(command + ["-Action", "Stop"] + args,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        assert stop.returncode == 0
