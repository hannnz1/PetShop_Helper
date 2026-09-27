"""Execution boundary: schema, permission, retries, formatting and audit."""

import asyncio

import pytest

from app.tools import engine
from app.tools.registry import ToolSpec


SCHEMA = {
    "type": "object", "properties": {"order_id": {"type": "string", "minLength": 2}},
    "required": ["order_id"], "additionalProperties": False,
}


class FakeTool:
    def __init__(self, fn):
        self.fn = fn

    async def ainvoke(self, args):
        return await self.fn(args)


def spec(fn, *, name="query_order", permission="read", schema=SCHEMA,
         source="builtin", timeout=0.05, inject=False, formatter=None):
    return ToolSpec(name, "测试工具", schema, FakeTool(fn), permission, source,
                    timeout=timeout, inject_conversation=inject, format_result=formatter)


@pytest.fixture
def audits(monkeypatch):
    rows = []

    async def record(**fields):
        rows.append(fields)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", record)
    return rows


def call(name="query_order", args=None):
    return {"id": "tc-1", "name": name, "args": args if args is not None else {"order_id": "1001"}}


async def test_unknown_tool_is_audited_as_failure(audits):
    run = await engine.execute_tool_call(call("missing"), 1, {})
    assert not run.ok and "未知工具" in run.tool_message.content
    assert audits[-1]["status"] == "失败"


async def test_schema_blocks_missing_type_range_and_model_injected_field(audits):
    async def should_not_run(_):
        raise AssertionError("invalid arguments reached tool")

    tool_spec = spec(should_not_run)
    for bad_args in ({}, {"order_id": 1}, {"order_id": "x"},
                     {"order_id": "1001", "conversation_id": 99}):
        run = await engine.execute_tool_call(call(args=bad_args), 1, {"query_order": tool_spec})
        assert not run.ok and run.status == "校验拦下"
        assert "参数校验未通过" in run.tool_message.content
    assert [row["status"] for row in audits] == ["校验拦下"] * 4


async def test_unconfirmed_write_denied_but_confirmed_write_injects_trusted_id(audits):
    seen = []

    async def write(args):
        seen.append(args)
        return {"ticket_no": "T1"}

    tool_spec = spec(write, name="create_ticket", permission="write", inject=True,
                     schema={"type": "object", "properties": {"description": {"type": "string"}},
                             "required": ["description"], "additionalProperties": False})
    payload = call("create_ticket", {"description": "物流信息异常"})
    denied = await engine.execute_tool_call(payload, 7, {"create_ticket": tool_spec})
    assert denied.status == "权限拒绝" and not seen
    accepted = await engine.execute_tool_call(payload, 7, {"create_ticket": tool_spec}, confirmed=True)
    assert accepted.ok and seen == [{"description": "物流信息异常", "conversation_id": 7}]
    assert [row["status"] for row in audits] == ["权限拒绝", "成功"]


async def test_read_timeout_retries_twice_and_write_timeout_never_retries(audits):
    attempts = []

    async def slow(_):
        attempts.append(1)
        await asyncio.sleep(1)

    read_spec = spec(slow, timeout=0.01)
    read = await engine.execute_tool_call(call(), 1, {"query_order": read_spec})
    assert read.status == "超时" and read.retry_count == 2 and len(attempts) == 3
    assert audits[-1]["duration_ms"] >= 0

    write_spec = spec(slow, name="create_ticket", permission="write", timeout=0.01)
    write = await engine.execute_tool_call(call("create_ticket"), 1,
                                           {"create_ticket": write_spec}, confirmed=True)
    assert write.status == "超时" and write.retry_count == 0 and len(attempts) == 4


async def test_network_error_retries_but_business_failure_does_not(audits):
    attempts = []

    async def flaky(_):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("network blip")
        return {"status": "运输中"}

    run = await engine.execute_tool_call(call(), 1, {"query_order": spec(flaky)})
    assert run.ok and run.retry_count == 1 and "运输中" in run.tool_message.content

    business_calls = []

    async def business_failure(_):
        business_calls.append(1)
        raise ValueError("not found")

    failed = await engine.execute_tool_call(call(), 1, {"query_order": spec(business_failure)})
    assert failed.status == "失败" and failed.retry_count == 0 and len(business_calls) == 1


async def test_formatter_removes_internal_fields_and_audit_failure_is_nonblocking(audits, monkeypatch):
    async def result(_):
        return {"tracking_no": "SF1", "status_code": "TRANSIT", "internal_ref": "secret"}

    def public(value):
        return {"tracking_no": value["tracking_no"], "status": "运输中"}

    tool_spec = spec(result, name="query_logistics", formatter=public, source="mcp")
    run = await engine.execute_tool_call(call("query_logistics"), 1, {"query_logistics": tool_spec})
    assert run.ok and "运输中" in run.tool_message.content
    assert "internal_ref" not in run.tool_message.content
    assert audits[-1]["tool_source"] == "mcp"

    async def audit_failure(**_):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit_failure)
    again = await engine.execute_tool_call(call("query_logistics"), 1, {"query_logistics": tool_spec})
    assert again.ok


async def test_mcp_content_blocks_are_unwrapped_before_local_formatter(audits):
    async def result(_):
        return [{"type": "text", "text": '{"tracking_no":"SF1","status_code":"IN_TRANSIT","carrier_code":"private"}',
                 "id": "random-content-id"}]

    tool_spec = spec(result, name="query_logistics", source="mcp",
                     formatter=lambda value: {"tracking_no": value["tracking_no"],
                                              "status": "运输中"})
    run = await engine.execute_tool_call(call("query_logistics"), 1, {"query_logistics": tool_spec})
    assert run.ok
    assert '"status": "运输中"' in run.tool_message.content
    assert "random-content-id" not in run.tool_message.content
