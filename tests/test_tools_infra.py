"""Tool execution boundary: trusted context, retries, and failure classes."""

import asyncio
import json

import pytest
from sqlalchemy.exc import OperationalError

from app.tools import infra, registry


def test_registry_exposes_only_registered_read_tools():
    names = {tool.name for tool in registry.get_all_tools()}
    assert names == {"query_order", "query_product", "query_faq", "submit_refund"}
    assert registry.get_tool("query_logistics") is None  # MCP owns logistics after Task 5.
    assert registry.get_tool("create_ticket") is not None
    assert registry.get_tool("missing") is None


@pytest.mark.asyncio
async def test_unknown_and_malformed_calls_return_safe_error_messages():
    for call in (
        {"name": "missing", "args": {}, "id": "c1"},
        {"name": "query_order", "args": [], "id": "c2"},
        {"name": "query_order", "args": {}, "id": ""},
        {"args": {}, "id": "c4"},
        None,
    ):
        run = await infra.execute_tool_call(call, conversation_id=7)
        assert not run.ok
        assert run.tool_message.status == "error"
        assert run.tool_message.tool_call_id == run.tool_call_id
        assert "工具执行失败" in run.tool_message.content


@pytest.mark.asyncio
async def test_model_cannot_create_ticket_even_with_forged_context(monkeypatch):
    seen = []

    class Fake:
        async def ainvoke(self, args):
            seen.append(args)
            return {"ticket_no": "T1"}

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x", "ticket_type": "售后", "conversation_id": 999}, "id": "c1"},
        conversation_id=7,
    )
    assert not run.ok
    assert seen == []


@pytest.mark.asyncio
async def test_other_tools_reject_model_supplied_conversation_id(monkeypatch):
    called = False

    class Fake:
        async def ainvoke(self, args):
            nonlocal called
            called = True

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1001", "conversation_id": 999}, "id": "c1"}, 7
    )
    assert not run.ok and not called


@pytest.mark.asyncio
async def test_read_only_tool_retries_then_succeeds(monkeypatch):
    attempts = 0

    class Fake:
        async def ainvoke(self, args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient")
            return {"ok": True}

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call({"name": "query_order", "args": {}, "id": "c1"}, 7, max_retries=1)
    assert run.ok and attempts == 2


@pytest.mark.asyncio
async def test_timeout_has_bounded_read_only_retries(monkeypatch):
    attempts = 0

    class Fake:
        async def ainvoke(self, args):
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(1)

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {}, "id": "c1"}, 7, timeout=0.01, max_retries=1
    )
    assert not run.ok and attempts == 2


@pytest.mark.asyncio
async def test_create_ticket_is_rejected_before_retry(monkeypatch):
    attempts = 0

    class Fake:
        async def ainvoke(self, args):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("failed")

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call({"name": "create_ticket", "args": {}, "id": "c1"}, 7)
    assert not run.ok and attempts == 0


@pytest.mark.asyncio
async def test_sqlalchemy_and_connection_failures_propagate(monkeypatch):
    for failure in (
        OperationalError("SELECT 1", {}, Exception("database down")),
        ConnectionError("database down"),
    ):
        class Fake:
            async def ainvoke(self, args):
                raise failure

        monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
        with pytest.raises(type(failure)):
            await infra.execute_tool_call(
                {"name": "query_faq", "args": {"keyword": "退货"}, "id": "c1"}, 7, max_retries=0
            )


@pytest.mark.asyncio
async def test_blank_faq_keyword_rejected_before_db(monkeypatch):
    called = False

    class Fake:
        async def ainvoke(self, args):
            nonlocal called
            called = True

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call(
        {"name": "query_faq", "args": {"keyword": "   "}, "id": "c1"}, 7
    )
    assert not run.ok and not called


@pytest.mark.asyncio
async def test_database_tool_timeout_propagates_as_infrastructure_failure(monkeypatch):
    class Fake:
        async def ainvoke(self, args):
            await asyncio.sleep(1)

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    with pytest.raises(infra.ToolInfrastructureError):
        await infra.execute_tool_call(
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "c1"},
            7,
            timeout=0.01,
            max_retries=0,
        )


@pytest.mark.asyncio
async def test_ticket_is_rejected_before_timeout(monkeypatch):
    attempts = 0

    class Fake:
        async def ainvoke(self, args):
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(1)

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    run = await infra.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "投诉", "ticket_type": "投诉"}, "id": "c1"},
        7, timeout=0.01,
    )
    assert not run.ok and attempts == 0


@pytest.mark.asyncio
async def test_database_tool_uses_default_retry_count(monkeypatch):
    attempts = 0

    class Fake:
        async def ainvoke(self, args):
            nonlocal attempts
            attempts += 1
            raise ConnectionError("offline")

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    with pytest.raises(ConnectionError):
        await infra.execute_tool_call(
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "c1"}, 7
        )
    assert attempts == 3


@pytest.mark.asyncio
async def test_faq_gets_longer_default_timeout_for_rag_pipeline(monkeypatch):
    seen = []
    original_wait_for = infra.asyncio.wait_for

    async def record_timeout(awaitable, timeout):
        seen.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    class Fake:
        async def ainvoke(self, args):
            return {"sufficient": True}

    monkeypatch.setattr(registry, "get_tool", lambda name: Fake())
    monkeypatch.setattr(infra.asyncio, "wait_for", record_timeout)
    for name in ("query_faq", "query_order"):
        await infra.execute_tool_call(
            {"name": name, "args": {"keyword": "MH-W40"}, "id": name}, 7,
        )
    assert seen[0] >= 30
    assert seen[1] == 5
