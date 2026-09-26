"""Real isolated MySQL contract for the chapter-eight audit row."""

from sqlalchemy import select

from app.db import repository
from app.db.models import ToolAuditLog


async def test_insert_tool_audit_without_conversation(db_session_factory, db_clean):
    await repository.insert_tool_audit(
        conversation_id=None, tool_call_id=None, tool_name="query_order",
        tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
        result_summary="{}", status="成功", error_message=None,
        retry_count=0, duration_ms=12,
    )
    async with db_session_factory() as session:
        row = (await session.execute(select(ToolAuditLog))).scalars().one()
    assert row.conversation_id is None
    assert row.arguments == {"order_id": "1001"}
    assert (row.tool_name, row.status, row.retry_count) == ("query_order", "成功", 0)


async def test_insert_tool_audit_records_every_outcome(db_session_factory, db_clean):
    statuses = ("成功", "失败", "超时", "校验拦下", "权限拒绝")
    for status in statuses:
        await repository.insert_tool_audit(
            conversation_id=101, tool_call_id=f"tc-{status}", tool_name="create_ticket",
            tool_source="builtin", mcp_server=None, arguments={},
            result_summary=None, status=status, error_message="原因",
            retry_count=0, duration_ms=None,
        )
    async with db_session_factory() as session:
        rows = (await session.execute(select(ToolAuditLog))).scalars().all()
    assert {row.status for row in rows} == set(statuses)
    assert all(row.conversation_id == 101 for row in rows)
