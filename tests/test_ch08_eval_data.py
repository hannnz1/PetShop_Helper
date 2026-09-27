"""Annotated evaluator must reject generic replies and failed tool calls."""

from scripts.eval_ch08 import _confirmed_ticket_passed, _logistics_passed, load_cases


def test_ch08_marked_cases_are_present():
    assert len(load_cases()) == 5


def test_ticket_confirmation_needs_actual_number_and_success_audit():
    assert not _confirmed_ticket_passed(1, "T-001", "工单已创建", ("成功", "{}"))
    assert not _confirmed_ticket_passed(1, "T-001", "工单 T-001 已创建", ("失败", None))
    assert _confirmed_ticket_passed(1, "T-001", "工单 T-001 已创建", ("成功", "{}"))


def test_logistics_needs_successful_mcp_audit_and_translated_status_in_answer():
    tools = {"query_order", "query_logistics"}
    assert not _logistics_passed(tools, "物流暂时不可用", ("超时", None))
    assert not _logistics_passed(tools, "物流到了", ("成功", '{"status":"运输中"}'))
    assert _logistics_passed(tools, "演示记录显示运输中", ("成功", '{"status":"运输中"}'))
