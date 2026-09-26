"""Chapter 7 model view stays separate from the persisted conversation."""

from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.context_budget import ContextBudget, ContextBudgetExceeded
from app.core.context_layers import build_history_context, build_model_context
from app.db.repository import ContextSnapshot, SummarySegment, VisibleMessage


def budget(*, recent=800, older=800, current=800):
    return ContextBudget(recent + older, recent, older, current)


def snapshot():
    # IDs are global: another conversation owns the gaps 11, 14, and 18.
    return ContextSnapshot(
        conversation_id=7, summary_upto_msg_id=10, layer1_from_msg_id=15,
        messages=(
            VisibleMessage(12, "user", "我订的蓝色猫窝怎么样？"),
            VisibleMessage(13, "assistant", "旧回复" + "甲" * 100),
            VisibleMessage(16, "user", "可以送货吗？"),
            VisibleMessage(17, "assistant", "可以送货，费用见页面。"),
        ),
        summaries=(SummarySegment(1, 1, 10, "用户订了蓝色猫窝。"),),
    )


def graph_messages():
    return [
        HumanMessage("我订的蓝色猫窝怎么样？"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "old-call"}]),
        ToolMessage(content="私密工具大块" * 100, tool_call_id="old-call"),
        AIMessage("旧回复" + "甲" * 100),
        HumanMessage("可以送货吗？"), AIMessage("可以送货，费用见页面。"),
        HumanMessage("当前问题"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "new-call"}]),
        ToolMessage(content="本轮工具结果", tool_call_id="new-call"),
    ]


def test_model_view_uses_global_anchors_and_keeps_current_exchange_then_context():
    source = graph_messages()
    saved = deepcopy(source)
    model = build_model_context(snapshot(), source, "当前问题", "已核验证据", "系统红线", budget())
    assert source == saved
    assert isinstance(model.messages[0], SystemMessage)
    assert [message.content for message in model.messages if isinstance(message, HumanMessage)][:-1] == [
        "我订的蓝色猫窝怎么样？", "可以送货吗？", "当前问题",
    ]
    assert model.messages[-1].content.index("用户订了蓝色猫窝") < model.messages[-1].content.index("已核验证据")
    assert isinstance(model.messages[-2], ToolMessage)
    assert model.messages[-2].content == "本轮工具结果"
    assert model.injected_summary == "用户订了蓝色猫窝。"
    assert "私密工具大块" not in str(model.messages)
    assert "旧回复" in str(model.messages)
    assert "甲" * 100 not in str(model.messages)
    assert "可以送货，费用见页面。" in str(model.messages)
    assert model.token_count > 0
    assert model.window_rows
    assert any("旧工具结果已省略" in row.content for row in model.window_rows)


def test_history_view_is_distinct_and_has_only_prior_visible_turns():
    history = build_history_context(snapshot(), budget())
    assert "用户订了蓝色猫窝。" in history
    assert "我订的蓝色猫窝怎么样？" in history
    assert "可以送货吗？" in history
    assert "旧回复" in history
    assert "甲" * 100 not in history
    assert "当前问题" not in history
    assert "私密工具大块" not in history


def test_model_view_rejects_current_exchange_that_exceeds_its_reserve():
    with pytest.raises(ContextBudgetExceeded, match="current|当前"):
        build_model_context(snapshot(), graph_messages(), "当前问题", "证据", "系统", budget(current=1))


def test_model_view_rejects_stale_query_even_if_prior_turn_has_same_text():
    with pytest.raises(ValueError, match="current query"):
        build_model_context(snapshot(), [HumanMessage("旧问题"), AIMessage("旧答"),
                                         HumanMessage("实际当前问题")],
                            "旧问题", "", "系统", budget())


def test_orphan_prior_user_is_not_injected_as_a_half_turn():
    source = ContextSnapshot(
        7, 0, 0,
        (VisibleMessage(1, "user", "已完成问"), VisibleMessage(2, "assistant", "已完成答"),
         VisibleMessage(3, "user", "未完成问")),
        (),
    )
    model = build_model_context(source, [HumanMessage("当前问题")], "当前问题", "", "系统", budget())
    assert "未完成问" not in str(model.messages)
    assert "已完成问" in str(model.messages)


def test_each_layer_trims_whole_turns_when_its_budget_is_zero():
    model = build_model_context(snapshot(), graph_messages(), "当前问题", "", "系统", budget(recent=0))
    assert "可以送货吗？" not in str(model.messages)
    assert "可以送货，费用见页面。" not in str(model.messages)
    assert "我订的蓝色猫窝怎么样？" in str(model.messages)


def test_layer2_reply_length_uses_setting(monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("LAYER2_ASSISTANT_CHARS", "4")
    get_settings.cache_clear()
    try:
        model = build_model_context(snapshot(), graph_messages(), "当前问题", "", "系统", budget())
        assert "旧回复甲…" in str(model.messages)
        assert "旧回复甲甲" not in str(model.messages)
    finally:
        get_settings.cache_clear()
