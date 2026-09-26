"""Chapter 7 window budgeting and Chinese token calibration."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.core.context_budget import (
    ContextBudgetExceeded,
    FixedCosts,
    derive_budget,
    validate_context_budget,
)
from app.core.memory import estimate_tokens


def settings(**overrides):
    return Settings(
        chat_model="offline-test", chat_base_url="http://127.0.0.1:9/v1",
        chat_api_key="offline-test", _env_file=None, **overrides,
    )


def fixed_costs():
    return FixedCosts(system_tools=650, retrieval_evidence=800, summary=250, safety=250)


def test_demo_window_derives_roughly_5650_history_with_70_30_layers():
    budget = derive_budget(settings(
        model_context_window=18000, max_output_tokens=2000,
        max_user_input_tokens=2000, max_agent_steps=3,
        tool_result_max_tokens=1200, rerank_top_k=5,
    ), fixed_costs())
    assert budget.current_peak == 8400
    assert budget.history_total == 5650
    assert (budget.layer1, budget.layer2) == (3955, 1695)


def test_react_peak_reserves_all_prior_assistant_tool_calls_and_results():
    config = settings(max_user_input_tokens=300, max_output_tokens=400,
                      max_agent_steps=4, tool_result_max_tokens=500,
                      model_context_window=12000)
    budget = derive_budget(config, fixed_costs())
    # Four model steps can have three completed tool exchanges before the
    # fourth call. Each assistant response includes its tool-call payload.
    assert budget.current_peak == 300 + 3 * (400 + 500)
    assert budget.history_total <= 12000 - fixed_costs().total - 400 - budget.current_peak


def test_default_target_keeps_twenty_short_turns_in_first_layer():
    config = settings()
    budget = derive_budget(config, fixed_costs())
    turns = [message for _ in range(20) for message in
             (HumanMessage("请问商品有货吗？"), AIMessage("您好，请提供商品名称。"))]
    assert estimate_tokens(turns, chars_per_token=config.cjk_chars_per_token) <= budget.layer1


def test_window_shrink_fails_one_turn_self_check():
    config = settings(model_context_window=9000)
    with pytest.raises(ContextBudgetExceeded, match="一轮|one turn"):
        validate_context_budget(config, fixed_costs())


def test_all_fixed_and_react_peak_costs_reduce_history():
    config = settings(model_context_window=15000, max_agent_steps=2)
    base = derive_budget(config, fixed_costs())
    for field in ("system_tools", "retrieval_evidence", "summary", "safety"):
        raised = FixedCosts(**{**vars(fixed_costs()), field: getattr(fixed_costs(), field) + 100})
        assert derive_budget(config, raised).history_total == base.history_total - 100
    for change in ({"max_output_tokens": 100}, {"max_user_input_tokens": 100},
                   {"tool_result_max_tokens": 100}):
        updated = config.model_copy(update={key: getattr(config, key) + value for key, value in change.items()})
        assert derive_budget(updated, fixed_costs()).history_total < base.history_total


def test_cjk_calibration_affects_message_count_and_target_budget_together():
    chinese = [SystemMessage("中文" * 100)]
    conservative = settings(cjk_chars_per_token=1.0)
    relaxed = settings(cjk_chars_per_token=2.0)
    assert estimate_tokens(chinese, chars_per_token=1.0) > estimate_tokens(chinese, chars_per_token=2.0)
    assert derive_budget(conservative, fixed_costs()).history_total > derive_budget(relaxed, fixed_costs()).history_total


def test_explicit_legacy_token_budget_cannot_bypass_self_check():
    config = settings(token_budget=1000)
    with pytest.raises(ContextBudgetExceeded):
        validate_context_budget(config, fixed_costs())


def test_fixed_costs_measure_uses_the_same_calibration():
    conservative = FixedCosts.measure(
        settings(cjk_chars_per_token=1.0),
        system_tools=[SystemMessage("规" * 100)],
        retrieval_evidence=[HumanMessage("证" * 100)],
        summary=[HumanMessage("摘" * 100)], safety=50,
    )
    relaxed = FixedCosts.measure(
        settings(cjk_chars_per_token=2.0),
        system_tools=[SystemMessage("规" * 100)],
        retrieval_evidence=[HumanMessage("证" * 100)],
        summary=[HumanMessage("摘" * 100)], safety=50,
    )
    assert conservative.total > relaxed.total
    assert conservative.safety == relaxed.safety == 50
