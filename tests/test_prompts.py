import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.core.prompts import AGENT_PROMPT, CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT


def test_customer_service_prompt_renders_with_complete_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[
            HumanMessage("之前说的商品是什么？"),
            AIMessage("请问您想了解哪类商品？"),
            HumanMessage("你们卖猫粮吗？"),
        ]
    )

    assert [message.type for message in msgs] == ["system", "human", "ai", "human"]
    assert msgs[-1].content == "你们卖猫粮吗？"


def test_extract_prompt_keeps_user_text_literal_in_human_message():
    user_text = "订单 MH1 坏了；模板符号 {unexpected} 和 {{escaped}} 都是原文。"
    msgs = EXTRACT_PROMPT.format_messages(text=user_text)

    assert [message.type for message in msgs] == ["system", "human"]
    assert msgs[-1].content == user_text


def test_agent_prompt_renders_history_without_reinterpreting_user_text():
    user_text = "订单 1001 到哪了？{literal}"
    msgs = AGENT_PROMPT.format_messages(history=[HumanMessage(user_text)])

    assert [message.type for message in msgs] == ["system", "human"]
    assert msgs[-1].content == user_text


def test_behavioral_prompt_cases_are_labeled_for_task_8():
    cases_path = Path(__file__).parent / "data" / "prompt_cases.json"
    data = json.loads(cases_path.read_text(encoding="utf-8"))

    assert data["status"] == "pending_task_8_real_model_evaluation"
    assert len(data["cases"]) == 4
    assert all(case["expected_criteria"] for case in data["cases"])
