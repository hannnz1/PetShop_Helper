"""Collect actual model outputs for human review of the chapter-two agent prompt."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import AGENT_PROMPT
from app.tools.registry import get_all_tools


CASES = Path(__file__).resolve().parents[1] / "tests/data/agent_prompt_cases.json"


def main() -> int:
    settings = get_settings()
    if settings.chat_model != "glm-5.2":
        print("需要在 .env 中配置 glm-5.2 才能评估 Task 10。")
        return 2

    model = get_chat_model()
    with_tools = model.bind_tools(get_all_tools())
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        messages = AGENT_PROMPT.format_messages(history=[HumanMessage(case["user"])])
        if "final_tool" in case:
            call_id = f"eval_{case['id']}"
            messages.extend(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": case["final_tool"], "args": {}, "id": call_id}
                        ],
                    ),
                    ToolMessage(
                        content=json.dumps(case["final_result"], ensure_ascii=False),
                        tool_call_id=call_id,
                        status="error" if case["final_result"].get("ok") is False else "success",
                    ),
                ]
            )
            response = model.invoke(messages)  # 收敛轮不绑定工具。
            actual = {"answer": response.content}
        else:
            response = with_tools.invoke(messages)
            actual = {
                "tool_calls": [
                    {"name": call["name"], "args": call["args"]}
                    for call in response.tool_calls
                ],
                "answer_before_tool": response.content,
            }
        print(json.dumps({"id": case["id"], "actual": actual, "criteria": case["criteria"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
