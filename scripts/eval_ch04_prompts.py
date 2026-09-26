"""Small labeled live check for Chapter 4 RAG prompt behavior."""

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.core.selfcheck import check_sufficient


DATA = Path(__file__).resolve().parents[1] / "tests" / "data"


class _Faithful(BaseModel):
    faithful: bool = Field(description="回答的关键事实是否有证据支持")
    reason: str = Field(default="", description="判断依据")


def _samples(filename: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / filename).read_text(encoding="utf-8").splitlines() if line]


async def main() -> None:
    checks = _samples("selfcheck_samples.jsonl")
    check_passed = 0
    for sample in checks:
        result = await check_sufficient(sample["query"], sample["evidence"])
        passed = result["useful"] is sample["useful"]
        check_passed += passed
        print(f"selfcheck {'OK' if passed else 'MISS'} {sample['query']} -> {result}")

    prompts = _samples("rag_prompt_samples.jsonl")
    answer_chain = RAG_ANSWER_PROMPT | get_chat_model()
    faithful_chain = FAITHFULNESS_PROMPT | get_chat_model(temperature=0).with_structured_output(
        _Faithful, method=get_settings().structured_output_method,
    )
    prompt_passed = 0
    for sample in prompts:
        if sample["kind"] == "answer":
            response = await answer_chain.ainvoke({"query": sample["query"], "evidence": sample["evidence"]})
            content = str(response.content)
            passed = all(text in content for text in sample["must_contain"])
            passed = passed and not any(text in content for text in sample["forbid_any"])
            print(f"answer {'OK' if passed else 'MISS'} {sample['query']} -> {content}")
        else:
            result: _Faithful = await faithful_chain.ainvoke({
                "evidence": sample["evidence"], "answer": sample["answer"],
            })
            passed = result.faithful is sample["faithful"]
            print(f"faithfulness {'OK' if passed else 'MISS'} -> {result.model_dump()}")
        prompt_passed += passed
    print(f"Task 8 labeled evaluation: selfcheck {check_passed}/{len(checks)}, prompts {prompt_passed}/{len(prompts)}")
    if check_passed < 4 or prompt_passed != len(prompts):
        raise SystemExit("Task 8 prompt evaluation failed")


if __name__ == "__main__":
    asyncio.run(main())
