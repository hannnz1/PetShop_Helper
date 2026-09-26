"""Small labeled set for prompt-only faithfulness calibration (six model calls)."""

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT


DATA = Path(__file__).resolve().parents[1] / "tests/data/ch04_judge_boundaries.jsonl"


class Verdict(BaseModel):
    faithful: bool = Field(description="具体事实是否忠实于证据")
    reason: str = Field(description="一句话判断依据")


async def main() -> int:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]
    model = get_chat_model(temperature=0).with_structured_output(
        Verdict, method=get_settings().structured_output_method,
    )
    chain = FAITHFULNESS_PROMPT | model
    passed = 0
    for row in rows:
        try:
            result = await chain.ainvoke({"evidence": row["evidence"], "answer": row["answer"]})
            actual = result.faithful if result is not None else None
        except Exception as exc:  # noqa: BLE001 - show which sample could not be checked
            actual = None
            print(f"{row['id']}: 调用失败 {type(exc).__name__}")
        correct = actual is row["faithful"]
        passed += correct
        print(f"{row['id']}: {'通过' if correct else '未通过'}，期望={row['faithful']}，实际={actual}")
    print(f"裁判边界样例：{passed}/{len(rows)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
