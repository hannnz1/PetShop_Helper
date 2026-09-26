"""Replay reviewed faithfulness cases against the current judge without retrieval."""

import asyncio

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT
from app.db import repository


class _Faith(BaseModel):
    faithful: bool = Field(description="回答的具体事实是否忠实于证据")
    reason: str = Field(description="一句话判断依据")


def evidence_from_citations(citations: list[dict] | None) -> str:
    """Rebuild eval_ch04._evidence exactly, tolerating older snapshots without n."""
    return "\n".join(
        f"[{item.get('n', index)}] {item.get('question', '')}: {item.get('answer', '')}"
        for index, item in enumerate(citations or [], 1)
    )


def expected_faithful(status: str) -> bool | None:
    return {"已解决": False, "无需解决": True}.get(status)


def gradable(rows: list[dict]) -> list[dict]:
    return [row for row in rows if expected_faithful(row["status"]) is not None
            and row.get("citations")]


async def _one(row: dict, judge, gate: asyncio.Semaphore) -> dict:
    expected = expected_faithful(row["status"])
    result = {"id": row["eval_id"], "status": row["status"], "expected": expected,
              "actual": None, "agree": False, "reason": "", "error": None}
    async with gate:
        for _ in range(2):
            try:
                verdict = await judge(evidence_from_citations(row["citations"]), row["answer"])
                if verdict is None:
                    raise ValueError("empty judge response")
                actual = verdict["faithful"] if isinstance(verdict, dict) else verdict.faithful
                if type(actual) is not bool:
                    raise ValueError("invalid judge verdict")
                result["actual"] = actual
                result["agree"] = actual == expected
                result["reason"] = (
                    verdict.get("reason", "") if isinstance(verdict, dict) else verdict.reason
                )
                return result
            except Exception as exc:  # noqa: BLE001 - one failed case must not hide others
                result["error"] = type(exc).__name__
        return result


async def check_cases(rows: list[dict], judge, concurrency: int = 3) -> list[dict]:
    gate = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(_one(row, judge, gate) for row in gradable(rows)))
    return sorted(results, key=lambda item: item["id"])


def score(results: list[dict]) -> dict[str, int]:
    valid = [item for item in results if item["actual"] is not None]
    return {"agreed": sum(item["agree"] for item in valid),
            "graded": len(valid), "failed": len(results) - len(valid)}


async def _reviewed_rows() -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    page = 1
    counts: dict[str, int] = {}
    while True:
        listing = await repository.list_faith_cases(page=page, size=100)
        rows.extend(listing["items"])
        counts = listing["counts"]
        if page >= listing["pages"]:
            return rows, counts
        page += 1


async def main() -> int:
    rows, counts = await _reviewed_rows()
    cases = gradable(rows)
    if not cases:
        print("没有已人工处置且带证据快照的个案；裁判回归尚不能验收。")
        return 2
    model = get_chat_model(temperature=0).with_structured_output(
        _Faith, method=get_settings().structured_output_method,
    )
    chain = FAITHFULNESS_PROMPT | model

    async def judge(evidence: str, answer: str):
        return await chain.ainvoke({"evidence": evidence, "answer": answer})

    results = await check_cases(cases, judge)
    summary = score(results)
    print(f"裁判回归：台账 {len(rows)} 条；可考 {len(cases)} 条；处置分布 {counts}")
    for item in results:
        label = "调用失败" if item["actual"] is None else "一致" if item["agree"] else "不一致"
        print(f"{item['id']} {item['status']} 期望={item['expected']} 实际={item['actual']} {label}")
        if label != "一致":
            print(f"  理由：{item['error'] or item['reason']}")
    print(f"一致 {summary['agreed']}/{summary['graded']}；调用失败 {summary['failed']}")
    return 0 if summary["graded"] and not summary["failed"] and summary["agreed"] == summary["graded"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
