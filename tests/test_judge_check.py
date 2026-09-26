"""The reviewed ledger is a repeatable judge-only regression set."""

import pytest

from scripts import judge_check


def _case(eval_id, status, citations=None, answer="回答[1]"):
    return {"eval_id": eval_id, "status": status, "citations": citations,
            "answer": answer}


def test_snapshot_roundtrip_and_review_labels():
    citations = [
        {"n": 2, "question": "退货", "answer": "签收后7天"},
        {"question": "价保", "answer": "下单后7天"},
    ]
    assert judge_check.evidence_from_citations(citations) == (
        "[2] 退货: 签收后7天\n[2] 价保: 下单后7天"
    )
    assert judge_check.expected_faithful("已解决") is False
    assert judge_check.expected_faithful("无需解决") is True
    assert judge_check.expected_faithful("未解决") is None
    rows = [_case("A1", "已解决", citations),
            _case("A2", "无需解决", citations),
            _case("A3", "未解决", citations),
            _case("A4", "已解决", [])]
    assert [row["eval_id"] for row in judge_check.gradable(rows)] == ["A1", "A2"]


@pytest.mark.asyncio
async def test_judge_retries_empty_response_then_counts_only_valid_verdicts():
    citations = [{"n": 1, "question": "邮费", "answer": "满99包邮"}]
    rows = [_case("A1", "已解决", citations), _case("A2", "无需解决", citations)]
    calls = {"A1": 0, "A2": 0}

    async def fake_judge(evidence, answer):
        eval_id = "A1" if answer == "编造[1]" else "A2"
        calls[eval_id] += 1
        if eval_id == "A1" and calls[eval_id] == 1:
            return None
        if eval_id == "A2":
            raise RuntimeError("upstream empty")
        return {"faithful": False, "reason": "证据未写该事实"}

    rows[0]["answer"] = "编造[1]"
    results = await judge_check.check_cases(rows, fake_judge)
    assert calls == {"A1": 2, "A2": 2}
    assert results[0]["agree"] is True and results[0]["actual"] is False
    assert results[1]["actual"] is None and results[1]["error"] == "RuntimeError"
    assert judge_check.score(results) == {"agreed": 1, "graded": 1, "failed": 1}
