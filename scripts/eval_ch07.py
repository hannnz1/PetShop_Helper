"""Deterministic Chapter 7 acceptance; never opens a chat-model connection."""

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.core.context_budget import FixedCosts, derive_budget
from app.core.context_layers import build_model_context
from app.core.memory import estimate_tokens
from app.db.repository import ContextSnapshot, SummarySegment, VisibleMessage


REPORT = ROOT / "data/ch07/reports/offline_eval.json"


def config(**overrides):
    return Settings(_env_file=None, chat_model="offline-test",
                    chat_base_url="http://127.0.0.1:9/v1", chat_api_key="offline-test",
                    **overrides)


def default_twenty_turns():
    settings = config()
    budget = derive_budget(settings, FixedCosts(678, 1700, 250, 250))
    turns = [row for _ in range(20) for row in
             (HumanMessage("请问商品有货吗？"), AIMessage("您好，请提供商品名称。"))]
    assert estimate_tokens(turns, chars_per_token=settings.cjk_chars_per_token) <= budget.layer1
    return {"turns": 20, "layer1_budget": budget.layer1}


def demo_cascade_and_early_order():
    settings = config(model_context_window=18000, max_output_tokens=2000,
                      max_user_input_tokens=2000, max_agent_steps=3,
                      tool_result_max_tokens=1200, rerank_top_k=5)
    budget = derive_budget(settings, FixedCosts(678, 1700, 250, 250))
    assert budget.history_total == 5650
    rows = []
    for turn in range(1, 23):
        rows.extend((VisibleMessage(turn * 2 - 1, "user", f"第 {turn} 轮：" + "猫砂" * 70),
                     VisibleMessage(turn * 2, "assistant", "已记录：" + "请核实" * 70)))
    # Synthetic post-worker state: the first complete turn has been summarized,
    # middle completed turns downgraded, and the latest turns remain verbatim.
    segment = SummarySegment(1, 1, 2, "最早的订单 1001 是猫砂，退款条件尚未确认。")
    snapshot = ContextSnapshot(7, 2, 28, tuple(rows), (segment,))
    model = build_model_context(snapshot, [HumanMessage("最早订单怎么样？")],
                                "最早订单怎么样？", "", "系统", budget, settings=settings)
    assert "1001" in model.injected_summary
    assert "退款条件尚未确认" in model.injected_summary
    assert any(row.layer == 2 for row in model.window_rows)
    assert any(row.layer == 1 for row in model.window_rows)
    assert model.messages[-1].name == "verified_context"
    return {"turns": 22, "history_budget": budget.history_total,
            "layer1_budget": budget.layer1, "layer2_budget": budget.layer2}


def labeled_summary_cases():
    path = ROOT / "tests/data/ch07_summary_cases.jsonl"
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        reference = case["reference"]
        assert all(phrase in reference for phrase in case["must_include"]), case["id"]
        assert all(phrase not in reference for phrase in case["must_exclude"]), case["id"]
        if not case["must_include"]:
            assert reference == "无明确事实", case["id"]
        ids.append(case["id"])
    assert ids
    return {"labeled_case_ids": ids, "scope": "reference labels only; no model generation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", required=True)
    args = parser.parse_args()
    del args
    results = []
    for case_id, check in (("default_20_turns", default_twenty_turns),
                           ("demo_22_turn_cascade", demo_cascade_and_early_order),
                           ("labeled_summary_references", labeled_summary_cases)):
        try:
            details = check()
            results.append({"id": case_id, "status": "pass", "details": details})
        except Exception as exc:
            results.append({"id": case_id, "status": "fail", "details": str(exc)})
    for case_id in ("real_glm_default_20_turns", "real_glm_demo_cascade",
                    "real_glm_early_order_recall", "real_glm_summary_quality_and_usage"):
        results.append({"id": case_id, "status": "pending_upstream",
                        "details": "glm balance unavailable; no paid request made"})
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("pass", "fail", "pending_upstream")}
    report = {"mode": "offline", "counts": counts, "cases": results,
              "limits": ["Synthetic cascade validates rendering, not async scheduling or real model quality.",
                         "SSE and page switching are covered by separate regression/browser acceptance."]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
