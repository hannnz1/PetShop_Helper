"""Evaluate labeled chapter-two tool choice through the real HTTP agent route."""

import argparse
import asyncio
import json
from uuid import uuid4

import httpx


CASES = [
    ("logistics", "订单 1001 的物流到哪了", "query_logistics"),
    ("policy", "退货政策是什么", "query_faq"),
    ("postage", "邮费是多少", "query_faq"),
    ("product", "iPhone 还有货吗", "query_product"),
    ("order", "订单 2002 多少钱", "query_order"),
    ("ticket", "我要投诉，给我登记一下", "create_ticket"),
    ("out_of_scope", "今天天气怎么样", None),
    ("greeting", "你好，你能帮我做什么？", None),
    ("missing_order", "我的订单到哪了？", None),
]


async def main(base_url: str) -> int:
    passed = 0
    postage_miss = None
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        for case_id, message, expected in CASES:
            response = await client.post(
                f"{base_url.rstrip('/')}/api/agent",
                json={"user_id": f"eval-{uuid4().hex[:16]}", "message": message},
            )
            if response.status_code != 200:
                print(json.dumps({"id": case_id, "http_status": response.status_code, "pass": False}, ensure_ascii=False))
                continue
            body = response.json()
            selected = [call["name"] for call in body["tool_calls"]]
            ok = selected == [expected] if expected else not selected
            passed += ok
            trace = []
            for result in body["tool_results"]:
                item = {"name": result["name"], "ok": result["ok"]}
                if case_id == "postage" and result["name"] == "query_faq":
                    try:
                        payload = json.loads(result["content"])
                        item["faq_hits"] = len(payload.get("hits", []))
                    except (ValueError, TypeError):
                        item["faq_hits"] = None
                trace.append(item)
            if case_id == "postage":
                keywords = [call["args"].get("keyword") for call in body["tool_calls"] if call["name"] == "query_faq"]
                faq = [item for item in trace if item["name"] == "query_faq"]
                postage_miss = keywords == ["邮费"] and len(faq) == 1 and faq[0]["ok"] is True and faq[0].get("faq_hits") == 0
            print(json.dumps({
                "id": case_id, "expected_tool": expected, "selected": selected,
                "tool_args": [call["args"] for call in body["tool_calls"]],
                "pass": ok, "faq_literal_miss": postage_miss if case_id == "postage" else None,
                "trace": trace, "answer": body["answer"],
            }, ensure_ascii=False))
    print(f"tool_choice={passed}/{len(CASES)}")
    print(f"postage_literal_miss={'PASS' if postage_miss else 'FAIL'}")
    return 0 if passed == len(CASES) and postage_miss else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.base_url)))
