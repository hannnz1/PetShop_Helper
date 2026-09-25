"""Print the three chapter-two acceptance traces through the real JSON API."""

import argparse
import asyncio
import json
from uuid import uuid4

import httpx


async def main(base_url: str) -> int:
    cases = [
        ("物流工具", "订单 1001 的物流到哪了", "query_logistics"),
        ("退货政策", "退货政策是什么", "query_faq"),
        ("FAQ 字面漏召回", "邮费是多少", "query_faq"),
    ]
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        for label, message, expected_tool in cases:
            response = await client.post(
                f"{base_url.rstrip('/')}/api/agent",
                json={"user_id": f"demo-{uuid4().hex[:16]}", "message": message},
            )
            print(f"== {label} ==")
            if response.status_code != 200:
                print(f"HTTP {response.status_code}: {response.text[:200]}")
                return 1
            body = response.json()
            print(json.dumps(body, ensure_ascii=False, indent=2))
            if [call["name"] for call in body["tool_calls"]] != [expected_tool]:
                print("验收失败：选中工具与期望不符")
                return 1
            if label == "退货政策":
                faq_result = next((result for result in body["tool_results"] if result["name"] == "query_faq"), None)
                try:
                    hits = json.loads(faq_result["content"]).get("hits") if faq_result and faq_result["ok"] is True else None
                except (ValueError, TypeError):
                    hits = None
                if not hits or "7 天无理由" not in hits[0].get("answer", ""):
                    print("验收失败：退货政策 FAQ 未成功命中")
                    return 1
            if label == "FAQ 字面漏召回":
                keyword = body["tool_calls"][0]["args"].get("keyword")
                faq_result = next((result for result in body["tool_results"] if result["name"] == "query_faq"), None)
                try:
                    hits = json.loads(faq_result["content"]).get("hits") if faq_result and faq_result["ok"] is True else None
                except (ValueError, TypeError):
                    hits = None
                if keyword != "邮费" or hits != []:
                    print("验收失败：未观察到“邮费”原词的 FAQ 空命中")
                    return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.base_url)))
