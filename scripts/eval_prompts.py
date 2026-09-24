"""Collect real model answers for human review against labeled behavior cases."""

import argparse
import json
import sys
from pathlib import Path

import httpx

try:
    from scripts.demo_chat import StreamError, stream_turn
    from scripts.eval_extract import ConfigurationError, live_client, require_live_config, validate_ticket
except ModuleNotFoundError:  # python scripts/eval_prompts.py
    from demo_chat import StreamError, stream_turn
    from eval_extract import ConfigurationError, live_client, require_live_config, validate_ticket

CASES = Path(__file__).resolve().parents[1] / "tests/data/prompt_cases.json"


def evaluate(cases: list[dict], client: httpx.Client) -> int:
    errors = 0
    for index, case in enumerate(cases, 1):
        case_id = case["id"]
        endpoint = case.get("endpoint") or ("extract" if case_id.startswith("extract_") else "chat")
        print(f"\n=== #{index} {case_id} ({endpoint}) ===")
        print(f"输入: {case['prompt']}")
        try:
            if endpoint == "extract":
                response = client.post("/api/extract", json={"text": case["prompt"]})
                response.raise_for_status()
                answer = validate_ticket(response.json())
                print("实际输出: " + json.dumps(answer, ensure_ascii=False))
            elif endpoint == "chat":
                print("实际输出: ", end="", flush=True)
                stream_turn(client, f"prompt-case-{index}", case["prompt"], lambda delta: print(delta, end="", flush=True))
                print()
            else:
                raise ValueError(f"未知 endpoint: {endpoint}")
        except (httpx.HTTPError, StreamError, ValueError, TypeError, KeyError) as exc:
            errors += 1
            print(f"协议/请求失败: {type(exc).__name__}: {exc}")
        print("人工复核清单（逐条判断；脚本不评价语义质量）：")
        for criterion in case["expected_criteria"]:
            print(f"  [ ] {criterion}")
    print(f"\n请求/协议错误: {errors}/{len(cases)}；语义质量仍需人工复核。")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        require_live_config()
        dataset = json.loads(CASES.read_text(encoding="utf-8"))
        with live_client(args.base_url) as client:
            return evaluate(dataset["cases"], client)
    except (ConfigurationError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"无法运行真实评估: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
