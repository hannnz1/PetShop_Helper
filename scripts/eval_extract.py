"""Evaluate labeled extraction cases against an explicitly configured live service."""

import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests/data/extract_samples.json"
VALID_TYPES = {"退款", "换货", "维修", "投诉", "其他"}


class ConfigurationError(ValueError):
    pass


def live_client(base_url: str) -> httpx.Client:
    """Create a live API client without inheriting machine proxy settings."""
    return httpx.Client(base_url=base_url, timeout=60, trust_env=False)


def require_live_config(path: Path = ROOT / ".env") -> None:
    """Require an explicit, non-example .env before any live request."""
    if not path.is_file():
        raise ConfigurationError(f"缺少 {path}；请复制 .env.example 为 .env 并填写真实 CHAT_* 配置。")
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"\'')
    for key in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        value = values.get(key, "")
        if not value or any(mark in value.lower() for mark in ("your_", "your-", "placeholder", "example", "changeme")):
            raise ConfigurationError(f"{path} 中 {key} 缺失或仍为示例值；请填写真实服务配置。")


def validate_ticket(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("响应不是 JSON 对象")
    if set(payload) != {"order_id", "request_type", "expected_solution"}:
        raise ValueError("响应字段不符合提取接口约定")
    if payload["order_id"] is not None and not isinstance(payload["order_id"], str):
        raise ValueError("order_id 类型错误")
    if payload["request_type"] not in VALID_TYPES:
        raise ValueError("request_type 无效")
    if not isinstance(payload["expected_solution"], str) or not payload["expected_solution"].strip():
        raise ValueError("expected_solution 无效")
    return payload


def evaluate(samples: list[dict], client: httpx.Client) -> int:
    failures = 0
    for index, sample in enumerate(samples, 1):
        try:
            response = client.post("/api/extract", json={"text": sample["text"]})
            response.raise_for_status()
            got = validate_ticket(response.json())
            expected = sample["expected"]
            passed = (got["order_id"] == expected["order_id"] and
                      got["request_type"] == expected["request_type"])
            print(f"[{'PASS' if passed else 'FAIL'}] #{index} {sample['text']}")
            print(f"  期望: {expected}; 实际: {got}")
        except httpx.HTTPStatusError as exc:
            passed = False
            print(f"[FAIL] #{index} HTTP {exc.response.status_code}")
        except httpx.RequestError as exc:
            passed = False
            print(f"[FAIL] #{index} 连接/传输失败: {type(exc).__name__}")
        except (ValueError, TypeError, KeyError) as exc:
            passed = False
            print(f"[FAIL] #{index} 无效响应: {exc}")
        failures += not passed
    print(f"\n{len(samples) - failures}/{len(samples)} 通过；expected_solution 请人工复核。")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        require_live_config()
        samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
        with live_client(args.base_url) as client:
            return evaluate(samples, client)
    except (ConfigurationError, OSError, json.JSONDecodeError) as exc:
        print(f"无法运行真实评估: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
