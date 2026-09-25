"""Stream two live chat turns under one persisted conversation."""

import argparse
import json
import os
import sys

import httpx

try:
    from scripts.eval_extract import ConfigurationError, live_client, require_live_config
except ModuleNotFoundError:  # python scripts/demo_chat.py
    from eval_extract import ConfigurationError, live_client, require_live_config


class StreamError(RuntimeError):
    pass


def stream_turn(client: httpx.Client, user_id: str, conversation_id: int | None, message: str, emit) -> tuple[str, int]:
    chunks = []
    returned_id = None
    try:
        with client.stream("POST", "/api/chat", json={"user_id": user_id, "conversation_id": conversation_id, "message": message}) as response:
            response.raise_for_status()
            if not response.headers.get("content-type", "").lower().startswith("text/event-stream"):
                raise StreamError("响应不是 text/event-stream")
            event = []
            done = False
            for line in response.iter_lines():
                if line:
                    event.append(line)
                    continue
                if not event:
                    continue
                frame = event
                event = []
                if any(part == "event: error" for part in frame):
                    raise StreamError("服务端 SSE error 事件")
                data = [part[6:] for part in frame if part.startswith("data: ")]
                if len(data) != 1:
                    raise StreamError("SSE 帧缺少单一 data 字段")
                if data[0] == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data[0])
                except json.JSONDecodeError as exc:
                    raise StreamError("SSE data 不是有效 JSON") from exc
                if not isinstance(obj, dict):
                    raise StreamError("SSE JSON 格式错误")
                if obj.get("event") == "tool" and isinstance(obj.get("name"), str):
                    continue
                if obj.get("event") == "done" and isinstance(obj.get("conversation_id"), int):
                    returned_id = obj["conversation_id"]
                    continue
                if not isinstance(obj.get("delta"), str):
                    raise StreamError("SSE delta 格式错误")
                chunks.append(obj["delta"])
                emit(obj["delta"])
            if event or not done or returned_id is None or not chunks:
                raise StreamError("SSE 流截断、缺少会话完成帧或没有内容")
    except httpx.HTTPStatusError as exc:
        raise StreamError(f"HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise StreamError(f"连接/传输失败: {type(exc).__name__}") from exc
    return "".join(chunks), returned_id


def run_demo(client: httpx.Client, user_id: str) -> None:
    turns = [
        ("第一轮", "我叫王小明,昨天买了你们的智能猫砂盆"),
        ("第二轮", "还记得我叫什么、买了什么吗?"),
    ]
    conversation_id = None
    for title, message in turns:
        print(f"\n=== {title}: {message} ===", flush=True)
        _, conversation_id = stream_turn(client, user_id, conversation_id, message, lambda delta: print(delta, end="", flush=True))
        print(flush=True)
    print("请人工核对第二轮是否正确复述「王小明」和「智能猫砂盆」；脚本仅验证 SSE 协议完整。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        require_live_config()
        with live_client(args.base_url) as client:
            run_demo(client, f"demo-{os.getpid()}")
    except (ConfigurationError, StreamError) as exc:
        print(f"演示失败: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
