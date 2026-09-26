"""Offline socket integration: verify SSE arrives before fake producer resumes."""

import asyncio
import json
import socket
import sys
from pathlib import Path

import httpx
import uvicorn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.graph.runtime import GraphRuntime
from app.main import create_app
from langchain_core.messages import AIMessageChunk


class GatedFakeProducer:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, user_id, message, conversation_id, *, model):
        self.entered.set()
        yield "messages", (AIMessageChunk(content="首帧"), {"langgraph_node": "final_answer"})
        await self.release.wait()
        yield "messages", (AIMessageChunk(content="第二帧"), {"langgraph_node": "final_answer"})
        yield "updates", {"log_turn": {"conversation_id": 1}}


async def _next_frame(lines) -> str:
    parts = []
    async for line in lines:
        if not line:
            if parts:
                return "\n".join(parts)
        else:
            parts.append(line)
    raise AssertionError("SSE 提前结束")


async def check_socket_stream() -> None:
    fake = GatedFakeProducer()
    original_stream = GraphRuntime.prepare_stream_turn
    async def prepare(self, user_id, message, conversation_id, *, model):
        return fake.stream(user_id, message, conversation_id, model=model)
    GraphRuntime.prepare_stream_turn = prepare
    settings = Settings(
        _env_file=None,
        chat_model="offline-fake",
        chat_base_url="http://127.0.0.1/unused",
        chat_api_key="offline-only",
        token_budget=32768,
    )
    app = create_app(settings=settings, model=object())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="on", log_config=None, access_log=False))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(15):
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:
                for attempt in range(100):
                    if server_task.done():
                        await server_task
                        raise AssertionError("Uvicorn 提前退出")
                    try:
                        health = await client.get("/health")
                        assert health.status_code == 200
                        break
                    except httpx.ConnectError:
                        await asyncio.sleep(0.05)
                else:
                    raise AssertionError("Uvicorn /health 未就绪")

                async with client.stream("POST", "/api/chat", json={"user_id": "offline-socket", "message": "你好"}) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream")
                    lines = response.aiter_lines()
                    first = await asyncio.wait_for(_next_frame(lines), 5)
                    assert first.startswith("data: ")
                    assert json.loads(first[6:]) == {"delta": "首帧"}
                    assert fake.entered.is_set()
                    assert not fake.release.is_set(), "首帧到达前生产者已释放"
                    fake.release.set()
                    second = await asyncio.wait_for(_next_frame(lines), 5)
                    done = await asyncio.wait_for(_next_frame(lines), 5)
                    terminal = await asyncio.wait_for(_next_frame(lines), 5)
                    assert json.loads(second[6:]) == {"delta": "第二帧"}
                    assert json.loads(done[6:]) == {"event": "done", "conversation_id": 1}
                    assert terminal == "data: [DONE]"
    finally:
        fake.release.set()
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, 5)
        except asyncio.TimeoutError:
            server.force_exit = True
            await asyncio.wait_for(server_task, 5)
        finally:
            sock.close()
            GraphRuntime.prepare_stream_turn = original_stream


if __name__ == "__main__":
    asyncio.run(check_socket_stream())
    print("PASS: offline socket integration，首帧在 fake producer 释放前抵达；未调用真实模型。")
