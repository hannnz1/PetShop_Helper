"""Offline preview-card demo: no model, database, or external service."""

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse


app = FastAPI()
calls: list[dict] = []
page = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def stream(*events):
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    if events and events[-1].get("event") == "done":
        yield "data: [DONE]\n\n"


@app.get("/")
def index():
    return FileResponse(page)


@app.get("/_calls")
def recorded_calls():
    return calls


@app.get("/api/conversations")
def conversations():
    return {"items": [], "next_cursor": None}


@app.post("/api/chat")
async def chat(request: Request):
    calls.append({"path": "/api/chat", "payload": await request.json()})
    return StreamingResponse(stream({
        "event": "interrupt", "kind": "confirm_ticket", "conversation_id": 6008,
        "preview": {"ticket_type": "售后", "description": "猫砂盆漏电"},
    }), media_type="text/event-stream")


@app.post("/api/actions/resume")
async def resume(request: Request):
    data = await request.json()
    calls.append({"path": "/api/actions/resume", "payload": data})
    answer = "演示工单 T-DEMO-8 已创建，待人工处理。" if data.get("confirmed") else "已取消，本次未建工单。"
    return StreamingResponse(stream({"delta": answer},
                                  {"event": "done", "conversation_id": 6008}),
                             media_type="text/event-stream")
