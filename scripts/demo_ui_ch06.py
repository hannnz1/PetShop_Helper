"""Offline Ch06 browser demo; serves synthetic SSE and records action POSTs."""

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


app = FastAPI()
CALLS = []
PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def stream(*events):
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    if events and events[-1].get("event") == "done":
        yield "data: [DONE]\n\n"


@app.get("/")
async def index():
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


@app.get("/_calls")
async def calls():
    return JSONResponse(CALLS)


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    CALLS.append({"path": "/api/chat", "payload": data})
    return StreamingResponse(stream({"event": "interrupt", "kind": "select_order",
                                    "conversation_id": 6001, "orders": [
                                        {"order_id": "1001", "product": "演示猫粮", "status": "已签收", "amount": 88}]}),
                             media_type="text/event-stream")


@app.post("/api/actions/resume")
async def resume(request: Request):
    data = await request.json()
    CALLS.append({"path": "/api/actions/resume", "payload": data})
    return StreamingResponse(stream(
        {"delta": "订单可以申请退款，请确认后提交。"},
        {"event": "actions", "items": [{"type": "refund_form", "draft":
            {"order_id": "1001", "reason": "质量问题"}}]},
        {"event": "done", "conversation_id": 6001},
    ), media_type="text/event-stream")


@app.post("/api/actions/create-refund")
async def create_refund(request: Request):
    data = await request.json()
    CALLS.append({"path": "/api/actions/create-refund", "payload": data})
    return {"refund_no": "R-DEMO-1", "status": "待人工审核"}
