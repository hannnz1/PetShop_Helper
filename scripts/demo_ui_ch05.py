"""Offline browser demo for the complaint action UI; no model or database."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()
_tickets = {}


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.post("/api/chat")
async def chat():
    async def events():
        for item in (
            {"delta": "很抱歉给您带来困扰。您可以选择转人工或填写工单。"},
            {"event": "actions", "items": [{"type": "transfer_human"},
                                          {"type": "create_ticket", "draft": {
                                              "ticket_type": "投诉", "description": "我要投诉服务"}}]},
            {"event": "done", "conversation_id": 1},
        ):
            yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


class Ticket(BaseModel):
    request_id: str


@app.post("/api/actions/create-ticket")
def ticket(req: Ticket):
    return {"ticket_no": _tickets.setdefault(req.request_id, "T-OFFLINE-DEMO")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8767)
