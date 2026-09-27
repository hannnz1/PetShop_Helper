"""Chapter 8 annotated cases. Offline validates labels; --live uses configured services."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "ch08_eval.jsonl"
EXPECTED = {"ticket_missing", "ticket_preview", "ticket_confirm", "ticket_cancel", "logistics"}


def load_cases() -> dict[str, dict]:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cases = {row["id"]: row for row in rows}
    if len(rows) != len(cases) or set(cases) != EXPECTED:
        raise ValueError("Chapter 8 evaluation needs five unique annotated cases")
    if any(not isinstance(row.get("expect"), str) or not row["expect"] for row in rows):
        raise ValueError("Every Chapter 8 case needs an expected outcome")
    return cases


async def _sse(client: httpx.AsyncClient, path: str, payload: dict) -> tuple[list[dict], str]:
    events: list[dict] = []
    async with client.stream("POST", path, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            value = json.loads(line[6:])
            if isinstance(value, dict):
                events.append(value)
    answer = "".join(event.get("delta", "") for event in events)
    return events, answer


async def _count_tickets(conversation_id: int) -> int:
    from sqlalchemy import func, select

    from app.db.base import async_session
    from app.db.models import Ticket

    async with async_session() as session:
        return int(await session.scalar(select(func.count()).select_from(Ticket).where(
            Ticket.conversation_id == conversation_id)) or 0)


async def _ticket_number(conversation_id: int) -> str | None:
    from sqlalchemy import select

    from app.db.base import async_session
    from app.db.models import Ticket

    async with async_session() as session:
        return await session.scalar(select(Ticket.ticket_no).where(
            Ticket.conversation_id == conversation_id).limit(1))


async def _last_tool_audit(conversation_id: int, name: str) -> tuple[str, str | None] | None:
    from sqlalchemy import select

    from app.db.base import async_session
    from app.db.models import ToolAuditLog

    async with async_session() as session:
        row = (await session.execute(select(
            ToolAuditLog.status, ToolAuditLog.result_summary,
        ).where(
            ToolAuditLog.conversation_id == conversation_id,
            ToolAuditLog.tool_name == name,
        ).order_by(ToolAuditLog.id.desc()).limit(1))).first()
    return (row.status, row.result_summary) if row else None


def _interrupt(events: list[dict], kind: str) -> dict | None:
    return next((event for event in events if event.get("event") == "interrupt"
                 and event.get("kind") == kind), None)


def _confirmed_ticket_passed(count: int, ticket_no: str | None, answer: str,
                             audit: tuple[str, str | None] | None) -> bool:
    return bool(count == 1 and ticket_no and ticket_no in answer
                and audit and audit[0] == "成功")


def _logistics_passed(tools: set[str | None], answer: str,
                      audit: tuple[str, str | None] | None) -> bool:
    if not {"query_order", "query_logistics"} <= tools or not audit or audit[0] != "成功":
        return False
    try:
        payload = json.loads(audit[1]) if audit[1] else {}
    except json.JSONDecodeError:
        return False
    status = payload.get("status") if isinstance(payload, dict) else None
    return status in {"已揽件", "运输中", "派送中", "已签收"} and status in answer


async def evaluate_live(base_url: str) -> dict:
    results = []
    async with httpx.AsyncClient(base_url=base_url, timeout=90.0, trust_env=False) as client:
        user_id = f"ch08-eval-{uuid4().hex[:12]}"
        events, answer = await _sse(client, "/api/chat", {
            "user_id": user_id, "conversation_id": None, "message": "帮我建个工单",
        })
        cid = next((e.get("conversation_id") for e in events if e.get("event") == "done"), None)
        missing_ok = (isinstance(cid, int) and _interrupt(events, "confirm_ticket") is None
                      and await _count_tickets(cid) == 0
                      and any(word in answer for word in ("问题", "描述", "具体")))
        results.append({"id": "ticket_missing", "passed": bool(missing_ok)})

        events, _ = await _sse(client, "/api/chat", {
            "user_id": user_id, "conversation_id": cid,
            "message": "帮我建售后工单，猫砂盆漏电",
        })
        preview = _interrupt(events, "confirm_ticket")
        preview_ok = bool(preview and preview.get("preview", {}).get("description")
                          and preview.get("preview", {}).get("ticket_type")
                          and await _count_tickets(cid) == 0)
        results.append({"id": "ticket_preview", "passed": preview_ok})

        if preview_ok:
            events, answer = await _sse(client, "/api/actions/resume", {
                "user_id": user_id, "conversation_id": cid, "confirmed": True,
            })
            ticket_no = await _ticket_number(cid)
            ticket_audit = await _last_tool_audit(cid, "create_ticket")
            confirm_ok = _confirmed_ticket_passed(
                await _count_tickets(cid), ticket_no, answer, ticket_audit,
            )
        else:
            confirm_ok = False
        results.append({"id": "ticket_confirm", "passed": confirm_ok})

        cancel_user = f"ch08-eval-{uuid4().hex[:12]}"
        events, _ = await _sse(client, "/api/chat", {
            "user_id": cancel_user, "conversation_id": None,
            "message": "帮我建售后工单，猫砂盆漏电",
        })
        cancel_preview = _interrupt(events, "confirm_ticket")
        cancel_cid = cancel_preview.get("conversation_id") if cancel_preview else None
        if cancel_cid:
            await _sse(client, "/api/actions/resume", {
                "user_id": cancel_user, "conversation_id": cancel_cid, "confirmed": False,
            })
            cancel_ok = (await _count_tickets(cancel_cid) == 0
                         and (await _last_tool_audit(cancel_cid, "create_ticket") or (None,))[0]
                         == "权限拒绝")
        else:
            cancel_ok = False
        results.append({"id": "ticket_cancel", "passed": cancel_ok})

        events, answer = await _sse(client, "/api/chat", {
            "user_id": f"ch08-eval-{uuid4().hex[:12]}", "conversation_id": None,
            "message": "我的订单 1001 的物流到哪了",
        })
        tools = {e.get("name") for e in events if e.get("event") == "tool"}
        logistics_cid = next((e.get("conversation_id") for e in events
                              if e.get("event") == "done"), None)
        audit = await _last_tool_audit(logistics_cid, "query_logistics") if logistics_cid else None
        results.append({"id": "logistics", "passed": _logistics_passed(tools, answer, audit)})
    return {"status": "completed", "results": results,
            "passed": sum(item["passed"] for item in results), "total": len(results)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call running app, model and MCP services")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    cases = load_cases()
    if not args.live:
        print(json.dumps({"status": "pending_upstream", "validated_samples": len(cases),
                          "cases": sorted(cases)}, ensure_ascii=False))
        return 0
    result = asyncio.run(evaluate_live(args.base_url))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
