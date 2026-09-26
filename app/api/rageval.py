"""Serve the latest saved RAG evaluation report to the local dashboard."""

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.db import repository


router = APIRouter(prefix="/api/rag-eval", tags=["rag-evaluation"])
REPORT_PATH = Path(__file__).resolve().parents[2] / "data/ch04/reports/rag_eval.json"


FaithStatus = Literal["未解决", "已解决", "无需解决"]


class FaithStatusBody(BaseModel):
    status: FaithStatus
    resolution: str | None = None


def _best(retrieval: dict) -> dict | None:
    ranked = []
    for strategy, buckets in retrieval.items():
        total = sum(item.get("count", 0) for item in buckets.values())
        if total:
            mrr = sum(item.get("mrr", 0) * item.get("count", 0) for item in buckets.values()) / total
            ranked.append((mrr, strategy))
    if not ranked:
        return None
    mrr, strategy = max(ranked)
    return {"strategy": strategy, "mrr": mrr}


def _report() -> dict | None:
    try:
        data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("retrieval"), dict):
        return None
    if not isinstance(data.get("evidence_coverage"), dict) or not isinstance(data.get("meta"), dict):
        return None
    return data


def _hallucination(report: dict | None, counts: dict[str, int], statuses: dict[str, str]) -> dict:
    """Score this report's findings, keeping historical ledger counts separate."""
    ledger = {"total": sum(counts.values()), **counts}
    empty = {"evaluated": None, "graded": None, "absent": None,
             "refusal_missed": None, "cases_judged": 0, "cases_confirmed": 0,
             "dismissed": 0, "pending": 0, "judged": 0, "confirmed": 0,
             "judged_rate": None, "confirmed_rate": None, "ledger": ledger}
    if (not report or report.get("meta", {}).get("status") == "partial"
            or not isinstance(report.get("generation"), dict)):
        return empty
    evaluated = int(report.get("meta", {}).get("question_count") or 0)
    buckets = report.get("retrieval", {}).get("hybrid_rerank", {})
    graded = sum(int(item.get("count", 0)) for item in buckets.values())
    absent = max(0, evaluated - graded)
    generation = report["generation"]
    case_ids = {
        item.get("id") for item in generation.get("faithfulness_cases", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    refusal_rate = generation.get("refusal_rate")
    refusal_missed = (
        round(absent * (1 - refusal_rate))
        if isinstance(refusal_rate, (int, float)) and 0 <= refusal_rate <= 1 else None
    )
    cases_confirmed = sum(statuses.get(case_id) == "已解决" for case_id in case_ids)
    dismissed = sum(statuses.get(case_id) == "无需解决" for case_id in case_ids)
    pending = len(case_ids) - cases_confirmed - dismissed
    judged = len(case_ids) + refusal_missed if refusal_missed is not None else None
    confirmed = cases_confirmed + refusal_missed if refusal_missed is not None else None
    return {
        "evaluated": evaluated, "graded": graded, "absent": absent,
        "refusal_missed": refusal_missed, "cases_judged": len(case_ids),
        "cases_confirmed": cases_confirmed, "dismissed": dismissed,
        "pending": pending, "judged": judged, "confirmed": confirmed,
        "judged_rate": round(judged / evaluated, 4) if judged is not None and evaluated else None,
        "confirmed_rate": round(confirmed / evaluated, 4) if confirmed is not None and evaluated else None,
        "ledger": ledger,
    }


@router.get("/overview")
def overview(request: Request) -> dict:
    runner = getattr(request.app.state, "jobs", None)
    job = runner.status("eval-rag") if runner else {"name": "eval-rag", "state": "idle", "heavy": True}
    report = _report()
    if report is None:
        return {"present": False, "job": job, "generation_done": False}
    return {
        "present": True, "meta": report["meta"],
        "retrieval": report["retrieval"],
        "evidence_coverage": report["evidence_coverage"],
        "generation": report.get("generation"),
        "generation_done": (report.get("generation") is not None
                            and report["meta"].get("status") != "partial"),
        "generation_error_count": len((report.get("generation") or {}).get("errors") or []),
        "best": _best(report["retrieval"]), "job": job,
    }


@router.get("/faith-cases")
async def faith_cases(
    status: FaithStatus | None = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> dict:
    listing = await repository.list_faith_cases(status=status, page=page, size=size)
    report = _report()
    cases = (report or {}).get("generation") or {}
    ids = [item.get("id") for item in cases.get("faithfulness_cases", [])
           if isinstance(item, dict) and isinstance(item.get("id"), str)]
    statuses = await repository.faith_case_status_map(ids)
    listing["hallucination"] = _hallucination(report, listing["counts"], statuses)
    return listing


@router.post("/faith-cases/{case_id}/status")
async def update_faith_case_status(case_id: int, body: FaithStatusBody) -> dict:
    try:
        row = await repository.set_faith_case_status(
            case_id, body.status, body.resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if row is None:
        raise HTTPException(status_code=404, detail="faith case not found")
    return row
