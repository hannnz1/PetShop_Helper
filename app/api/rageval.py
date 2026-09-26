"""Serve the latest saved RAG evaluation report to the local dashboard."""

import json
from pathlib import Path

from fastapi import APIRouter, Request


router = APIRouter(prefix="/api/rag-eval", tags=["rag-evaluation"])
REPORT_PATH = Path(__file__).resolve().parents[2] / "data/ch04/reports/rag_eval.json"


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
        "generation_done": report.get("generation") is not None,
        "best": _best(report["retrieval"]), "job": job,
    }
