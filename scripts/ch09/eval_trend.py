"""Archive one Chapter 4 evaluation and record each strategy in MySQL."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from app.config import get_settings
from app.db.observability import record_eval_run
from app.observability.eval_trend import extract_summary

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "tests/data/eval_ch04.jsonl"
REPORT_DIR = ROOT / "data/ch09/eval-runs"
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def upstream_unavailable_reason() -> str | None:
    settings = get_settings()
    if "glm-5.2" not in settings.chat_model.lower():
        return "glm-5.2 model not configured"
    if settings.siliconflow_api_key is None:
        return "SiliconFlow embedding or rerank credential unavailable"
    parsed = urlparse(settings.milvus_uri)
    if parsed.scheme in ("http", "https", "tcp"):
        try:
            with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 19530), timeout=1):
                pass
        except OSError:
            return "Milvus unavailable"
    elif not Path(settings.milvus_uri).exists():
        return "Milvus unavailable"
    return None


def _git_sha() -> str:
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


async def run_once(*, run_id: str | None = None, report_file: Path | None = None,
                   live: bool = False) -> dict:
    run_id = run_id or str(uuid4())
    # Four strategy rows append ":hybrid_rerank" to this ID in a VARCHAR(64).
    if not run_id or len(run_id) > 50 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in run_id):
        raise ValueError("run_id must use 1-50 ASCII letters, digits, hyphens or underscores")
    dataset_hash = hashlib.sha256(SAMPLES.read_bytes()).hexdigest()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"{run_id}.json"
    if destination.exists():
        report = json.loads(destination.read_text(encoding="utf-8"))
        if report.get("meta", {}).get("dataset_hash") != dataset_hash:
            raise ValueError("run_id already belongs to a different dataset")
    else:
        started = datetime.now(timezone.utc).isoformat()
        if report_file is not None:
            report = json.loads(report_file.read_text(encoding="utf-8"))
            report.setdefault("meta", {})["sample_mode"] = True
        else:
            reason = upstream_unavailable_reason() if live else "live evaluation not requested"
            if reason:
                report = {"meta": {"status": "partial", "question_count": len(SAMPLES.read_text(encoding="utf-8").splitlines()),
                                   "evaluated_at": datetime.now(timezone.utc).isoformat(),
                                   "unavailable_reason": reason},
                          "retrieval": {}, "evidence_coverage": {}, "generation": None}
            else:
                from scripts import eval_ch04
                report = await eval_ch04.main(use_cache=False)
        report.setdefault("meta", {})
        report["meta"].update(dataset_hash=dataset_hash, git_sha=_git_sha(),
                              started_at=started,
                              evaluated_at=report["meta"].get("evaluated_at") or datetime.now(timezone.utc).isoformat(),
                              report_path=str(destination),
                              judge_model=report["meta"].get("judge_model") or get_settings().chat_model)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)

    for strategy in STRATEGIES:
        await record_eval_run(f"{run_id}:{strategy}", dataset_hash, strategy, report)
    statuses = {strategy: extract_summary(report, strategy)["status"] for strategy in STRATEGIES}
    return {"run_id": run_id, "dataset_hash": dataset_hash,
            "report_path": str(destination),
            "status": "passed" if all(v == "passed" for v in statuses.values()) else "pending_upstream",
            "strategies": statuses}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--report-file", type=Path, help="import a labeled local report offline")
    parser.add_argument("--live", action="store_true", help="run Chapter 4 evaluator when dependencies are ready")
    args = parser.parse_args(argv)
    result = asyncio.run(run_once(run_id=args.run_id, report_file=args.report_file, live=args.live))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
