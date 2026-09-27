"""Extract comparable Chapter 4 metrics without changing their definitions."""

from __future__ import annotations


def extract_summary(report: dict, strategy: str) -> dict:
    meta = report.get("meta") or {}
    retrieval = (report.get("retrieval") or {}).get(strategy) or {}
    evidence = (report.get("evidence_coverage") or {}).get(strategy) or {}
    generation = report.get("generation") or {}
    records = [row for row in generation.get("records") or []
               if row.get("strategy") == strategy]
    retrieval_valid = not meta.get("retrieval_error_count")
    metrics: dict[str, dict] = {}
    denominators: dict[str, int] = {}

    def add(name: str, value: float | None, count: int, valid: bool = True) -> None:
        denominators[name] = count
        metrics[name] = (
            {"value": float(value), "sample_count": count}
            if value is not None and count > 0 and valid
            else {"value": None, "sample_count": count, "status": "pending_upstream"}
        )

    for bucket, values in retrieval.items():
        count = int(values.get("count") or 0)
        for key in ("recall_at_5", "mrr"):
            add(f"{key}:{bucket}", values.get(key), count, retrieval_valid)
        add(f"evidence_coverage:{bucket}", evidence.get(bucket), count, retrieval_valid)
        coverage = (generation.get("answer_coverage") or {}).get(strategy) or {}
        completed = sum(row.get("bucket") == bucket and row.get("covered") is not None
                        for row in records)
        add(f"answer_coverage:{bucket}", coverage.get(bucket), count,
            completed == count and meta.get("status") == "complete")

    if strategy == "hybrid_rerank":
        refusal = [row for row in records if row.get("bucket") == "D_absent"]
        faith = [row for row in records if row.get("bucket") != "D_absent"]
        add("refusal_rate", generation.get("refusal_rate"), len(refusal),
            bool(refusal) and all("refused" in row for row in refusal))
        add("faithfulness", generation.get("faithfulness"), len(faith),
            bool(faith) and all(row.get("faithful") is not None for row in faith))

    if meta.get("sample_mode"):
        for item in metrics.values():
            item["value"] = None
            item["status"] = "pending_upstream"
    pending = not metrics or any(item["value"] is None for item in metrics.values())
    status = "pending_upstream" if pending or meta.get("status") != "complete" or meta.get("sample_mode") else "passed"
    if meta.get("status") == "failed":
        status = "failed"
    return {"status": status, "sample_count": int(meta.get("question_count") or 0),
            "metrics": metrics, "denominators": denominators,
            "reason": "upstream metrics unavailable or incomplete" if pending else None}
