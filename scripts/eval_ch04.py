"""Chapter 4 four-strategy retrieval, evidence, and answer evaluation.

Run from the repository root after building the Standalone knowledge collection.
The retrieval/evidence report is written even when generation is unavailable.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import query_understanding, retrieval
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests/data/eval_ch04.jsonl"
REPORT_DIR = ROOT / "data/ch04/reports"
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
K = 10
CALL_TIMEOUT = 45.0


class _CoverageJudge(BaseModel):
    covered_count: int = Field(description="回答明确覆盖的标注事实数量")
    reason: str = Field(default="", description="简短判断依据")


class _FaithJudge(BaseModel):
    faithful: bool = Field(description="回答是否完全由编号证据支持")
    reason: str = Field(default="", description="简短判断依据")


def _norm(value: str) -> str:
    return "".join((value or "").split())


def _first_rank(sample: dict, hits: list[dict]) -> int | None:
    expected = sample.get("expect_section") or []
    for rank, hit in enumerate(hits, 1):
        section = str(hit.get("section_path") or "")
        if expected and any(term in section for term in expected):
            return rank
    return None


def _coverage_mech(points: list[str], hits: list[dict]) -> float | None:
    if not points:
        return None
    evidence = _norm("\n".join(str(hit.get("answer") or "") for hit in hits))
    return sum(_norm(point) in evidence for point in points) / len(points)


def _deterministic_summary(
    samples: list[dict], hits: dict[tuple[str, str], list[dict]],
    strategies: tuple[str, ...] | list[str] = STRATEGIES, k: int = K,
) -> dict:
    retrieval_scores: dict = {}
    evidence_scores: dict = {}
    for strategy in strategies:
        by_bucket = defaultdict(list)
        for sample in samples:
            if not sample.get("should_refuse"):
                by_bucket[sample["bucket"]].append(sample)
        retrieval_scores[strategy] = {}
        evidence_scores[strategy] = {}
        for bucket, rows in by_bucket.items():
            ranks = [_first_rank(row, hits.get((strategy, row["id"]), [])[:k]) for row in rows]
            coverages = [
                _coverage_mech(row["expect_points"], hits.get((strategy, row["id"]), [])[:k])
                for row in rows
            ]
            retrieval_scores[strategy][bucket] = {
                "count": len(rows),
                "recall_at_k": sum(rank is not None for rank in ranks) / len(rows),
                "mrr": sum(1 / rank for rank in ranks if rank is not None) / len(rows),
            }
            evidence_scores[strategy][bucket] = sum(
                score or 0 for score in coverages
            ) / len(rows)
    return {"retrieval": retrieval_scores, "evidence_coverage": evidence_scores}


def _load_samples(path: Path = SAMPLES) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if row["id"] in ids or not row["query"].strip():
            raise ValueError("duplicate ID or blank query in evaluation set")
        ids.add(row["id"])
        if row["bucket"] == "D_absent" and not row["should_refuse"]:
            raise ValueError("D_absent must be a refusal question")
        if row["bucket"] != "D_absent" and (row["should_refuse"] or not row["expect_points"]):
            raise ValueError("answerable question needs facts and must not expect refusal")
    return rows


async def _retrieve_all(samples: list[dict], lines: list[str]) -> dict[tuple[str, str], list[dict]]:
    gate = asyncio.Semaphore(8)
    rewrite_gate = asyncio.Semaphore(3)
    results = {}
    rewritten: dict[str, tuple[str, str]] = {}

    async def rewrite(sample: dict):
        async with rewrite_gate:
            try:
                understood = await asyncio.wait_for(
                    query_understanding.understand(sample["query"]), timeout=CALL_TIMEOUT,
                )
                standard = understood["standard"]
                expanded = understood["expanded"]
                lexical = standard + (" " + " ".join(expanded) if expanded else "")
                rewritten[sample["id"]] = (standard, lexical)
            except Exception as exc:  # noqa: BLE001 - keep retrieval measurable on raw query
                lines.append(f"REWRITE_ERROR {sample['id']}: {type(exc).__name__}")
                rewritten[sample["id"]] = (sample["query"], sample["query"])

    await asyncio.gather(*(rewrite(sample) for sample in samples))

    async def one(strategy: str, sample: dict):
        async with gate:
            try:
                value = await asyncio.wait_for(
                    retrieval.search_knowledge(
                        rewritten[sample["id"]][0], top_k=K, strategy=strategy,
                        bm25_query=rewritten[sample["id"]][1],
                    ),
                    timeout=90,
                )
                results[(strategy, sample["id"])] = value
            except Exception as exc:  # noqa: BLE001 - keep partial report usable
                lines.append(f"RETRIEVAL_ERROR {strategy}/{sample['id']}: {type(exc).__name__}")
                results[(strategy, sample["id"])] = []

    await asyncio.gather(*(one(strategy, sample) for strategy in STRATEGIES for sample in samples))
    return results


def _evidence(hits: list[dict]) -> str:
    return "\n".join(
        f"[{index}] {hit.get('question', '')}: {hit.get('answer', '')}"
        for index, hit in enumerate(hits, 1)
    )


async def _try_call(coroutine, label: str, errors: list[str]):
    try:
        return await asyncio.wait_for(coroutine, timeout=CALL_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - a single upstream call must not abort the report
        errors.append(f"{label}: {type(exc).__name__}")
        return None


async def _generation(samples: list[dict], hits: dict, lines: list[str]) -> dict | None:
    settings = get_settings()
    model = get_chat_model()
    answer_chain = RAG_ANSWER_PROMPT | model
    judge_model = model.with_structured_output(
        _CoverageJudge, method=settings.structured_output_method,
    )
    coverage_prompt = (
        "你是答案覆盖度评审员。只数回答中由证据支持且明确覆盖的标注要点。"
        "不得把相关但未回答的要点算入。只输出 JSON：covered_count 整数、reason 字符串。\n"
        "用户问题：{query}\n标注要点：{points}\n证据：{evidence}\n回答：{answer}"
    )
    faith_chain = FAITHFULNESS_PROMPT | model.with_structured_output(
        _FaithJudge, method=settings.structured_output_method,
    )
    gate = asyncio.Semaphore(3)
    errors: list[str] = []
    records: list[dict] = []

    async def one(strategy: str, sample: dict):
        async with gate:
            found = hits.get((strategy, sample["id"]), [])
            evidence = _evidence(found)
            record = {"id": sample["id"], "bucket": sample["bucket"], "strategy": strategy}
            if not found:
                record.update(answer="暂时没有查到相关信息", covered=0, faithful=True)
                records.append(record)
                return
            response = await _try_call(
                answer_chain.ainvoke({"query": sample["query"], "evidence": evidence}),
                f"answer {strategy}/{sample['id']}", errors,
            )
            if response is None:
                record["error"] = "answer unavailable"
                records.append(record)
                return
            answer = str(response.content)
            record["answer"] = answer
            if not sample["should_refuse"]:
                coverage = await _try_call(
                    judge_model.ainvoke(coverage_prompt.format(
                        query=sample["query"], points=json.dumps(sample["expect_points"], ensure_ascii=False),
                        evidence=evidence, answer=answer,
                    )),
                    f"coverage {strategy}/{sample['id']}", errors,
                )
                record["covered"] = (
                    max(0, min(len(sample["expect_points"]), coverage.covered_count))
                    if coverage is not None else None
                )
                if strategy == "hybrid_rerank":
                    faith = await _try_call(
                        faith_chain.ainvoke({"evidence": evidence, "answer": answer}),
                        f"faith {sample['id']}", errors,
                    )
                    record["faithful"] = faith.faithful if faith is not None else None
                    record["faith_reason"] = faith.reason if faith is not None else ""
            else:
                record["refused"] = "暂时没有查到" in answer or "无法" in answer or "不能" in answer
            records.append(record)

    await asyncio.gather(*(one(strategy, sample) for strategy in STRATEGIES for sample in samples))
    records.sort(key=lambda record: (record["strategy"], record["id"]))
    lines.extend(f"GENERATION_ERROR {error}" for error in errors)
    if not records:
        return None

    answer_coverage = {}
    for strategy in STRATEGIES:
        answer_coverage[strategy] = {}
        for bucket in sorted({sample["bucket"] for sample in samples if not sample["should_refuse"]}):
            items = [
                record for record in records if record["strategy"] == strategy
                and record["bucket"] == bucket and record.get("covered") is not None
            ]
            point_counts = {sample["id"]: len(sample["expect_points"]) for sample in samples}
            answer_coverage[strategy][bucket] = (
                sum(record["covered"] / point_counts[record["id"]] for record in items) / len(items)
                if items else None
            )
    faith = [record["faithful"] for record in records if record["strategy"] == "hybrid_rerank"
             and not record.get("bucket") == "D_absent" and record.get("faithful") is not None]
    refusals = [record["refused"] for record in records if record["strategy"] == "hybrid_rerank"
                and record["bucket"] == "D_absent" and "refused" in record]
    return {
        "answer_coverage": answer_coverage,
        "faithfulness": sum(faith) / len(faith) if faith else None,
        "refusal_rate": sum(refusals) / len(refusals) if refusals else None,
        "records": records,
        "errors": errors,
    }


def _write_report(report: dict, lines: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rag_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (REPORT_DIR / "rag_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main(skip_generation: bool = False, samples_path: Path = SAMPLES) -> dict:
    samples = _load_samples(samples_path)
    lines = [f"RAG evaluation: {len(samples)} questions, Top-{K}"]
    hits = await _retrieve_all(samples, lines)
    summary = _deterministic_summary(samples, hits)
    lines.append("RETRIEVAL " + json.dumps(summary["retrieval"], ensure_ascii=False))
    lines.append("EVIDENCE " + json.dumps(summary["evidence_coverage"], ensure_ascii=False))
    generation = None
    if not skip_generation:
        try:
            generation = await _generation(samples, hits, lines)
        except Exception as exc:  # noqa: BLE001 - deterministic results still publish
            lines.append(f"GENERATION_UNAVAILABLE {type(exc).__name__}")
    lines.append("GENERATION " + json.dumps(
        {key: value for key, value in (generation or {}).items() if key not in {"records", "errors"}},
        ensure_ascii=False,
    ))
    settings = get_settings()
    report = {
        "meta": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(), "question_count": len(samples),
            "top_k": K, "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model, "judge_model": settings.chat_model,
        },
        **summary, "generation": generation,
    }
    _write_report(report, lines)
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generation", action="store_true")
    options = parser.parse_args()
    asyncio.run(main(skip_generation=options.skip_generation))
