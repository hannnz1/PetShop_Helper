"""Chapter 4 four-strategy retrieval, evidence, and answer evaluation.

Run from the repository root after building the Standalone knowledge collection.
The retrieval/evidence report is written even when generation is unavailable.
"""

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import query_understanding, retrieval
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.db import repository


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests/data/eval_ch04.jsonl"
REPORT_DIR = ROOT / "data/ch04/reports"
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
K = 10
RECALL_K = 5
CALL_TIMEOUT = 45.0
CACHE_VERSION = 1
CACHE_DIR = ROOT / "work/ch04-eval-cache"


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


def _evidence_ranks(sample: dict, hits: list[dict]) -> list[int | None]:
    """One first-hit rank per required evidence group; aliases share a group."""

    groups = sample.get("expect_sections_all") or [sample.get("expect_section") or []]
    return [
        _first_rank({"expect_section": group if isinstance(group, list) else [group]}, hits)
        for group in groups
    ]


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
            ranks = [_evidence_ranks(row, hits.get((strategy, row["id"]), [])[:k]) for row in rows]
            coverages = [
                _coverage_mech(row["expect_points"], hits.get((strategy, row["id"]), [])[:k])
                for row in rows
            ]
            retrieval_scores[strategy][bucket] = {
                "count": len(rows),
                "recall_at_5": sum(
                    sum(rank is not None and rank <= RECALL_K for rank in row_ranks) / len(row_ranks)
                    for row_ranks in ranks
                ) / len(rows),
                "mrr": sum(
                    sum(1 / rank for rank in row_ranks if rank is not None) / len(row_ranks)
                    for row_ranks in ranks
                ) / len(rows),
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


def _read_cache(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    entries = {}
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue  # A partial final line must not hide earlier checkpoints.
            if isinstance(row, dict) and isinstance(row.get("key"), str):
                entries[row["key"]] = row.get("value")
    return entries


def _append_cache(path: Path | None, key: str, value: object) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        # A crash may leave an incomplete last row. Start a fresh line before
        # appending so the next valid result remains readable.
        if path.stat().st_size:
            with path.open("rb") as existing:
                existing.seek(-1, 2)
                if existing.read(1) != b"\n":
                    stream.write("\n")
        stream.write(json.dumps({"key": key, "value": value}, ensure_ascii=False, default=str) + "\n")


def _cache_key(kind: str, sample: dict) -> str:
    return json.dumps([kind, sample["id"], sample["query"]], ensure_ascii=False)


async def _cache_path() -> Path:
    """Change the cache file when KB authority, model config or pipeline code changes."""
    settings = get_settings()
    chunks = await repository.list_all_chunks()
    authority = [
        [row.id, row.category, row.questions, row.answer, row.section_path,
         row.content_type, row.vectorize_status]
        for row in chunks
    ]
    code = [
        (ROOT / name).read_bytes().hex()
        for name in ("scripts/eval_ch04.py", "app/core/retrieval.py",
                     "app/core/query_understanding.py", "app/core/rerank.py",
                     "app/core/embeddings.py")
    ]
    scope = [CACHE_VERSION, authority, code, settings.milvus_uri,
             settings.chat_base_url, settings.chat_model, settings.structured_output_method,
             settings.embed_base_url, settings.embed_model,
             settings.rerank_base_url, settings.rerank_model,
             settings.recall_top_k, K]
    digest = hashlib.sha256(json.dumps(scope, ensure_ascii=False, default=str).encode()).hexdigest()[:20]
    target = CACHE_DIR / f"retrieval-{digest}.jsonl"
    from scripts.ch04_cache_migration import migrate_legacy
    migrate_legacy(ROOT, target, scope)
    return target


def _generation_cache_path(retrieval_cache: Path) -> Path:
    """Generation also depends on prompt wording and the model adapter."""
    scope = [retrieval_cache.name]
    for name in ("app/core/prompts.py", "app/core/llm.py", "app/core/model_guard.py"):
        scope.append(hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(scope).encode()).hexdigest()[:20]
    return retrieval_cache.with_name(f"generation-{digest}.jsonl")


async def _retrieve_all(
    samples: list[dict], lines: list[str], cache_path: Path | None = None,
) -> dict[tuple[str, str], list[dict]]:
    gate = asyncio.Semaphore(8)
    rewrite_gate = asyncio.Semaphore(3)
    results = {}
    rewritten: dict[str, tuple[str, str]] = {}
    cached = _read_cache(cache_path)
    progress = {"rewrite": 0, "retrieval": 0}

    def tick(stage: str, total: int) -> None:
        progress[stage] += 1
        if progress[stage] % 50 == 0 or progress[stage] == total:
            print(f"{stage}: {progress[stage]}/{total}", flush=True)

    async def rewrite(sample: dict):
        key = _cache_key("rewrite", sample)
        if isinstance(cached.get(key), list) and len(cached[key]) == 2:
            rewritten[sample["id"]] = tuple(cached[key])
            tick("rewrite", len(samples))
            return
        async with rewrite_gate:
            try:
                understood = await asyncio.wait_for(
                    query_understanding.understand(sample["query"]), timeout=CALL_TIMEOUT,
                )
                standard = understood["standard"]
                expanded = understood["expanded"]
                lexical = standard + (" " + " ".join(expanded) if expanded else "")
                rewritten[sample["id"]] = (standard, lexical)
                _append_cache(cache_path, key, [standard, lexical])
            except Exception as exc:  # noqa: BLE001 - keep retrieval measurable on raw query
                lines.append(f"REWRITE_ERROR {sample['id']}: {type(exc).__name__}")
                rewritten[sample["id"]] = (sample["query"], sample["query"])
            finally:
                tick("rewrite", len(samples))

    await asyncio.gather(*(rewrite(sample) for sample in samples))

    async def one(strategy: str, sample: dict):
        key = _cache_key(strategy, sample)
        if isinstance(cached.get(key), list):
            results[(strategy, sample["id"])] = cached[key]
            tick("retrieval", len(STRATEGIES) * len(samples))
            return
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
                _append_cache(cache_path, key, value)
            except Exception as exc:  # noqa: BLE001 - keep partial report usable
                lines.append(f"RETRIEVAL_ERROR {strategy}/{sample['id']}: {type(exc).__name__}")
                results[(strategy, sample["id"])] = []
            finally:
                tick("retrieval", len(STRATEGIES) * len(samples))

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


async def _save_unfaithful_case(
    sample: dict, answer: str, reason: str, hits: list[dict],
    judge_model: str, errors: list[str],
) -> bool:
    """Store the exact ranked evidence sent to the judge; failure is advisory."""
    citations = [
        {"n": index, "chunk_id": hit.get("id"),
         "section_path": hit.get("section_path", ""),
         "question": hit.get("question", ""), "answer": hit.get("answer", "")}
        for index, hit in enumerate(hits, 1)
    ]
    try:
        await repository.upsert_faith_case(
            sample["id"], bucket=sample["bucket"], query=sample["query"],
            answer=answer, reason=reason, citations=citations,
            judge_model=judge_model,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - ledger must not erase evaluation results
        errors.append(f"ledger {sample['id']}: {type(exc).__name__}")
        return False


async def _generation(
    samples: list[dict], hits: dict, lines: list[str], cache_path: Path | None = None,
    concurrency: int = 3,
) -> dict | None:
    from app.core.model_guard import GUARD_VERSION, repair_once
    if not 1 <= concurrency <= 12:
        raise ValueError("generation concurrency must be between 1 and 12")
    settings = get_settings()
    model = get_chat_model()
    answer_chain = RAG_ANSWER_PROMPT | model
    judge_model = get_chat_model(temperature=0).with_structured_output(
        _CoverageJudge, method=settings.structured_output_method,
    )
    coverage_prompt = (
        "你是答案覆盖度评审员。只数回答中由证据支持且明确覆盖的标注要点。"
        "不得把相关但未回答的要点算入。只输出 JSON：covered_count 整数、reason 字符串。\n"
        "用户问题：{query}\n标注要点：{points}\n证据：{evidence}\n回答：{answer}"
    )
    faith_chain = FAITHFULNESS_PROMPT | get_chat_model(temperature=0).with_structured_output(
        _FaithJudge, method=settings.structured_output_method,
    )
    gate = asyncio.Semaphore(concurrency)
    errors: list[str] = []
    records: list[dict] = []
    cached = _read_cache(cache_path)
    completed = 0

    def checkpoint(key: str, record: dict) -> None:
        try:
            _append_cache(cache_path, key, record)
        except OSError as exc:
            lines.append(f"GENERATION_CACHE_ERROR {record['strategy']}/{record['id']}: {type(exc).__name__}")

    def tick() -> None:
        nonlocal completed
        completed += 1
        total = len(STRATEGIES) * len(samples)
        if completed % 50 == 0 or completed == total:
            print(f"generation: {completed}/{total}", flush=True)

    async def one(strategy: str, sample: dict):
        found = hits.get((strategy, sample["id"]), [])
        cache_key = "generation:" + hashlib.sha256(
            json.dumps([GUARD_VERSION, strategy, sample, found], ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()
        saved = cached.get(cache_key)
        if isinstance(saved, dict) and saved.get("strategy") == strategy and saved.get("id") == sample["id"]:
            records.append(saved)
            tick()
            return
        async with gate:
            evidence = _evidence(found)
            record = {"id": sample["id"], "bucket": sample["bucket"], "strategy": strategy}
            if not found:
                record.update(answer="暂时没有查到相关信息", original_answer="暂时没有查到相关信息",
                              unsupported_models=[], repair_status="not_needed", guard_passed=True,
                              guard_version=GUARD_VERSION, covered=0, faithful=True)
                records.append(record)
                checkpoint(cache_key, record)
                tick()
                return
            response = await _try_call(
                answer_chain.ainvoke({"query": sample["query"], "evidence": evidence}),
                f"answer {strategy}/{sample['id']}", errors,
            )
            if response is None:
                record["error"] = "answer unavailable"
                records.append(record)
                tick()
                return
            async def repair(hint):
                repaired = await _try_call(
                    answer_chain.ainvoke({'query': sample['query'] + '\n\n' + hint, 'evidence': evidence}),
                    f"model repair {strategy}/{sample['id']}", errors,
                )
                return str(repaired.content) if repaired is not None else None

            record.update(await repair_once(str(response.content), evidence, repair))
            answer = record['answer']
            ledger_saved = True
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
                    record["faithful"] = (faith.faithful if faith is not None else None) if record['guard_passed'] else False
                    record["faith_reason"] = faith.reason if faith is not None else ""
                    if not record['guard_passed']:
                        record['faith_reason'] = '型号机械校验失败；' + record['faith_reason']
                    if record['faithful'] is False:
                        ledger_saved = await _save_unfaithful_case(
                            sample, answer, record['faith_reason'], found,
                            settings.chat_model, errors,
                        )
            else:
                record["refused"] = "暂时没有查到" in answer or "无法" in answer or "不能" in answer
            records.append(record)
            if (sample["should_refuse"] or record.get("covered") is not None) and ledger_saved:
                if strategy != "hybrid_rerank" or sample["should_refuse"] or record.get("faithful") is not None:
                    checkpoint(cache_key, record)
            tick()

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
        "faithfulness_cases": [
            {"id": record["id"], "bucket": record["bucket"],
             "answer": record["answer"], "reason": record.get("faith_reason", "")}
            for record in records
            if record["strategy"] == "hybrid_rerank" and record.get("faithful") is False
        ],
        "records": records,
        "errors": errors,
    }


def _write_report(report: dict, lines: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rag_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (REPORT_DIR / "rag_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_exit_code(report: dict) -> int:
    return 2 if report["meta"]["status"] == "partial" else 0


async def main(skip_generation: bool = False, samples_path: Path = SAMPLES,
               use_cache: bool = False, generation_concurrency: int = 3) -> dict:
    samples = _load_samples(samples_path)
    lines = [f"RAG evaluation: {len(samples)} questions, Top-{K}"]
    cache_path = await _cache_path() if use_cache else None
    if cache_path is not None:
        print(f"retrieval cache: {cache_path}", flush=True)
    hits = await _retrieve_all(samples, lines, cache_path=cache_path)
    summary = _deterministic_summary(samples, hits)
    lines.append("RETRIEVAL " + json.dumps(summary["retrieval"], ensure_ascii=False))
    lines.append("EVIDENCE " + json.dumps(summary["evidence_coverage"], ensure_ascii=False))
    generation = None
    if not skip_generation:
        try:
            generation_cache = _generation_cache_path(cache_path) if cache_path is not None else None
            if generation_cache is not None:
                print(f"generation cache: {generation_cache}", flush=True)
            generation = await _generation(
                samples, hits, lines, cache_path=generation_cache,
                concurrency=generation_concurrency,
            )
        except Exception as exc:  # noqa: BLE001 - deterministic results still publish
            lines.append(f"GENERATION_UNAVAILABLE {type(exc).__name__}")
    lines.append("GENERATION " + json.dumps(
        {key: value for key, value in (generation or {}).items() if key not in {"records", "errors"}},
        ensure_ascii=False,
    ))
    pipeline_errors = sum(
        line.startswith(("REWRITE_ERROR ", "RETRIEVAL_ERROR ")) for line in lines
    )
    if skip_generation:
        status = "partial" if pipeline_errors else "retrieval_only"
    else:
        records = (generation or {}).get("records") or []
        complete = (
            generation is not None
            and not pipeline_errors
            and not generation.get("errors")
            and len(records) == len(samples) * len(STRATEGIES)
            and not any(record.get("error") for record in records)
            and (not any(sample["should_refuse"] for sample in samples)
                 or generation.get("refusal_rate") is not None)
            and (not any(not sample["should_refuse"] for sample in samples)
                 or generation.get("faithfulness") is not None)
        )
        status = "complete" if complete else "partial"
    lines.append(f"STATUS {status}")
    settings = get_settings()
    report = {
        "meta": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(), "question_count": len(samples),
            "top_k": K, "recall_k": RECALL_K, "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model, "judge_model": settings.chat_model,
            "status": status, "retrieval_error_count": pipeline_errors,
        },
        **summary, "generation": generation,
    }
    _write_report(report, lines)
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore the local retrieval cache")
    parser.add_argument("--generation-concurrency", type=int, default=3,
                        help="parallel generation jobs (1-12; default 3)")
    options = parser.parse_args()
    report = asyncio.run(main(
        skip_generation=options.skip_generation,
        use_cache=not options.fresh,
        generation_concurrency=options.generation_concurrency,
    ))
    raise SystemExit(_report_exit_code(report))
