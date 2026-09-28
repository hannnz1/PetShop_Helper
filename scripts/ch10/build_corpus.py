"""Build reviewable private topic corpus; never silently externalize pool text."""

import argparse
import asyncio
import hashlib
import ipaddress
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from sqlalchemy import select

import app.db.base as db
from app.config import get_settings
from app.core.taxonomy import LABEL2ID, TOPIC_NAMES, terminology_table
from app.db.models import LowConfidenceQuestion
from scripts.ch10.corpus_lib import dedupe, desensitize
from scripts.ch10.prelabel import LabelResult
from scripts.ch10.validate_golden import GoldenReport, load_golden, validate_golden


class _GeneratedItem(BaseModel):
    text: str
    labels: list[str]


class _GeneratedBatch(BaseModel):
    items: list[_GeneratedItem]


SIMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "为猫用品电商生成 {count} 条口语化、相互不同的用户问题，每条须命中“{target}”。"
     "约 15% 方言、10% 错字、15% 多诉求。只标字面诉求，不能脑补。"
     "所有标签必须取下面术语表类目名原文：\n{taxonomy}"),
    ("human", "请生成 {count} 条“{target}”问句。"),
])


@dataclass(frozen=True)
class CorpusReport:
    status: str
    labeled: tuple[dict, ...] = ()
    failures: tuple[int, ...] = ()
    missing: dict[str, int] | None = None
    failed_rows: tuple[dict, ...] = ()
    raw_rows: tuple[dict, ...] = ()
    clean_rows: tuple[dict, ...] = ()


def _local_url(value: str) -> bool:
    hostname = urlparse(value).hostname
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        return False


async def load_pool(limit: int = 1000) -> list[dict]:
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")
    async with db.async_session() as session:
        rows = (await session.scalars(select(LowConfidenceQuestion).order_by(
            LowConfidenceQuestion.id,
        ).limit(limit))).all()
    return [{"id": row.id, "text": desensitize(row.raw_question), "origin": "pool"} for row in rows]


def missing_by_class(rows: list[dict], *, target: int = 100) -> dict[str, int]:
    if target < 0:
        raise ValueError("target must be nonnegative")
    counts = {name: 0 for name in TOPIC_NAMES}
    for row in rows:
        for label in set(row.get("labels", [])):
            if label not in LABEL2ID:
                raise ValueError("unknown topic label")
            counts[label] += 1
    return {name: max(0, target - count) for name, count in counts.items()}


def audit_sample(rows: list[dict], *, per_class: int = 5, seed: int = 42) -> list[dict]:
    if per_class < 0:
        raise ValueError("per_class must be nonnegative")
    rng = random.Random(seed)
    chosen = list(row for row in rows if row.get("origin") == "pool")
    seen = {row["text"] for row in chosen}
    for label in TOPIC_NAMES:
        candidates = [row for row in rows if row.get("origin") == "simulated" and label in row["labels"]]
        for row in rng.sample(candidates, min(per_class, len(candidates))):
            if row["text"] not in seen:
                chosen.append(row)
                seen.add(row["text"])
    return chosen


async def build_corpus(
    raw_rows: list[dict], *, prelabel, golden_report: GoldenReport | None = None,
    simulate=None, allow_external_real_text: bool = False, target_per_class: int = 100, clean_model=None,
) -> CorpusReport:
    """Prepare a batch only after privacy and reviewed-golden gates."""
    if not allow_external_real_text and not _local_url(get_settings().chat_base_url):
        return CorpusReport("pending_upstream")
    if golden_report is None or not golden_report.passed:
        return CorpusReport("pending_review")
    from scripts.ch10.text_jobs import prepare_rows, clean_rows
    prepared = prepare_rows(raw_rows)
    masked = (await clean_rows(prepared, clean_model, allow_external_real_text=allow_external_real_text)
              if clean_model is not None else prepared)
    labeled: list[dict] = []
    failures: list[int] = []
    failed_rows: list[dict] = []
    for row in masked:
        result = await prelabel(row["text"])
        if not isinstance(result, LabelResult) or result.status != "labeled":
            failures.append(row["id"])
            failed_rows.append({**row, "prelabel_status":
                                result.status if isinstance(result, LabelResult) else "invalid_result"})
            continue
        labeled.append({**row, "labels": list(result.labels)})
    missing = missing_by_class(labeled, target=target_per_class)
    if simulate is not None:
        for label, need in missing.items():
            if need <= 0:
                continue
            try:
                generated = await simulate(label, need)
            except Exception:
                continue
            for item in generated:
                labels = tuple(item.get("labels", ()))
                if label not in labels or any(name not in LABEL2ID for name in labels):
                    continue
                labeled.append({"text": desensitize(item["text"]), "labels": list(labels),
                                "origin": "simulated"})
        labeled = dedupe(labeled)
        missing = missing_by_class(labeled, target=target_per_class)
    status = "partial" if failures else ("pending_data" if any(missing.values()) else "ready")
    return CorpusReport(status, tuple(labeled), tuple(failures), missing, tuple(failed_rows),
                        tuple(prepared), tuple(masked))


def write_corpus(report: CorpusReport, directory: Path) -> None:
    """Write review artifacts only; no raw customer text or credentials."""
    if report.status not in {"partial", "ready", "pending_data"}:
        raise ValueError("corpus is not ready for review export")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in (*report.labeled, *report.failed_rows):
        if row.get('review_id'):
            review_id = row['review_id']
        elif row.get("id") is not None:
            review_id = f"pool-{row['id']}"
        else:
            digest = hashlib.sha256((row["origin"] + "\0" + row["text"]).encode("utf-8")).hexdigest()[:16]
            review_id = f"{row['origin']}-{digest}"
        rows.append({**row, "review_id": review_id})
    if len({row["review_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate review ID")
    (directory / "corpus_labeled.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8",
    )
    for name, stage in (('raw', report.raw_rows), ('clean', report.clean_rows)):
        (directory / f'corpus_{name}.jsonl').write_text(
            '\n'.join(json.dumps(row, ensure_ascii=False) for row in stage), encoding='utf-8',
        )
    audit = audit_sample(rows)
    (directory / 'corpus_report.json').write_text(json.dumps({
        'status': report.status, 'counts': {'raw': len(report.raw_rows), 'clean': len(report.clean_rows),
            'labeled': len(report.labeled), 'failed': len(report.failed_rows)}, 'missing_by_class': report.missing,
        'metadata': {'taxonomy_hash': hashlib.sha256(terminology_table().encode()).hexdigest(),
            'clean_hash': hashlib.sha256((directory/'corpus_clean.jsonl').read_bytes()).hexdigest()},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / "audit.md").write_text(
        "\n".join(f"- {row['review_id']} [{','.join(row.get('labels', [])) or '待人工标注'}] {row['text']}"
                  for row in audit),
        encoding="utf-8",
    )


def run_synthetic(directory: Path) -> CorpusReport:
    """Offline fixture export; source examples are already human annotated."""
    rows = [{**row, "origin": "synthetic_reviewed"} for row in load_golden()]
    report = CorpusReport("pending_data", tuple(rows), missing=missing_by_class(rows))
    write_corpus(report, directory)
    return report


async def simulate_class(model, target: str, count: int) -> list[dict]:
    structured = model.with_structured_output(
        _GeneratedBatch, method=get_settings().structured_output_method,
    )
    out: list[dict] = []
    for _ in range((count + 19) // 20):
        need = min(count - len(out), 20)
        if need <= 0:
            break
        prompt = SIMULATE_PROMPT.invoke({"target": target, "count": need,
                                         "taxonomy": terminology_table()})
        result: _GeneratedBatch = await structured.ainvoke(prompt)
        out.extend({"text": item.text, "labels": item.labels} for item in result.items[:need])
    return out[:count]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--allow-external-real-text", action="store_true")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/ch10/corpus"))
    args = parser.parse_args()
    if args.synthetic_only:
        report = run_synthetic(args.out)
        print(f"status={report.status} synthetic_rows={len(report.labeled)}")
        return
    rows = await load_pool(args.limit)
    if not rows:
        print("status=pending_data pool_rows=0")
        return
    if not args.allow_external_real_text and not _local_url(get_settings().chat_base_url):
        print(f"status=pending_upstream pool_rows={len(rows)}")
        return
    from app.core.llm import get_chat_model
    from scripts.ch10.prelabel import prelabel_one, prelabel_fingerprint

    model = get_chat_model(temperature=0)
    golden = load_golden()
    predictions = [(await prelabel_one(row["text"], model)).labels for row in golden]
    golden_report = validate_golden(golden, predictions)
    from scripts.ch10.validate_golden import GOLDEN
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'golden_report.json').write_text(json.dumps({
        **asdict(golden_report), 'status': 'passed' if golden_report.passed else 'failed',
        'metadata': {'golden_hash': hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
                     'taxonomy_hash': hashlib.sha256(terminology_table().encode()).hexdigest(),
                     'prelabel_model': get_settings().chat_model},
        'prelabel_version': prelabel_fingerprint(),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if not golden_report.passed:
        print(f"status=pending_review golden_rate={golden_report.rate:.3f}")
        return
    report = await build_corpus(
        rows, prelabel=lambda value: prelabel_one(value, model),
        golden_report=golden_report,
        simulate=lambda label, need: simulate_class(model, label, need),
        allow_external_real_text=args.allow_external_real_text,
        clean_model=model,
    )
    write_corpus(report, args.out)
    print(f"status={report.status} labeled={len(report.labeled)} failures={len(report.failures)}")


if __name__ == "__main__":
    asyncio.run(main())
