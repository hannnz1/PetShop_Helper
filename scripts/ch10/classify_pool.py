"""Aggregate predictions for masked low-confidence questions without raw text export."""

import argparse
import asyncio
import json
from pathlib import Path

from app.core.taxonomy import TOPIC_NAMES
from app.topics.batch import classify_pool
from sqlalchemy import update
import app.db.base as db
from app.db.models import TopicClassificationRun


def categorize_pool(rows: list[dict], predictions: list[dict]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("pool and prediction length mismatch")
    seen = set()
    counts = {name: 0 for name in TOPIC_NAMES}
    for row, result in zip(rows, predictions, strict=True):
        row_id = row["id"]
        if row_id in seen:
            raise ValueError("duplicate pool ID")
        seen.add(row_id)
        for label in set(result["labels"]):
            if label not in counts:
                raise ValueError("unknown predicted label")
            counts[label] += 1
    return {"question_count": len(seen), "class_counts": counts}


async def run(limit: int, output: Path, *, min_batch=10, force=False, reclassify=False) -> dict:
    report = await classify_pool(limit=limit, min_batch=min_batch, force=force, reclassify=reclassify)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.parent / 'classification-runs' / f"{report['run_id']}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Store a portable report reference; paths do not contain raw questions.
    async with db.async_session.begin() as session:
        await session.execute(update(TopicClassificationRun).where(
            TopicClassificationRun.run_id == report['run_id']).values(report_path=f'classification-runs/{archive.name}'))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--min-batch', type=int, default=10)
    parser.add_argument('--reclassify', action='store_true')
    parser.add_argument("--out", type=Path, default=Path("data/ch10/pool_categories.json"))
    args = parser.parse_args()
    report = asyncio.run(run(args.limit, args.out, min_batch=args.min_batch, force=args.force, reclassify=args.reclassify))
    print(f"status={report['status']} question_count={report['question_count']} written={report['written']}")
    if report['status'] in ('failed', 'partial'):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
