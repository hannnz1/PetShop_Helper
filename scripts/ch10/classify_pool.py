"""Aggregate predictions for masked low-confidence questions without raw text export."""

import argparse
import asyncio
import json
from pathlib import Path

from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.build_corpus import load_pool
from scripts.ch10.inference_lib import load_runtime


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


async def run(model_dir: Path, limit: int, output: Path) -> dict:
    rows = await load_pool(limit)
    runtime = load_runtime(model_dir)
    predictions = runtime.classify([row["text"] for row in rows]) if rows else []
    report = categorize_pool(rows, predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/ch10/pool_categories.json"))
    args = parser.parse_args()
    report = asyncio.run(run(args.model, args.limit, args.out))
    print(f"question_count={report['question_count']}")


if __name__ == "__main__":
    main()
