"""Validate marked Ch05 intent examples; live evaluation requires explicit --live."""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.intent import ModelIntentClassifier, safe_classify
from app.graph.routing import INTENT_ROUTES

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "intent_ch05.jsonl"


def load_samples(path: Path = DATA) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = set(INTENT_ROUTES) - {"unknown"}
    seen = set()
    for row in rows:
        if set(row) != {"query", "intent"} or not isinstance(row["query"], str):
            raise ValueError("invalid intent sample fields")
        query = row["query"].strip()
        if not query or query in seen or row["intent"] not in labels:
            raise ValueError("blank, duplicate or invalid intent sample")
        seen.add(query)
    counts = Counter(row["intent"] for row in rows)
    if set(counts) != labels or min(counts.values()) < 2:
        raise ValueError("all seven intents need at least two marked samples")
    return rows


async def _live(rows: list[dict[str, str]]) -> dict:
    from app.config import get_settings
    from app.core.llm import get_chat_model

    settings = get_settings()
    model = get_chat_model(settings=settings)
    classifier = ModelIntentClassifier(model, settings.structured_output_method)
    counts = Counter()
    confusion = Counter()
    for row in rows:
        predicted = await safe_classify(classifier, row["query"])
        counts["total"] += 1
        counts["correct"] += predicted == row["intent"]
        confusion[(row["intent"], predicted)] += 1
    return {"status": "completed", "total": counts["total"],
            "correct": counts["correct"], "confusion": [
                {"expected": expected, "predicted": predicted, "count": count}
                for (expected, predicted), count in sorted(confusion.items())
            ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call configured upstream for every sample")
    args = parser.parse_args()
    rows = load_samples()
    result = asyncio.run(_live(rows)) if args.live else {
        "status": "pending_upstream", "validated_samples": len(rows),
        "counts": dict(sorted(Counter(row["intent"] for row in rows).items())),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
