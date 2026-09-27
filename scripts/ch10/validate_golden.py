"""Exact-set prelabel gate for reviewed synthetic examples."""

import json
from dataclasses import dataclass
from pathlib import Path


GOLDEN = Path(__file__).with_name("golden_samples.jsonl")


@dataclass(frozen=True)
class GoldenReport:
    total: int
    hits: int
    rate: float
    passed: bool


def validate_golden(samples: list[dict], predictions: list[tuple[str, ...]]) -> GoldenReport:
    if not samples or len(samples) != len(predictions):
        raise ValueError("golden examples and predictions must be nonempty and aligned")
    hits = sum(set(row["labels"]) == set(prediction)
               for row, prediction in zip(samples, predictions, strict=True))
    rate = hits / len(samples)
    return GoldenReport(len(samples), hits, rate, rate >= 0.8)


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
