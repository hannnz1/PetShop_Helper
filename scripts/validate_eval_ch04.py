"""Check annotated Chapter 4 questions against reviewed local source chunks."""

import json
from collections import Counter
from pathlib import Path

from app.kb import sources
from scripts.build_kb import source_chunks


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests/data/eval_ch04.jsonl"


def _norm(value: str) -> str:
    return "".join(value.split())


def validate() -> dict[str, int]:
    rows = [json.loads(line) for line in SAMPLES.read_text(encoding="utf-8").splitlines() if line]
    chunks = [
        chunk
        for filename, kind in sources.SOURCE_TYPES.items()
        for chunk in source_chunks(filename, (sources.KB_DIR / filename).read_text(encoding="utf-8"), kind)
    ]
    counts = Counter()
    ids = set()
    for row in rows:
        assert row["id"] not in ids and row["query"].strip(), row
        ids.add(row["id"])
        bucket = row["bucket"]
        counts[bucket] += 1
        if bucket == "D_absent":
            assert row["should_refuse"] and not row["expect_section"] and not row["expect_points"], row
            continue
        assert not row["should_refuse"] and row["expect_section"] and row["expect_points"], row
        candidates = [
            chunk for chunk in chunks
            if any(section in chunk.section_path for section in row["expect_section"])
        ]
        assert candidates, f"{row['id']}: no annotated source section"
        evidence = _norm("\n".join(chunk.answer for chunk in candidates))
        for point in row["expect_points"]:
            assert _norm(point) in evidence, f"{row['id']}: unsupported annotated point {point}"
    assert counts == {"A_policy": 20, "B_model": 20, "C_colloquial": 20, "D_absent": 20}, counts
    return dict(counts)


if __name__ == "__main__":
    print(validate())
