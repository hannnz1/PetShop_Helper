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
    queries = set()
    for row in rows:
        assert row["id"] not in ids and row["query"].strip(), row
        assert row["query"] not in queries, f"duplicate query: {row['id']}"
        ids.add(row["id"])
        queries.add(row["query"])
        bucket = row["bucket"]
        counts[bucket] += 1
        if bucket == "D_absent":
            assert row["should_refuse"] and not row["expect_section"] and not row["expect_points"], row
            continue
        assert not row["should_refuse"] and row["expect_section"] and row["expect_points"], row
        groups = row.get("expect_sections_all") or [row["expect_section"]]
        candidates = []
        for group in groups:
            aliases = group if isinstance(group, list) else [group]
            matched = [
                chunk for chunk in chunks
                if any(section in chunk.section_path for section in aliases)
            ]
            assert matched, f"{row['id']}: no annotated source section for {aliases}"
            candidates.extend(matched)
        evidence = _norm("\n".join(chunk.answer for chunk in candidates))
        for point in row["expect_points"]:
            assert _norm(point) in evidence, f"{row['id']}: unsupported annotated point {point}"
    assert counts == {"A_policy": 60, "B_model": 60, "C_colloquial": 60,
                      "D_absent": 60, "E_multi": 60}, counts
    return dict(counts)


if __name__ == "__main__":
    print(validate())
