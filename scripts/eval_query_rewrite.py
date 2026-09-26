"""Small labeled evaluation for Chapter 4 query rewriting."""

import asyncio
import json
from pathlib import Path

from app.core.query_understanding import understand


SAMPLES = Path(__file__).resolve().parents[1] / "tests" / "data" / "query_rewrite_samples.jsonl"


async def main() -> None:
    passed = 0
    samples = [json.loads(line) for line in SAMPLES.read_text(encoding="utf-8").splitlines() if line]
    for sample in samples:
        result = await understand(sample["query"])
        blob = result["standard"] + " " + " ".join(result["expanded"])
        hit = any(word in blob for word in sample["expect_any"])
        hit = hit and not any(word in blob for word in sample.get("forbid_any", []))
        passed += hit
        print(f"{'OK' if hit else 'MISS'} {sample['query']} -> {result}")
    print(f"Query rewrite labeled evaluation: {passed}/{len(samples)}")
    if passed < 4 or len(samples) != 5:
        raise SystemExit("Query rewrite did not meet the 4/5 acceptance gate")


if __name__ == "__main__":
    asyncio.run(main())
