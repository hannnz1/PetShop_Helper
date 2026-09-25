"""Annotated extraction eval. Requires a configured chat API.

Run from the repository root: ``python -m scripts.eval_mining``.
"""

import asyncio
import json
from pathlib import Path

from scripts.eval_retrieval import check_answer_rules


SAMPLES = Path(__file__).resolve().parents[1] / "tests/data/mining_samples.json"


def _field(pair: object, name: str) -> str:
    if isinstance(pair, dict):
        return str(pair.get(name) or "")
    return str(getattr(pair, name, "") or "")


def check_sample(sample: dict, pairs: list[object]) -> tuple[bool, list[str]]:
    """Check extraction, source-answer fidelity, and private/unsupported claims."""
    reasons: list[str] = []
    if sample.get("expect_empty"):
        if pairs:
            reasons.append("expected no reusable knowledge")
        return not reasons, reasons

    expected_count = sample.get("expect_pair_count")
    if expected_count is not None and len(pairs) != expected_count:
        reasons.append(f"expected {expected_count} pair(s), got {len(pairs)}")

    all_text = "\n".join(_field(pair, "question") + "\n" + _field(pair, "answer") for pair in pairs)
    for forbidden in sample.get("forbid_anywhere", []):
        if forbidden in all_text:
            reasons.append(f"private or unsupported text leaked: {forbidden}")

    for expected in sample.get("expect_pairs", []):
        matches = [
            pair for pair in pairs
            if all(fact in _field(pair, "question") for fact in expected.get("question_contains", []))
        ]
        if not matches:
            reasons.append("expected question missing")
            continue
        if not any(
            not check_answer_rules(expected, _field(pair, "answer"))
            and not check_answer_rules({
                "forbid_answer_contains": sample.get("forbid_answer_contains", []),
                "forbid_answer_regex_any": sample.get("forbid_answer_regex_any", []),
            }, _field(pair, "answer"))
            for pair in matches
        ):
            reasons.append("answer lost source facts or contains forbidden claim")
    return not reasons, reasons


async def main(samples_path: Path = SAMPLES) -> int:
    from app.kb import mining

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    failures = 0
    for sample in samples:
        pairs = await mining.extract_qa(sample["conversations"])
        ok, reasons = check_sample(sample, pairs)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {sample['id']}: {', '.join(reasons) if reasons else 'extraction verified'}")
    print(f"mining: {len(samples) - failures}/{len(samples)} passed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
