"""Annotated top-hit retrieval eval. Requires a built/vectorized KB and embed API.

Run from the repository root: ``python -m scripts.eval_retrieval``.
"""

import asyncio
import json
import re
from pathlib import Path


SAMPLES = Path(__file__).resolve().parents[1] / "tests/data/retrieval_samples.json"


def check_answer_rules(rules: dict, answer: str) -> list[str]:
    """Match annotated source facts and their relationship, not isolated tokens."""
    reasons: list[str] = []
    for fact in rules.get("expect_answer_contains_all", rules.get("answer_contains_all", [])):
        if fact not in answer:
            reasons.append(f"missing annotated answer fact: {fact}")
    for pattern in rules.get("answer_regex_all", []):
        if re.search(pattern, answer) is None:
            reasons.append("missing annotated answer relationship")
    for forbidden in rules.get("forbid_answer_contains", []):
        if forbidden in answer:
            reasons.append(f"unexpected answer claim: {forbidden}")
    for pattern in rules.get("forbid_answer_regex_any", []):
        if re.search(pattern, answer):
            reasons.append("unsupported answer promise")
    return reasons


def check_sample(sample: dict, top: dict | None) -> tuple[bool, list[str]]:
    """Check the first retrieved answer against independently annotated facts."""
    if top is None:
        return False, ["no top hit"]
    question = str(top.get("question") or "")
    answer = str(top.get("answer") or "")
    reasons: list[str] = []
    expected_question = sample.get("expect_question_contains")
    if expected_question and expected_question not in question:
        reasons.append("wrong top question")
    reasons.extend(check_answer_rules(sample, answer))
    return not reasons, reasons


async def main(samples_path: Path = SAMPLES) -> int:
    from app.core import retrieval

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    failures = 0
    for sample in samples:
        hits = await retrieval.search_knowledge(sample["query"])
        ok, reasons = check_sample(sample, hits[0] if hits else None)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {sample['id']}: {', '.join(reasons) if reasons else 'top answer verified'}")
    print(f"retrieval: {len(samples) - failures}/{len(samples)} passed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
