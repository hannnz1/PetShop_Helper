"""Import explicit human decisions by stable review ID before dataset building."""

import argparse
import json
from pathlib import Path

from app.core.taxonomy import LABEL2ID
from scripts.ch10.corpus_lib import desensitize


def apply_reviews(rows: list[dict], decisions: list[dict]) -> list[dict]:
    source = {}
    for row in rows:
        review_id = row.get("review_id") or (f"pool-{row['id']}" if row.get("id") is not None else None)
        if not review_id or review_id in source:
            raise ValueError("missing or duplicate review ID")
        source[review_id] = {**row, "review_id": review_id}
    approved = []
    seen = set()
    for decision in decisions:
        review_id = decision.get("review_id")
        if review_id not in source or review_id in seen:
            raise ValueError("unknown or duplicate review ID")
        seen.add(review_id)
        if decision.get("approved") is not True:
            continue
        labels = decision.get("labels")
        if not labels or any(label not in LABEL2ID for label in labels):
            raise ValueError("approved review has unknown or empty labels")
        row = source[review_id]
        if desensitize(row["text"]) != row["text"]:
            raise ValueError("reviewed text contains private identifier")
        approved.append({**row, "labels": list(dict.fromkeys(labels)), "reviewed": True})
    return approved


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/ch10/corpus_labeled.jsonl"))
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/ch10/corpus_reviewed.jsonl"))
    args = parser.parse_args()
    rows = apply_reviews(_read_jsonl(args.source), _read_jsonl(args.decisions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                        encoding="utf-8")
    print(f"approved_rows={len(rows)}")


if __name__ == "__main__":
    main()
