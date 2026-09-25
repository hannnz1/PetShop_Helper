"""Question-only deduplication for mined knowledge candidates."""

import re
import unicodedata


_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_question(q: str) -> str:
    """Fold Unicode variants, then remove whitespace and punctuation."""
    return _STRIP_RE.sub("", unicodedata.normalize("NFKC", q).strip().lower())


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    """Keep the first new question, preserving input order and item identity."""
    seen = {normalize_question(question) for question in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
            continue
        seen.add(key)
        kept.append(item)
    return kept, discarded
