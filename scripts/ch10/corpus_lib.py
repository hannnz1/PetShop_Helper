"""Pure local operations for private, reproducible topic corpora."""

import random
import re

from app.core.taxonomy import LABEL2ID


_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{10,}(?!\d)")
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SOCIAL = re.compile(r"(微信|weixin|wx|QQ)\s*(?:号|[:：])?\s*[A-Za-z][A-Za-z0-9_-]{5,}", re.I)
_QQ_NUMBER = re.compile(r"(QQ\s*(?:号\s*)?[:：]?\s*)[1-9]\d{4,11}(?!\d)", re.I)
_ORDER = re.compile(r"(订单(?:编号|号)?\s*[:：#]?\s*)[A-Za-z0-9][A-Za-z0-9_-]{4,}", re.I)


def desensitize(value: str) -> str:
    """Mask common private identifiers without changing product model names."""
    value = _EMAIL.sub("[邮箱]", value)
    value = _PHONE.sub("[手机号]", value)
    value = _QQ_NUMBER.sub(r"\1[账号]", value)
    value = _LONG_NUMBER.sub("[单号]", value)
    value = _SOCIAL.sub(lambda match: f"{match.group(1)}[账号]", value)
    return _ORDER.sub(r"\1[单号]", value)


def dedupe(samples: list[dict]) -> list[dict]:
    """Keep the first occurrence of each nonblank exact text."""
    seen: set[str] = set()
    out: list[dict] = []
    for sample in samples:
        value = sample["text"].strip()
        if value and value not in seen:
            seen.add(value)
            out.append({**sample, "text": value})
    return out


def split_dataset(samples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratify multilabel originals into deterministic train/val/test sets."""
    rng = random.Random(seed)
    combinations: dict[tuple[str, ...], list[dict]] = {}
    for sample in samples:
        labels = sample["labels"]
        if not labels or any(label not in LABEL2ID for label in labels):
            raise ValueError("unknown or empty labels")
        key = tuple(sorted(set(labels), key=LABEL2ID.get))
        combinations.setdefault(key, []).append(dict(sample))
    strata: dict[tuple[str, ...], list[dict]] = {}
    for key, rows in combinations.items():
        target = key if len(rows) >= 10 else (min(key, key=LABEL2ID.get),)
        strata.setdefault(target, []).extend(rows)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for key in sorted(strata, key=lambda labels: tuple(LABEL2ID[label] for label in labels)):
        rows = list(strata[key])
        rng.shuffle(rows)
        n = len(rows)
        n_holdout = max(1, round(n * 0.1)) if n >= 3 else 0
        test.extend(rows[:n_holdout])
        val.extend(rows[n_holdout:2 * n_holdout])
        train.extend(rows[2 * n_holdout:])
    return train, val, test
