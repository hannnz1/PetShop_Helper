"""Local masking and stable question normalization."""

import re

_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ORDER = re.compile(r"(订单(?:编号|号)?\s*[:：#]?\s*)[A-Za-z0-9][A-Za-z0-9_-]{3,}", re.I)


def mask_sensitive(value: str) -> str:
    value = _EMAIL.sub("[邮箱]", value)
    value = _PHONE.sub("[电话]", value)
    return _ORDER.sub(r"\1[订单号]", value)


def normalize_question(value: str) -> str:
    return re.sub(r"[\W_]+", "", mask_sensitive(value).casefold())
