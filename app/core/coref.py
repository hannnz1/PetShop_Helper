"""Resolve short references for routing and retrieval; never rewrite audit text."""

import re

from langchain_core.language_models import BaseChatModel

from app.core.prompts import COREF_PROMPT


_REFERENCE = re.compile(r"它|那个|这个|这单|那单|刚才|前面|上面|上一[个条]|同一|还有呢|那呢")


async def resolve(query: str, history: str, model: BaseChatModel) -> str:
    """Return a self-contained question, falling back to the original on failure."""
    if not history.strip() or not _REFERENCE.search(query):
        return query
    try:
        answer = await (COREF_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"},
        )
        text = answer.content.strip() if isinstance(answer.content, str) else ""
    except Exception:  # noqa: BLE001 - optional upstream rewrite must not block a turn
        return query
    return text or query
