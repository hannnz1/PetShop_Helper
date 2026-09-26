"""Resolve short references for routing and retrieval; never rewrite audit text."""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.core.memory import estimate_tokens
from app.core.prompts import COREF_PROMPT


_REFERENCE = re.compile(r"它|那个|这个|这单|那单|刚才|前面|上面|上一[个条]|同一|还有呢|那呢")


async def resolve(query: str, history: str, model: BaseChatModel,
                  *, max_tokens: int | None = None) -> str:
    """Return a self-contained question, falling back to the original on failure."""
    if not history.strip() or not _REFERENCE.search(query):
        return query
    budget = max_tokens if max_tokens is not None else max(1, get_settings().token_budget // 2)
    if budget <= 0:
        return query
    payload = {"query": query, "history": history}
    try:
        if estimate_tokens(COREF_PROMPT.invoke(payload).to_messages()) > budget:
            return query
        answer = await (COREF_PROMPT | model).ainvoke(payload)
        text = answer.content.strip() if isinstance(answer.content, str) else ""
        if text and estimate_tokens([AIMessage(content=text)]) > budget:
            return query
    except Exception:  # noqa: BLE001 - optional upstream rewrite must not block a turn
        return query
    return text or query
