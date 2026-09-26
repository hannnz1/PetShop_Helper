"""Flat structured check for retrieval evidence sufficiency."""

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答当前问题")
    reason: str = Field(default="", description="判断依据")


def _chain():
    model = get_chat_model().with_structured_output(
        _Check, method=get_settings().structured_output_method,
    )
    return SELF_CHECK_PROMPT | model


async def check_sufficient(query: str, evidence_texts: list[str]) -> dict[str, bool | str]:
    """Return a flat useful/reason decision, skipping the model with no evidence."""
    if not evidence_texts:
        return {"useful": False, "reason": "没有检索证据"}
    evidence = "\n".join(f"[{index}] {text}" for index, text in enumerate(evidence_texts, 1))
    result: _Check = await _chain().ainvoke({"query": query, "evidence": evidence})
    return {"useful": bool(result.useful), "reason": result.reason or ""}
