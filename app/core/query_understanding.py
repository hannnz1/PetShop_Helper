"""Normalize a customer question only for retrieval."""

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import POLICY_EXPANSION_PROMPT, QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="保留实体、型号和条件的标准问法")
    expanded: list[str] = Field(default_factory=list, description="相关同义词或近义词")


class _Expansion(BaseModel):
    queries: list[str] = Field(description="最多三条互补的退款政策检索问句")


async def expand_queries(query: str, model) -> list[str]:
    """Expand only on the refund path; preserve one fallback search query."""
    try:
        chain = POLICY_EXPANSION_PROMPT | model.with_structured_output(
            _Expansion, method=get_settings().structured_output_method,
        )
        parsed = _Expansion.model_validate(await chain.ainvoke({"query": query}))
        values = [item.strip() for item in parsed.queries if item.strip()][:3]
        return values or [query]
    except Exception:  # noqa: BLE001 - optional expansion cannot disable retrieval
        return [query]


async def understand(query: str) -> dict[str, str | list[str]]:
    """Return flat structured fields used by the retrieval layer."""
    if not query.strip():
        return {"standard": query, "expanded": []}
    model = get_chat_model().with_structured_output(
        _Rewrite, method=get_settings().structured_output_method,
    )
    result: _Rewrite = await (QUERY_REWRITE_PROMPT | model).ainvoke({"query": query})
    return {
        "standard": result.standard.strip() or query,
        "expanded": [term.strip() for term in result.expanded if term.strip()],
    }
