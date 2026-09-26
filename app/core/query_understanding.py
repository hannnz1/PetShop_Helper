"""Normalize a customer question only for retrieval."""

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="保留实体、型号和条件的标准问法")
    expanded: list[str] = Field(default_factory=list, description="相关同义词或近义词")


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
