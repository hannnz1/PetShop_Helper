"""The five chapter-two tools exposed to the model and execution layer."""

from langchain_core.tools import BaseTool

from app.tools.business import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
)

_ALL: tuple[BaseTool, ...] = (
    query_order,
    query_product,
    query_logistics,
    query_faq,
    create_ticket,
)
_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in _ALL}

NO_RETRY: set[str] = {"create_ticket"}
INJECT_CONVERSATION: set[str] = {"create_ticket"}
DATABASE_TOOLS: set[str] = {"query_faq", "create_ticket"}


def get_all_tools() -> list[BaseTool]:
    return list(_ALL)


def get_tool(name: str) -> BaseTool | None:
    return _BY_NAME.get(name)
