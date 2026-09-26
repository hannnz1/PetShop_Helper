"""Read tools exposed to models; the legacy ticket tool is not model-bound."""

from langchain_core.tools import BaseTool

from app.tools.business import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
)
from app.tools.refunds import submit_refund

_ALL: tuple[BaseTool, ...] = (
    query_order,
    query_product,
    query_logistics,
    query_faq,
)
_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in (*_ALL, create_ticket, submit_refund)}

NO_RETRY: set[str] = {"create_ticket"}
INJECT_CONVERSATION: set[str] = {"create_ticket"}
DATABASE_TOOLS: set[str] = {"query_faq", "create_ticket"}


def get_all_tools() -> list[BaseTool]:
    return list(_ALL)


def get_chat_tools(route: str) -> list[BaseTool]:
    """The graph binds only read tools; writes require a user action."""
    if route == "knowledge":
        return [query_order]
    if route == "refund":
        return [submit_refund]
    if route == "business":
        return [query_order, query_product, query_logistics]
    return []


def get_tool(name: str) -> BaseTool | None:
    return _BY_NAME.get(name)
