"""Ticket write-tool registration."""

from app.tools import registry
from app.tools.business import create_ticket

registry.register(registry.spec_from_langchain_tool(
    create_ticket, source="builtin", inject_conversation=True,
))
