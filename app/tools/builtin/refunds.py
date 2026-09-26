"""Refund form suggestion registration; actual submission remains user-confirmed."""

from app.tools import registry
from app.tools.refunds import submit_refund

registry.register(registry.spec_from_langchain_tool(submit_refund, source="builtin"))
