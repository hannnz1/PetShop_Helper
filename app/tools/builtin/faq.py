"""Knowledge lookup registration."""

from app.tools import registry
from app.tools.business import query_faq

registry.register(registry.spec_from_langchain_tool(query_faq, source="builtin", timeout=45.0))
