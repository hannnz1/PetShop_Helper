"""Order and product tool registrations."""

from app.tools import registry
from app.tools.business import query_order, query_product

registry.register(registry.spec_from_langchain_tool(query_order, source="builtin"))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
