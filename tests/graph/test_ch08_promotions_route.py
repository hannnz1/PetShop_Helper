"""A dropped-in read-only builtin must be reachable from activity chat."""

from app.graph import nodes
from app.graph.routing import route_by_intent
from app.tools import registry


def test_activity_route_exposes_new_read_only_builtin_without_core_edit():
    existing = registry.builtin_specs()[0]
    promotion = registry.ToolSpec(
        name="query_promotions", description="演示优惠", json_schema={"type": "object", "properties": {}},
        tool=existing.tool, permission="read", source="builtin",
    )
    assert route_by_intent("活动") == "business"
    visible = nodes._agent_specs({"route": "business", "intent": "活动"},
                                 [*registry.builtin_specs(), promotion])
    assert "query_promotions" in {spec.name for spec in visible}
