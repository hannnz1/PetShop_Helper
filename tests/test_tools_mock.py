"""Contract checks for the three deterministic business mock tools."""

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.tools.business import query_logistics, query_order, query_product


@pytest.mark.parametrize(
    ("tool", "argument", "value", "expected_keys"),
    [
        (
            query_order,
            "order_id",
            "1001",
            {"order_id", "status", "amount", "created_at", "product", "tracking_no"},
        ),
        (
            query_product,
            "product_name",
            "猫粮",
            {"product_name", "price", "stock", "spec"},
        ),
        (
            query_logistics,
            "order_id",
            "1001",
            {"order_id", "status", "location", "timeline"},
        ),
    ],
)
async def test_mock_tool_contract_and_determinism(tool, argument, value, expected_keys):
    assert isinstance(tool, BaseTool)
    assert tool.name in {"query_order", "query_product", "query_logistics"}
    assert tool.args_schema.model_fields[argument].is_required()
    assert tool.tool_call_schema.model_fields[argument].is_required()
    assert tool.args_schema.model_fields[argument].description

    first = await tool.ainvoke({argument: value})
    second = await tool.ainvoke({argument: value})
    assert first == second
    assert isinstance(first, dict)
    assert set(first) == expected_keys
    assert first[argument] == value


async def test_query_order_values_and_different_order_ids():
    first = await query_order.ainvoke({"order_id": "1001"})
    second = await query_order.ainvoke({"order_id": "2002"})
    assert first["status"] in {"待付款", "已付款", "已发货", "已签收"}
    assert 50 <= first["amount"] <= 2000
    assert first["created_at"].startswith("2026-07-")
    assert first["product"] in {"智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"}
    assert second["order_id"] == "2002"


async def test_query_order_provides_stable_tracking_number_for_mcp_lookup():
    first = await query_order.ainvoke({"order_id": "1001"})
    again = await query_order.ainvoke({"order_id": "1001"})
    assert first["tracking_no"].startswith("SF")
    assert first["tracking_no"] == again["tracking_no"]


async def test_query_product_values():
    result = await query_product.ainvoke({"product_name": "猫粮"})
    assert 20 <= result["price"] <= 999
    assert 0 <= result["stock"] <= 500
    assert result["spec"] in {"标准装", "家庭装", "试用装"}


async def test_query_logistics_timeline_matches_status_and_location():
    result = await query_logistics.ainvoke({"order_id": "1001"})
    assert result["status"] in {"已揽件", "运输中", "派送中", "已签收"}
    assert result["location"]
    assert isinstance(result["timeline"], list)
    assert len(result["timeline"]) >= 1
    assert any(result["status"] in event for event in result["timeline"])
    assert any(result["location"] in event for event in result["timeline"])


async def test_order_and_logistics_share_consistent_state_across_orders():
    seen_logistics = set()
    for number in range(120):
        order_id = f"CASE-{number:03d}"
        order = await query_order.ainvoke({"order_id": order_id})
        logistics = await query_logistics.ainvoke({"order_id": order_id})
        seen_logistics.add(logistics["status"])
        if logistics["status"] == "已签收":
            assert order["status"] == "已签收"
        else:
            assert order["status"] == "已发货"
    assert seen_logistics == {"已揽件", "运输中", "派送中", "已签收"}


async def test_each_logistics_phase_has_plausible_location_and_chronological_events():
    scenarios = {}
    for number in range(120):
        result = await query_logistics.ainvoke({"order_id": f"CASE-{number:03d}"})
        scenarios.setdefault(result["status"], result)
    assert set(scenarios) == {"已揽件", "运输中", "派送中", "已签收"}

    expected = {
        "已揽件": ("揽收网点", ["已揽件"]),
        "运输中": ("分拨中心", ["已揽件", "运输中"]),
        "派送中": ("派送站", ["已揽件", "运输中", "派送中"]),
        "已签收": ("收货地址", ["已揽件", "运输中", "派送中", "已签收"]),
    }
    for status, result in scenarios.items():
        location_suffix, events = expected[status]
        assert result["location"].endswith(location_suffix)
        assert len(result["timeline"]) == len(events)
        for event_text, phase in zip(result["timeline"], events, strict=True):
            assert phase in event_text
        assert result["location"] in result["timeline"][-1]


@pytest.mark.parametrize(
    ("tool", "expected_name"),
    [
        (query_order, "query_order"),
        (query_product, "query_product"),
        (query_logistics, "query_logistics"),
    ],
)
async def test_tool_names_and_missing_required_argument(tool, expected_name):
    assert tool.name == expected_name
    with pytest.raises(ValidationError):
        await tool.ainvoke({})
