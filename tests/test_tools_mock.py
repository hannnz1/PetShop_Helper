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
            {"order_id", "status", "amount", "created_at", "product"},
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


async def test_query_product_values():
    result = await query_product.ainvoke({"product_name": "猫粮"})
    assert 20 <= result["price"] <= 999
    assert 0 <= result["stock"] <= 500
    assert result["spec"] in {"标准装", "家庭装", "试用装"}


async def test_query_logistics_timeline_matches_status_and_location():
    result = await query_logistics.ainvoke({"order_id": "1001"})
    assert result["status"] in {"已揽件", "运输中", "派送中", "已签收"}
    assert result["location"].endswith("分拨中心")
    assert isinstance(result["timeline"], list)
    assert len(result["timeline"]) >= 2
    assert any(result["status"] in event for event in result["timeline"])
    assert any(result["location"] in event for event in result["timeline"])


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
