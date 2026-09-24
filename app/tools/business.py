"""Deterministic mock business lookups for chapter 2 tool calling."""

import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    order_id: str = Field(description="订单号，例如 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词，例如 猫粮")


class LogisticsInput(BaseModel):
    order_id: str = Field(description="订单号，用于查询该订单的物流轨迹")


_LOGISTICS_PHASES = ("已揽件", "运输中", "派送中", "已签收")
_PHASE_LOCATIONS = ("揽收网点", "分拨中心", "派送站", "收货地址")


def _mock_order_state(order_id: str) -> tuple[int, str]:
    """Choose one shared shipment phase and city for both order lookups."""

    rng = random.Random(f"order:{order_id}")
    return rng.randrange(len(_LOGISTICS_PHASES)), rng.choice(
        ["深圳", "广州", "杭州", "上海", "成都"]
    )


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单状态、金额、下单时间和商品名。用户询问具体订单时使用。"""

    phase, _ = _mock_order_state(order_id)
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": "已签收" if phase == 3 else "已发货",
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
    }


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品价格、库存和规格。用户咨询某个商品时使用。"""

    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }


@tool(args_schema=LogisticsInput)
async def query_logistics(order_id: str) -> dict:
    """查询订单物流状态、当前位置和轨迹。用户询问快递进度时使用。"""

    phase, city = _mock_order_state(order_id)
    location = f"{city}{_PHASE_LOCATIONS[phase]}"
    return {
        "order_id": order_id,
        "status": _LOGISTICS_PHASES[phase],
        "location": location,
        "timeline": [
            f"{city}{_PHASE_LOCATIONS[index]} {_LOGISTICS_PHASES[index]}"
            for index in range(phase + 1)
        ],
    }
