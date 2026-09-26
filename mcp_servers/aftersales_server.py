"""Mock warranty and return-status MCP Server over Streamable HTTP."""

import asyncio
import os
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(
    "aftersales", host="127.0.0.1", port=int(os.environ.get("PORT", "8102")),
    stateless_http=True, json_response=True,
)


async def _delay() -> None:
    seconds = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))
    if seconds > 0:
        await asyncio.sleep(seconds)


@mcp.tool()
async def query_warranty(order_id: Annotated[str, Field(description="订单号，例如 1001")]) -> dict:
    """查询演示订单商品的在保状态和保修截止日期。"""

    await _delay()
    rng = random.Random(f"warranty:{order_id}")
    return {
        "order_id": order_id, "warranty_code": rng.choice(("IN_WARRANTY", "EXPIRED")),
        "warranty_until": f"2026-{rng.randint(8, 12):02d}-{rng.randint(1, 28):02d}",
        "policy_ref": "AS-POLICY-07",
    }


@mcp.tool()
async def query_return_status(order_id: Annotated[str, Field(description="订单号，例如 1001")]) -> dict:
    """查询演示订单的退货进度。"""

    await _delay()
    rng = random.Random(f"return:{order_id}")
    return {
        "order_id": order_id,
        "return_code": rng.choice(("AUDITING", "RETURNING", "REFUNDED", "NONE")),
        "updated_at": f"2026-07-{rng.randint(1, 16):02d} 10:00",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
