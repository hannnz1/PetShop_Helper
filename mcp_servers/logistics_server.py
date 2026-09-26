"""Mock logistics MCP Server; run independently over Streamable HTTP."""

import asyncio
import os
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(
    "logistics", host="127.0.0.1", port=int(os.environ.get("PORT", "8101")),
    stateless_http=True, json_response=True,
)
_STATUS_CODES = ("PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED")
_CITIES = ("深圳", "广州", "杭州", "上海", "成都")


@mcp.tool()
async def query_logistics(
    tracking_no: Annotated[str, Field(description="物流单号；先查询订单取得 tracking_no")],
) -> dict:
    """按物流单号查询演示轨迹；物流单号不能用订单号代替。"""

    delay = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))
    if delay > 0:
        await asyncio.sleep(delay)
    rng = random.Random(f"logistics:{tracking_no}")
    code = rng.choice(_STATUS_CODES)
    city = rng.choice(_CITIES)
    return {
        "tracking_no": tracking_no, "status_code": code, "current_city": city,
        "trace": [f"{city}分拨中心已发出", f"内部状态码:{code}"],
        "carrier_code": "SF-EXP-01",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
