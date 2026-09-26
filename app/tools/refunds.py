"""A model-visible refund form suggestion, never a payment or database write."""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from app.db import repository


@tool
async def submit_refund(
    order_id: str,
    user_id: Annotated[str, InjectedToolArg],
    reason: str | None = None,
) -> dict:
    """Suggest a refund request form for an owned demo order; the user must confirm separately."""
    if await repository.get_owned_sample_order(user_id, order_id) is None:
        return {"status": "订单不属于当前用户", "order_id": order_id}
    return {"status": "待用户确认", "order_id": order_id, "reason": reason}
