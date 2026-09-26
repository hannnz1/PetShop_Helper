"""Opt-in, non-overwriting demonstration orders for the demo-user identity."""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import base as db  # noqa: E402
from app.db.models import SampleOrder  # noqa: E402


_DEMO_ORDERS = (
    ("1001", "已签收", "猫粮 5kg", Decimal("88.00")),
    ("1002", "已发货", "自动饮水机", Decimal("139.00")),
)


async def seed_demo_orders(session_factory) -> int:
    created = 0
    async with session_factory.begin() as session:
        for order_id, status, product, amount in _DEMO_ORDERS:
            if await session.get(SampleOrder, order_id) is None:
                session.add(SampleOrder(
                    order_id=order_id, user_id="demo-user", status=status,
                    product=product, amount=amount,
                ))
                created += 1
    return created


async def main() -> None:
    engine = db.engine
    if not engine.url.database or engine.url.database.endswith("_test"):
        raise RuntimeError("demo seed requires the configured application database")
    try:
        print(f"Created {await seed_demo_orders(db.async_session)} demo orders")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
