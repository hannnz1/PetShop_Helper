"""Apply Chapter 6's additive demo order and refund-request tables."""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402


async def apply_migration(engine) -> str:
    database = engine.url.database
    if not database:
        raise RuntimeError("MySQL database name is required")
    async with engine.connect() as conn:
        rows = (await conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema=:db AND table_name IN ('sample_orders', 'refund_requests')
        """), {"db": database})).scalars().all()
        found = set(rows)
        if found == {"sample_orders", "refund_requests"}:
            return "already applied"
        if found:
            raise RuntimeError(f"partial Ch06 schema: {sorted(found)}")
        ddl = (Path(__file__).resolve().parents[1] / "sql" / "ch06-ddl.sql").read_text(encoding="utf-8")
        lines = [line for line in ddl.splitlines() if not line.lstrip().startswith("--")]
        for statement in (part.strip() for part in "\n".join(lines).split(";")):
            if statement:
                await conn.execute(text(statement))
        await conn.commit()
        return "applied"


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        if not engine.url.database or engine.url.database.endswith("_test"):
            raise RuntimeError("migration requires the configured application database")
        print(f"Ch06 migration {await apply_migration(engine)}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
