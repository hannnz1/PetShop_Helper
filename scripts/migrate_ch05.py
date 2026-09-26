"""Apply the additive Chapter 5 ticket idempotency migration once."""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        database = engine.url.database
        if not database or database.endswith("_test"):
            raise RuntimeError("migration requires the configured application database")
        async with engine.connect() as conn:
            column = await conn.scalar(text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema=:db AND table_name='tickets' AND column_name='request_id'
            """), {"db": database})
            index = await conn.scalar(text("""
                SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema=:db AND table_name='tickets' AND index_name='uq_tickets_request_id'
            """), {"db": database})
            if column and index:
                print("Ch05 ticket migration already applied")
                return
            if column or index:
                raise RuntimeError("partial Ch05 ticket migration; inspect schema before retrying")
            ddl = (Path(__file__).resolve().parents[1] / "sql" / "ch05-ddl.sql").read_text(encoding="utf-8")
            statement = "\n".join(line for line in ddl.splitlines() if not line.lstrip().startswith("--")).strip().rstrip(";")
            await conn.execute(text(statement))
            await conn.commit()
            print("Ch05 ticket migration applied")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
