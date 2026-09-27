"""Apply the additive knowledge-flywheel schema without replaying existing DDL."""

import asyncio
import re
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings


_DDL = Path(__file__).resolve().parents[2] / "sql" / "ch09-flywheel.sql"


def _statements() -> list[str]:
    lines = [
        line for line in _DDL.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    ]
    return [item.strip() for item in "\n".join(lines).split(";") if item.strip()]


async def apply_flywheel_schema(engine: AsyncEngine) -> None:
    """Apply each missing additive object; reruns preserve existing rows."""
    async with engine.begin() as connection:
        source_ref_exists = bool(await connection.scalar(text(
            "SELECT COUNT(*) FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'low_confidence_questions' "
            "AND COLUMN_NAME = 'source_ref'"
        )))
        tables = set((await connection.execute(text(
            "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE()"
        ))).scalars())
        for statement in _statements():
            if statement.startswith("ALTER TABLE"):
                if source_ref_exists:
                    continue
            else:
                match = re.match(r"CREATE TABLE\s+(\w+)", statement, re.IGNORECASE)
                if match is None:
                    raise ValueError("unexpected flywheel DDL statement")
                if match.group(1) in tables:
                    continue
            await connection.execute(text(statement))


async def main() -> None:
    engine = create_async_engine(get_settings().database_url)
    try:
        await apply_flywheel_schema(engine)
    finally:
        await engine.dispose()
    print("Flywheel schema is current.")


if __name__ == "__main__":
    asyncio.run(main())
