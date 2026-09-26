"""Apply Chapter 7 schema additions without clearing existing conversations."""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402


_COLUMNS = ("summary", "summary_upto_msg_id", "layer1_from_msg_id")

# Single-statement triggers deliberately live outside semicolon-split DDL.
# They guard direct SQL writes as well as repository updates.
GUARD_TRIGGERS = {
    "ch07_conversation_anchor_guard": """CREATE TRIGGER ch07_conversation_anchor_guard
        BEFORE UPDATE ON conversations FOR EACH ROW BEGIN
          IF COALESCE(NEW.summary_upto_msg_id, 0) < COALESCE(OLD.summary_upto_msg_id, 0)
             OR COALESCE(NEW.layer1_from_msg_id, 0) < COALESCE(OLD.layer1_from_msg_id, 0)
             OR COALESCE(NEW.summary_upto_msg_id, 0) > COALESCE(NEW.layer1_from_msg_id, 0)
          THEN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='Ch07 anchor cannot rewind or cross';
          END IF;
        END""",
    "ch07_summary_immutable_update": """CREATE TRIGGER ch07_summary_immutable_update
        BEFORE UPDATE ON conversation_summaries FOR EACH ROW
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='Ch07 summary segments are append only'""",
    "ch07_summary_immutable_delete": """CREATE TRIGGER ch07_summary_immutable_delete
        BEFORE DELETE ON conversation_summaries FOR EACH ROW
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='Ch07 summary segments are append only'""",
}


async def apply_migration(engine) -> str:
    database = engine.url.database
    if not database:
        raise RuntimeError("MySQL database name is required")
    ddl = (Path(__file__).resolve().parents[1] / "sql" / "ch07-ddl.sql").read_text(encoding="utf-8")
    lines = [line for line in ddl.splitlines() if not line.lstrip().startswith("--")]
    statements = [part.strip() for part in "\n".join(lines).split(";") if part.strip()]
    if len(statements) != 4:
        raise RuntimeError("unexpected Ch07 DDL shape")
    async with engine.connect() as conn:
        columns = set((await conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=:db AND table_name='conversations'
        """), {"db": database})).scalars())
        if not columns:
            raise RuntimeError("conversations table is missing; apply earlier migrations first")
        tables = set((await conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema=:db AND table_name='conversation_summaries'
        """), {"db": database})).scalars())
        complete = set(_COLUMNS) <= columns and bool(tables)
        for column, statement in zip(_COLUMNS, statements[:3]):
            if column not in columns:
                await conn.execute(text(statement))
        if not tables:
            await conn.execute(text(statements[3]))
        triggers = set((await conn.execute(text("""
            SELECT trigger_name FROM information_schema.triggers
            WHERE trigger_schema=:db AND trigger_name IN
              ('ch07_conversation_anchor_guard', 'ch07_summary_immutable_update',
               'ch07_summary_immutable_delete')
        """), {"db": database})).scalars())
        for name, statement in GUARD_TRIGGERS.items():
            if name not in triggers:
                await conn.execute(text(statement))
                complete = False
        await conn.commit()
        return "already applied" if complete else "applied"


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        if not engine.url.database or engine.url.database.endswith("_test"):
            raise RuntimeError("migration requires the configured application database")
        print(f"Ch07 migration {await apply_migration(engine)}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
