"""Add topic tables without touching existing rows."""
import asyncio
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import get_settings

DDL = Path(__file__).resolve().parents[2] / 'sql/ch10-topic-classifications.sql'

async def apply_topic_schema(engine):
    async with engine.begin() as connection:
        for statement in DDL.read_text(encoding='utf-8').split(';'):
            if statement.strip():
                await connection.execute(text(statement))

async def main():
    engine = create_async_engine(get_settings().database_url)
    try:
        await apply_topic_schema(engine)
    finally:
        await engine.dispose()
    print('Topic schema is current.')

if __name__ == '__main__':
    asyncio.run(main())
