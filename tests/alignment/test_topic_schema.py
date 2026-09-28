import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_migration_is_additive_and_repeatable(_test_engine, db_session_factory, db_clean):
    from scripts.ch10.migrate_topics import apply_topic_schema
    await apply_topic_schema(_test_engine)
    async with db_session_factory.begin() as session:
        await session.execute(text("INSERT INTO topic_classification_runs (run_id,status,pending_count,success_count,failed_count) VALUES ('preserve','empty',0,0,0)"))
    await apply_topic_schema(_test_engine)
    async with db_session_factory() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM topic_classification_runs WHERE run_id='preserve'")) == 1
