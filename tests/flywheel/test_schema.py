"""Flywheel migration adds queue tables without replacing raw questions."""

import pytest
from sqlalchemy import text

from app.db import repository


@pytest.mark.asyncio
async def test_flywheel_tables_and_feedback_identity_exist(db_session_factory):
    async with db_session_factory() as session:
        tables = set((await session.execute(text(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN "
            "('canonical_questions', 'canonical_occurrences', 'flywheel_review_actions')"
        ))).scalars())
        column = (await session.execute(text(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'low_confidence_questions' "
            "AND COLUMN_NAME = 'source_ref'"
        ))).scalar_one_or_none()
    assert tables == {
        "canonical_questions", "canonical_occurrences", "flywheel_review_actions",
    }
    assert column == "source_ref"


@pytest.mark.asyncio
async def test_additive_migration_can_rerun_without_losing_existing_rows(
    _test_engine, db_session_factory, db_clean,
):
    from scripts.ch09.migrate_flywheel import apply_flywheel_schema

    row_id = await repository.insert_low_confidence(
        None, "旧的低置信问题", "self_check", None,
    )
    await apply_flywheel_schema(_test_engine)
    await apply_flywheel_schema(_test_engine)
    async with db_session_factory() as session:
        question = (await session.execute(text(
            "SELECT raw_question FROM low_confidence_questions WHERE id = :id"
        ), {"id": row_id})).scalar_one()
    assert question == "旧的低置信问题"
