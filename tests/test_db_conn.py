"""Exercise the chapter-two MySQL connection and isolated test schema."""

import pytest
from sqlalchemy import text

from app.db.base import Base, async_session, engine
from tests.conftest import ddl_statements, isolated_test_url


def test_ddl_parser_executes_names_and_all_four_tables():
    statements = ddl_statements()
    assert statements[0] == "SET NAMES utf8mb4"
    assert len(statements) == 5
    assert all(statement.startswith("CREATE TABLE") for statement in statements[1:])


@pytest.mark.parametrize(
    "url",
    [
        "mysql+asyncmy://root:root@localhost/mewhelp",
        "mysql+asyncmy://root:root@localhost/mewhelp_test_extra",
        "sqlite:///mewhelp_test",
    ],
)
def test_non_isolated_or_non_mysql_url_is_rejected(url):
    with pytest.raises(ValueError, match="isolated"):
        isolated_test_url(url)


def test_database_base_and_runtime_factory_are_available():
    assert Base.metadata is not None
    assert engine.url.drivername == "mysql+asyncmy"
    assert async_session.kw["expire_on_commit"] is False


@pytest.mark.asyncio
async def test_test_db_reachable_and_tables_exist(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        rows = (await conn.execute(text("SHOW TABLES"))).scalars().all()
    assert {"conversations", "messages", "faq", "tickets"} <= set(rows)


@pytest.mark.asyncio
async def test_test_factory_uses_isolated_database(db_session_factory, db_clean):
    async with db_session_factory() as session:
        name = (await session.execute(text("SELECT DATABASE()"))).scalar_one()
    assert name.endswith("_test")
