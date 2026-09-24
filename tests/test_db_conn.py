"""Exercise the chapter-two MySQL connection and isolated test schema."""

import pytest
from sqlalchemy import text
from types import SimpleNamespace

from app.db.base import Base, async_session, engine
from tests.conftest import ddl_statements, isolated_test_url
import tests.conftest as db_fixtures


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


@pytest.mark.parametrize(
    "app_url,test_url",
    [
        ("mysql+asyncmy://u:p@localhost/mewhelp_test", "mysql+asyncmy://u:p@localhost/mewhelp_test"),
        ("mysql+asyncmy://u:p@localhost/mewhelp_test", "mysql+asyncmy://other:pass@127.0.0.1:3306/mewhelp_test"),
        ("mysql+asyncmy://u:p@[::1]/MEWHELP_TEST", "mysql+asyncmy://u:p@127.0.0.1:3306/mewhelp_test"),
        ("mysql+asyncmy://u:p@db.example:3307/mewhelp_test", "mysql+asyncmy://u:p@other.example:3306/mewhelp_test"),
    ],
)
def test_same_schema_name_is_rejected_even_with_aliases(app_url, test_url):
    with pytest.raises(ValueError, match="application database"):
        db_fixtures.validate_database_isolation(test_url, app_url)


def test_distinct_application_and_test_schema_are_allowed():
    test_url = db_fixtures.validate_database_isolation(
        "mysql+asyncmy://u:p@127.0.0.1:3306/mewhelp_test",
        "mysql+asyncmy://u:p@localhost/mewhelp",
    )
    assert test_url.database == "mewhelp_test"


@pytest.mark.asyncio
async def test_fixture_rejects_shared_target_before_opening_any_engine(monkeypatch):
    app_url = "mysql+asyncmy://u:p@localhost/mewhelp_test"
    test_url = "mysql+asyncmy://other:pass@127.0.0.1:3306/mewhelp_test"
    monkeypatch.setattr(
        db_fixtures,
        "get_settings",
        lambda: SimpleNamespace(database_url=app_url, test_database_url=test_url),
    )

    def unexpected_engine(*args, **kwargs):
        raise AssertionError("engine constructed before isolation validation")

    monkeypatch.setattr(db_fixtures, "create_async_engine", unexpected_engine)
    fixture_generator = db_fixtures._test_engine.__wrapped__()
    with pytest.raises(ValueError, match="application database"):
        await anext(fixture_generator)


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
