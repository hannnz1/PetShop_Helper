"""Isolated MySQL fixtures for chapter-two integration tests."""

from pathlib import Path
import ipaddress
import re

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings


_DDL = Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql"
_TABLES = ("messages", "tickets", "conversations", "faq")


def isolated_test_url(raw_url: str) -> URL:
    """Reject any URL that might target the application database."""

    url = make_url(raw_url)
    name = url.database or ""
    if url.drivername != "mysql+asyncmy" or not re.fullmatch(r"[A-Za-z0-9_]+_test", name):
        raise ValueError("TEST_DATABASE_URL must use mysql+asyncmy and an isolated *_test database")
    return url


def _database_target(url: URL) -> tuple[str, int, str]:
    """Normalize common MySQL endpoint aliases for an isolation comparison."""

    host = (url.host or "localhost").lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host in {"localhost", "localhost.localdomain"}:
            host = "loopback"
    else:
        if address.is_loopback or address.is_unspecified:
            host = "loopback"
        else:
            host = address.compressed
    return host, url.port or 3306, (url.database or "").casefold()


def validate_database_isolation(test_raw_url: str, app_raw_url: str) -> URL:
    """Require distinct schemas before creating any engine or running any SQL.

    Rejecting the same schema name even on different named hosts is deliberately
    conservative: DNS aliases and tunnels can map those names to one server.
    """

    test_url = isolated_test_url(test_raw_url)
    app_url = make_url(app_raw_url)
    if app_url.drivername != "mysql+asyncmy" or not app_url.database:
        raise ValueError("DATABASE_URL must name the application MySQL database")
    test_target = _database_target(test_url)
    app_target = _database_target(app_url)
    if test_target == app_target or test_target[2] == app_target[2]:
        raise ValueError("TEST_DATABASE_URL must not target the application database")
    return test_url


def ddl_statements() -> list[str]:
    """Read the authoritative SQL file, preserving SQL but removing full-line comments."""

    lines = [
        line for line in _DDL.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    ]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _test_engine():
    settings = get_settings()
    url = validate_database_isolation(settings.test_database_url, settings.database_url)
    name = url.database
    admin_url = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        query=url.query,
    )
    admin = create_async_engine(admin_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4"))
    finally:
        await admin.dispose()

    engine = create_async_engine(url, pool_pre_ping=True, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            try:
                for table in _TABLES:
                    await conn.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
            finally:
                await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
            for statement in ddl_statements():
                await conn.execute(text(statement))
            await conn.commit()
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def db_session_factory(_test_engine, monkeypatch):
    """Repositories must read app.db.base.async_session inside their call path."""

    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory


@pytest_asyncio.fixture
async def db_clean(_test_engine):
    yield
    async with _test_engine.connect() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        try:
            for table in _TABLES:
                await conn.execute(text(f"TRUNCATE TABLE `{table}`"))
        finally:
            await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        await conn.commit()
