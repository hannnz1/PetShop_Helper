"""SQLAlchemy asynchronous engine and session factory."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def create_database_engine(settings: Settings | None = None):
    """Construct the engine without opening a database connection."""

    config = settings if settings is not None else get_settings()
    return create_async_engine(config.database_url, pool_pre_ping=True)


engine = create_database_engine()
async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)
