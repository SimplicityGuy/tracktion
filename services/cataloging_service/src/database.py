"""Database connection and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_config


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self) -> None:
        """Initialize the database manager."""
        config = get_config()
        self.engine = create_async_engine(
            config.database.url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession]:
        """Get a database session."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close the database engine."""
        await self.engine.dispose()


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager  # noqa: PLW0603 - Necessary for singleton pattern
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """Get a database session for FastAPI dependency injection.

    This function is specifically designed for FastAPI's dependency injection system.
    It yields a session and handles transaction management automatically.
    The session will be committed on success or rolled back on exception.
    """
    db_manager = get_db_manager()
    async with db_manager.async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
