"""Database connection and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (  # type: ignore[attr-defined]  # SQLAlchemy 2.0 features; project uses 2.0.43 but type stubs are 1.4.x
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

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


class DatabaseManagerSingleton:
    """Singleton wrapper for DatabaseManager."""

    _instance: DatabaseManager | None = None
    _singleton_instance: "DatabaseManagerSingleton | None" = None

    def __new__(cls) -> "DatabaseManagerSingleton":
        """Get the singleton DatabaseManager instance."""
        if cls._singleton_instance is None:
            cls._singleton_instance = super().__new__(cls)
            cls._instance = DatabaseManager()
        return cls._singleton_instance

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped database manager instance."""
        if self._instance is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._instance, name)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._singleton_instance = None


def get_db_manager() -> "DatabaseManagerSingleton":
    """Get the singleton database manager instance."""
    return DatabaseManagerSingleton()


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
