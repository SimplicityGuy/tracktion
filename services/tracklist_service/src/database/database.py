"""
Database connection and session management for tracklist service.

Provides SQLAlchemy database connection and session management
using PostgreSQL as the backend database.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.models.tracklist import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager with singleton pattern."""

    _instance: "DatabaseManager | None" = None
    _engine: Engine | None = None
    _session_local: sessionmaker[Session] | None = None
    _initialized: bool = False

    def __new__(cls) -> "DatabaseManager":
        """Get the singleton DatabaseManager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        if self._engine is None:
            self._initialize()
        if self._engine is None:
            raise RuntimeError("Failed to initialize database engine")
        return self._engine

    @property
    def session_local(self) -> sessionmaker[Session]:
        """Get the session factory."""
        if self._session_local is None:
            self._initialize()
        if self._session_local is None:
            raise RuntimeError("Failed to initialize session factory")
        return self._session_local

    def _initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        config = get_config()

        # Create database engine
        database_url = (
            f"postgresql://{config.database.user}:{config.database.password}"
            f"@{config.database.host}:{config.database.port}/{config.database.name}"
        )

        self._engine = create_engine(
            database_url,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_timeout=config.database.pool_timeout,
            pool_recycle=config.database.pool_recycle,
            echo=config.database.echo_queries,
        )

        # Create session factory
        self._session_local = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

        # Create tables
        Base.metadata.create_all(bind=self._engine)
        self._initialized = True
        logger.info("Database initialized successfully")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


# Singleton instance
_db_manager = DatabaseManager()

# Maintain compatibility with existing code
engine = _db_manager.engine
SessionLocal = _db_manager.session_local


def init_database() -> None:
    """Initialize database connection and create tables."""
    _db_manager._initialize()


def get_db_session() -> Generator[Session]:
    """
    Get database session for dependency injection.

    Yields:
        SQLAlchemy database session
    """
    if not SessionLocal:
        init_database()

    if not SessionLocal:
        raise RuntimeError("Failed to initialize database")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session]:
    """
    Get database session as context manager.

    Yields:
        SQLAlchemy database session
    """
    if not SessionLocal:
        init_database()

    if not SessionLocal:
        raise RuntimeError("Failed to initialize database")

    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def close_database() -> None:
    """Close database connections."""
    global engine  # noqa: PLW0602 - Global access necessary for proper database connection cleanup
    if engine:
        engine.dispose()
        logger.info("Database connections closed")
