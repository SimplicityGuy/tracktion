"""
Database connection and session management for tracklist service.

Provides SQLAlchemy database connection and session management
using PostgreSQL as the backend database.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from ..config import get_config
from ..models.tracklist import Base

logger = logging.getLogger(__name__)

# Global variables for connection
engine: Optional[Engine] = None
SessionLocal: Optional[sessionmaker[Session]] = None


def init_database() -> None:
    """Initialize database connection and create tables."""
    global engine, SessionLocal

    config = get_config()

    # Create database engine
    database_url = (
        f"postgresql://{config.database.user}:{config.database.password}"
        f"@{config.database.host}:{config.database.port}/{config.database.name}"
    )

    engine = create_engine(
        database_url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        pool_timeout=config.database.pool_timeout,
        pool_recycle=config.database.pool_recycle,
        echo=config.database.echo_queries,
    )

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def get_db_session() -> Generator[Session, None, None]:
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
def get_db_context() -> Generator[Session, None, None]:
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
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connections closed")
