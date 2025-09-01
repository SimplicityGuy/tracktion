"""Database connection and session management for Tracktion."""

import logging
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from neo4j import (
    Driver,
    GraphDatabase,
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# SQLAlchemy Base for model definitions
Base = declarative_base()


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable[..., Any]:
    """Decorator for retrying database operations on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            current_delay = delay

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise

                    logger.warning(f"Attempt {attempt} failed, retrying in {current_delay}s: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

            return None

        return wrapper

    return decorator


class DatabaseManager:
    """Manages database connections for PostgreSQL and Neo4j."""

    def __init__(self) -> None:
        """Initialize database connections with retry logic."""
        self.pg_engine: Engine | None = None
        self.neo4j_driver: Driver | None = None
        self.SessionLocal: sessionmaker[Session] | None = None
        self._initialize_connections()

    @retry_on_failure(max_attempts=5, delay=2.0)
    def _initialize_connections(self) -> None:
        """Initialize database connections with retry logic."""
        # PostgreSQL connection
        if not self.pg_engine:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not set")

            self.pg_engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                connect_args={
                    "connect_timeout": 10,
                    "options": "-c statement_timeout=30000",  # 30 second statement timeout
                },
            )

            # Test connection
            with self.pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.SessionLocal = sessionmaker(bind=self.pg_engine)
            logger.info("PostgreSQL connection established")

        # Neo4j connection
        if not self.neo4j_driver:
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection environment variables not set")

            # After validation, we know these are not None
            assert neo4j_uri is not None
            assert neo4j_user is not None
            assert neo4j_password is not None

            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
            )

            # Test connection
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection established")

    @contextmanager
    def get_db_session(self) -> Generator[Session]:
        """Context manager for database sessions.

        Yields:
            SQLAlchemy session with automatic commit/rollback
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_neo4j_driver(self) -> Driver:
        """Get Neo4j driver instance.

        Returns:
            Neo4j driver for graph database operations
        """
        if not self.neo4j_driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self.neo4j_driver

    def close(self) -> None:
        """Close all database connections."""
        if self.pg_engine:
            self.pg_engine.dispose()
            logger.info("PostgreSQL connection closed")

        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions for backward compatibility
def get_db_session() -> Session:
    """Get a database session.

    Returns:
        SQLAlchemy session
    """
    if not db_manager.SessionLocal:
        raise RuntimeError("Database not initialized")
    return db_manager.SessionLocal()


def get_neo4j_driver() -> Driver:
    """Get Neo4j driver instance.

    Returns:
        Neo4j driver for graph database operations
    """
    return db_manager.get_neo4j_driver()
