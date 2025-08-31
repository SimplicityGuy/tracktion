"""Async database connection and session management for Tracktion."""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Optional

import asyncpg  # type: ignore[import-untyped]
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text
from neo4j import AsyncGraphDatabase  # type: ignore[import-not-found]
from neo4j import AsyncDriver
from redis import asyncio as aioredis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# SQLAlchemy Base for model definitions (shared with sync version)
# Note: Base is imported from .database when needed to avoid circular imports


def async_retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable[..., Any]:
    """Decorator for retrying async database operations on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry

    Returns:
        Decorated async function with retry logic
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            current_delay = delay

            while attempt <= max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise

                    logger.warning(f"Attempt {attempt} failed, retrying in {current_delay}s: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

            return None

        return wrapper

    return decorator


class AsyncDatabaseManager:
    """Manages async database connections for PostgreSQL, Neo4j, and Redis."""

    def __init__(self) -> None:
        """Initialize async database connections."""
        self.pg_engine: Optional[AsyncEngine] = None
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.AsyncSessionLocal: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all async database connections."""
        if self._initialized:
            return

        await self._initialize_postgresql()
        await self._initialize_neo4j()
        await self._initialize_redis()
        self._initialized = True

    @async_retry_on_failure(max_attempts=5, delay=2.0)
    async def _initialize_postgresql(self) -> None:
        """Initialize async PostgreSQL connections."""
        # SQLAlchemy async engine
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/tracktion")

        # Convert to async URL format
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            async_url = database_url

        self.pg_engine = create_async_engine(
            async_url,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "server_settings": {"jit": "off"},
                "timeout": 10,
                "command_timeout": 30,
            },
        )

        # Test connection
        async with self.pg_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        self.AsyncSessionLocal = async_sessionmaker(
            self.pg_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Also create asyncpg pool for raw queries
        self.pg_pool = await asyncpg.create_pool(
            database_url,
            min_size=10,
            max_size=50,
            timeout=10,
            command_timeout=30,
        )

        logger.info("Async PostgreSQL connections established")

    @async_retry_on_failure(max_attempts=5, delay=2.0)
    async def _initialize_neo4j(self) -> None:
        """Initialize async Neo4j connection."""
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        self.neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=30,
        )

        # Test connection
        await self.neo4j_driver.verify_connectivity()
        logger.info("Async Neo4j connection established")

    @async_retry_on_failure(max_attempts=5, delay=2.0)
    async def _initialize_redis(self) -> None:
        """Initialize async Redis connection."""
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))

        self.redis_client = await aioredis.from_url(  # type: ignore[no-untyped-call]
            f"redis://{redis_host}:{redis_port}/{redis_db}",
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )

        # Test connection
        await self.redis_client.ping()
        logger.info("Async Redis connection established")

    @asynccontextmanager
    async def get_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async context manager for database sessions.

        Yields:
            SQLAlchemy async session with automatic commit/rollback
        """
        if not self.AsyncSessionLocal:
            await self.initialize()
            if not self.AsyncSessionLocal:
                raise RuntimeError("Database not initialized")

        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def get_pg_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get raw asyncpg connection from pool.

        Yields:
            asyncpg connection
        """
        if not self.pg_pool:
            await self.initialize()
            if not self.pg_pool:
                raise RuntimeError("PostgreSQL pool not initialized")

        async with self.pg_pool.acquire() as connection:
            yield connection

    def get_neo4j_driver(self) -> AsyncDriver:
        """Get async Neo4j driver instance.

        Returns:
            Async Neo4j driver for graph database operations
        """
        if not self.neo4j_driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self.neo4j_driver

    def get_redis_client(self) -> aioredis.Redis:
        """Get async Redis client instance.

        Returns:
            Async Redis client for cache operations
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        return self.redis_client

    async def close(self) -> None:
        """Close all async database connections."""
        if self.pg_engine:
            await self.pg_engine.dispose()
            logger.info("Async PostgreSQL engine closed")

        if self.pg_pool:
            await self.pg_pool.close()
            logger.info("Async PostgreSQL pool closed")

        if self.neo4j_driver:
            await self.neo4j_driver.close()
            logger.info("Async Neo4j connection closed")

        if self.redis_client:
            await self.redis_client.close()
            logger.info("Async Redis connection closed")


# Global async database manager instance
async_db_manager = AsyncDatabaseManager()


# Convenience functions
async def get_async_db_session() -> AsyncSession:
    """Get an async database session.

    Returns:
        SQLAlchemy async session
    """
    if not async_db_manager.AsyncSessionLocal:
        await async_db_manager.initialize()
        if not async_db_manager.AsyncSessionLocal:
            raise RuntimeError("Async database not initialized")
    return async_db_manager.AsyncSessionLocal()


async def get_async_neo4j_driver() -> AsyncDriver:
    """Get async Neo4j driver instance.

    Returns:
        Async Neo4j driver for graph database operations
    """
    if not async_db_manager.neo4j_driver:
        await async_db_manager.initialize()
    return async_db_manager.get_neo4j_driver()


async def get_async_redis_client() -> aioredis.Redis:
    """Get async Redis client instance.

    Returns:
        Async Redis client for cache operations
    """
    if not async_db_manager.redis_client:
        await async_db_manager.initialize()
    return async_db_manager.get_redis_client()
