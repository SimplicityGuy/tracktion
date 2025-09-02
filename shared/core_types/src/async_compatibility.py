"""Async/Sync compatibility layer for database operations."""

import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseCompatibilityLayer:
    """Provides compatibility between sync and async database operations."""

    def __init__(
        self,
        database_url: str,
        use_async: bool = True,
        feature_flag_callback: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize compatibility layer.

        Args:
            database_url: Database connection URL
            use_async: Whether to use async operations by default
            feature_flag_callback: Optional callback to check feature flag
        """
        self.use_async = use_async
        self.feature_flag_callback = feature_flag_callback

        # Setup both sync and async engines
        self.sync_url = database_url
        if database_url.startswith("postgresql+asyncpg://"):
            self.sync_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

        self.async_url = database_url
        if database_url.startswith("postgresql://"):
            self.async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        # Create engines
        self.sync_engine = create_engine(self.sync_url)
        self.async_engine = create_async_engine(self.async_url)

        # Create session factories
        self.sync_session_factory = sessionmaker(bind=self.sync_engine)

        self.async_session_factory = async_sessionmaker(bind=self.async_engine)

    def should_use_async(self) -> bool:
        """Check if async operations should be used.

        Returns:
            True if async should be used, False otherwise
        """
        if self.feature_flag_callback:
            return self.feature_flag_callback()
        return self.use_async

    async def get_async_session(self) -> AsyncSession:
        """Get an async database session.

        Returns:
            AsyncSession instance
        """
        return self.async_session_factory()

    def get_sync_session(self) -> Session:
        """Get a sync database session.

        Returns:
            Session instance
        """
        return self.sync_session_factory()

    def get_session(self) -> Session | AsyncSession:
        """Get appropriate session based on current mode.

        Returns:
            Session or AsyncSession based on mode
        """
        if self.should_use_async():
            return asyncio.run(self.get_async_session())
        return self.get_sync_session()

    async def execute_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an async function.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        # Run sync function in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    def execute_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a sync function.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if inspect.iscoroutinefunction(func):
            # Run async function synchronously
            return asyncio.run(func(*args, **kwargs))
        return func(*args, **kwargs)

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function in appropriate mode.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if self.should_use_async():
            if inspect.iscoroutinefunction(func):
                return asyncio.run(func(*args, **kwargs))

            # Wrap sync function for async execution
            async def wrapper() -> Any:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)

            return asyncio.run(wrapper())
        return self.execute_sync(func, *args, **kwargs)

    def close(self) -> None:
        """Close all database connections."""
        self.sync_engine.dispose()
        asyncio.run(self.async_engine.dispose())


def compatibility_wrapper(use_async: bool = True) -> Callable[..., Any]:
    """Decorator to make functions compatible with both sync and async modes.

    Args:
        use_async: Whether to use async mode by default

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if use_async and not inspect.iscoroutinefunction(func):
                # Convert sync function to async
                async def async_wrapper() -> Any:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, *args, **kwargs)

                return asyncio.run(async_wrapper())
            if not use_async and inspect.iscoroutinefunction(func):
                # Convert async function to sync
                return asyncio.run(func(*args, **kwargs))
            # Execute as-is
            return func(*args, **kwargs)

        # Handle async functions properly
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if use_async:
                    return await func(*args, **kwargs)
                # This branch shouldn't normally be hit for async functions
                return await func(*args, **kwargs)

            return async_wrapper

        return wrapper

    return decorator


class RepositoryCompatibilityMixin:
    """Mixin to add compatibility support to repository classes."""

    def __init__(self, compatibility_layer: DatabaseCompatibilityLayer):
        """Initialize with compatibility layer.

        Args:
            compatibility_layer: Database compatibility layer instance
        """
        self.compat = compatibility_layer

    async def _execute_async(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an operation asynchronously.

        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result
        """
        async with await self.compat.get_async_session() as session:
            kwargs["session"] = session
            result = await operation(*args, **kwargs)
            await session.commit()
            return result

    def _execute_sync(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an operation synchronously.

        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result
        """
        with self.compat.get_sync_session() as session:
            kwargs["session"] = session
            result = operation(*args, **kwargs)
            session.commit()
            return result

    def execute(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute operation in appropriate mode.

        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result
        """
        if self.compat.should_use_async():
            return asyncio.run(self._execute_async(operation, *args, **kwargs))
        return self._execute_sync(operation, *args, **kwargs)


# Feature flag management
class AsyncModeConfig:
    """Configuration manager for async mode setting."""

    _enabled: bool = True

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """Set async mode.

        Args:
            enabled: Whether to enable async mode
        """
        cls._enabled = enabled
        logger.info(f"Async mode set to: {enabled}")

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if async mode is enabled.

        Returns:
            True if async mode is enabled
        """
        return cls._enabled


def set_async_mode(enabled: bool) -> None:
    """Set global async mode.

    Args:
        enabled: Whether to enable async mode
    """
    AsyncModeConfig.set_enabled(enabled)


def is_async_enabled() -> bool:
    """Check if async mode is enabled.

    Returns:
        True if async mode is enabled
    """
    return AsyncModeConfig.is_enabled()


# Environment-based configuration
def configure_from_environment() -> None:
    """Configure async mode from environment variables."""
    async_mode = os.getenv("TRACKTION_ASYNC_MODE", "true").lower()
    set_async_mode(async_mode in ("true", "1", "yes", "on"))


# Auto-configure on import
configure_from_environment()
