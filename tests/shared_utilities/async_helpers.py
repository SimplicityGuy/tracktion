"""Async testing helpers and utilities."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(scope="session")
def async_event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def async_event_loop(async_event_loop_policy):
    """Create an event loop for async tests."""
    loop = async_event_loop_policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 5.0  # 5 second timeout for async operations


@pytest.fixture
def async_semaphore():
    """Semaphore for limiting concurrent operations in tests."""
    return asyncio.Semaphore(10)  # Limit to 10 concurrent operations


@pytest.fixture
async def async_task_pool() -> AsyncGenerator[Callable]:
    """Task pool for managing concurrent test operations."""
    tasks = []

    async def add_task(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    yield add_task

    # Cancel and cleanup all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def mock_async_context():
    """Create a mock async context manager."""
    context_mock = AsyncMock()
    context_mock.__aenter__ = AsyncMock(return_value=context_mock)
    context_mock.__aexit__ = AsyncMock(return_value=None)
    return context_mock


def async_test_decorator(timeout: float = 5.0):
    """Decorator for async tests with timeout."""

    def decorator(test_func):
        return pytest.mark.asyncio(timeout=timeout)(test_func)

    return decorator


class AsyncTestHelper:
    """Helper class for common async test operations."""

    @staticmethod
    async def wait_for_condition(condition: Callable[[], bool], timeout: float = 1.0, interval: float = 0.01):
        """Wait for a condition to become true with timeout."""
        start_time = asyncio.get_event_loop().time()
        while not condition():
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Condition not met within {timeout}s")
            await asyncio.sleep(interval)

    @staticmethod
    async def collect_async_results(async_iterable, max_items: int = 100):
        """Collect results from an async iterable with a limit."""
        results = []
        count = 0
        async for item in async_iterable:
            results.append(item)
            count += 1
            if count >= max_items:
                break
        return results

    @staticmethod
    def create_async_mock_with_side_effects(side_effects: list) -> AsyncMock:
        """Create an async mock with predefined side effects."""
        mock = AsyncMock()
        mock.side_effect = side_effects
        return mock
