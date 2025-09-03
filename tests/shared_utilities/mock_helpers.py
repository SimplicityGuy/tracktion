"""Mock helpers and utilities for testing."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch


def create_async_mock(**kwargs) -> AsyncMock:
    """Create an AsyncMock with commonly needed attributes."""
    mock = AsyncMock(**kwargs)

    # Add common async context manager support
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)

    return mock


def create_service_mock(service_name: str, methods: list[str] | None = None) -> AsyncMock:
    """Create a mock service with common service methods."""
    if methods is None:
        methods = ["start", "stop", "health_check", "process", "initialize"]

    mock = AsyncMock()
    mock.name = service_name
    mock.is_running = False
    mock.is_healthy = True

    # Setup common service methods
    for method in methods:
        if method == "health_check":
            getattr(mock, method).return_value = {"status": "healthy", "service": service_name}
        elif method == "start":

            def start_side_effect():
                mock.is_running = True
                return True

            getattr(mock, method).side_effect = start_side_effect
        elif method == "stop":

            def stop_side_effect():
                mock.is_running = False
                return True

            getattr(mock, method).side_effect = stop_side_effect
        else:
            getattr(mock, method).return_value = True

    return mock


def mock_with_spec(spec_class: type, **kwargs) -> Mock:
    """Create a mock with a specific class spec."""
    return Mock(spec=spec_class, **kwargs)


def auto_mock_dependencies(module_path: str, dependencies: dict[str, Any]):
    """Context manager to auto-mock multiple dependencies."""
    return patch.multiple(module_path, **dict(dependencies.items()))


class MockBuilder:
    """Builder pattern for creating complex mocks."""

    def __init__(self):
        self._mock = AsyncMock()
        self._attributes = {}
        self._methods = {}

    def with_attribute(self, name: str, value: Any):
        """Add an attribute to the mock."""
        self._attributes[name] = value
        return self

    def with_method(self, name: str, return_value: Any = None, side_effect: Any = None):
        """Add a method to the mock."""
        method_mock = AsyncMock() if name.startswith("async_") else Mock()

        if return_value is not None:
            method_mock.return_value = return_value
        if side_effect is not None:
            method_mock.side_effect = side_effect

        self._methods[name] = method_mock
        return self

    def with_async_context_manager(self):
        """Make the mock work as an async context manager."""
        self._mock.__aenter__ = AsyncMock(return_value=self._mock)
        self._mock.__aexit__ = AsyncMock(return_value=None)
        return self

    def with_context_manager(self):
        """Make the mock work as a context manager."""
        self._mock.__enter__ = Mock(return_value=self._mock)
        self._mock.__exit__ = Mock(return_value=None)
        return self

    def build(self):
        """Build and return the configured mock."""
        # Set attributes
        for name, value in self._attributes.items():
            setattr(self._mock, name, value)

        # Set methods
        for name, method in self._methods.items():
            setattr(self._mock, name, method)

        return self._mock


class DatabaseMockHelper:
    """Helper for creating database-related mocks."""

    @staticmethod
    def create_session_mock(query_results: dict[str, list] | None = None):
        """Create a database session mock with query results."""
        session = Mock()

        if query_results:

            def mock_query(model):
                query_mock = Mock()
                results = query_results.get(model.__name__, [])
                query_mock.all.return_value = results
                query_mock.first.return_value = results[0] if results else None
                query_mock.count.return_value = len(results)
                query_mock.filter.return_value = query_mock
                query_mock.order_by.return_value = query_mock
                query_mock.limit.return_value = query_mock
                query_mock.offset.return_value = query_mock
                return query_mock

            session.query = mock_query
        else:
            session.query = Mock()

        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        session.delete = Mock()
        session.merge = Mock()
        session.flush = Mock()

        return session

    @staticmethod
    def create_async_session_mock(query_results: dict[str, list] | None = None):
        """Create an async database session mock with query results."""
        session = AsyncMock()

        if query_results:

            async def mock_execute(stmt):
                result_mock = AsyncMock()
                # Try to determine model from statement
                model_name = "Unknown"
                results = query_results.get(model_name, [])

                result_mock.fetchall = AsyncMock(return_value=results)
                result_mock.fetchone = AsyncMock(return_value=results[0] if results else None)
                result_mock.rowcount = len(results)

                # Mock scalars for SQLAlchemy async
                scalars_mock = AsyncMock()
                scalars_mock.all = AsyncMock(return_value=results)
                scalars_mock.first = AsyncMock(return_value=results[0] if results else None)
                result_mock.scalars = AsyncMock(return_value=scalars_mock)

                return result_mock

            session.execute = mock_execute
        else:
            session.execute = AsyncMock()

        session.add = Mock()  # add is typically sync
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        session.delete = Mock()  # delete is typically sync
        session.merge = AsyncMock()
        session.flush = AsyncMock()
        session.get = AsyncMock()

        return session


class ServiceMockFactory:
    """Factory for creating service-specific mocks."""

    @staticmethod
    def create_analysis_service_mock():
        """Create a mock analysis service."""
        service = create_service_mock("analysis_service")
        service.analyze_audio = AsyncMock(
            return_value={"bpm": 128.0, "key": "A minor", "energy": 0.8, "duration": 240.0}
        )
        service.extract_metadata = AsyncMock(
            return_value={"title": "Test Song", "artist": "Test Artist", "album": "Test Album", "genre": "Electronic"}
        )
        return service

    @staticmethod
    def create_storage_service_mock():
        """Create a mock storage service."""
        service = create_service_mock("storage_service")
        service.store_recording = AsyncMock(return_value="recording_123")
        service.store_metadata = AsyncMock(return_value=True)
        service.get_recording = AsyncMock(return_value=None)
        service.delete_recording = AsyncMock(return_value=True)
        return service

    @staticmethod
    def create_notification_service_mock():
        """Create a mock notification service."""
        service = create_service_mock("notification_service")
        service.send_notification = AsyncMock(return_value={"id": "notif_123"})
        service.send_discord_message = AsyncMock(return_value={"message_id": "discord_123"})
        return service
