"""Shared test utilities library for Tracktion test suite.

This module provides common test utilities, fixtures, and data generation
functions that can be used across all service test suites to ensure
consistency and reduce duplication.
"""

__version__ = "1.0.0"

# Re-export public utilities for easy access
from .async_helpers import (
    AsyncTestHelper,
    async_event_loop,
    async_event_loop_policy,
    async_semaphore,
    async_task_pool,
    async_test_decorator,
    async_test_timeout,
    mock_async_context,
)
from .data_generators import (
    TestDataGenerator,
    generate_file_event,
    generate_hash,
    generate_recording_data,
    generate_sample_metadata,
    generate_timestamp,
    generate_track_data,
    generate_uuid_string,
)
from .database_helpers import (
    DatabaseTestHelper,
    database_test_data,
    mock_async_database_session,
    mock_database_session,
    mock_neo4j_driver,
    mock_redis_client,
    mock_redis_strict,
)
from .fixtures import (
    CommonTestFixtures,
    mock_audio_analyzer,
    mock_bpm_detector,
    mock_file_system,
    mock_key_detector,
    mock_message_queue,
    mock_metadata_extractor,
    mock_notification_service,
    mock_service_config,
    mock_storage_service,
    performance_metrics,
)
from .mock_helpers import (
    DatabaseMockHelper,
    MockBuilder,
    ServiceMockFactory,
    auto_mock_dependencies,
    create_async_mock,
    create_service_mock,
    mock_with_spec,
)

__all__ = [
    "AsyncTestHelper",
    "CommonTestFixtures",
    "DatabaseMockHelper",
    "DatabaseTestHelper",
    "MockBuilder",
    "ServiceMockFactory",
    "TestDataGenerator",
    "async_event_loop",
    # async_helpers
    "async_event_loop_policy",
    "async_semaphore",
    "async_task_pool",
    "async_test_decorator",
    "async_test_timeout",
    "auto_mock_dependencies",
    # mock_helpers
    "create_async_mock",
    "create_service_mock",
    "database_test_data",
    "generate_file_event",
    "generate_hash",
    "generate_recording_data",
    # data_generators
    "generate_sample_metadata",
    "generate_timestamp",
    "generate_track_data",
    "generate_uuid_string",
    "mock_async_context",
    "mock_async_database_session",
    "mock_audio_analyzer",
    "mock_bpm_detector",
    "mock_database_session",
    "mock_file_system",
    "mock_key_detector",
    "mock_message_queue",
    "mock_metadata_extractor",
    "mock_neo4j_driver",
    "mock_notification_service",
    # database_helpers
    "mock_redis_client",
    "mock_redis_strict",
    "mock_service_config",
    "mock_storage_service",
    "mock_with_spec",
    # fixtures
    "performance_metrics",
]
