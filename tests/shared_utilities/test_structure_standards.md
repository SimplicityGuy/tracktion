# Test Structure Standardization Guidelines

## Overview
This document outlines the standardized test structure and patterns to be used across all services in the Tracktion project, leveraging the shared test utilities library.

## Standard Test File Structure

```python
"""
Service-specific test module description.

Brief description of what this module tests.
"""

# Standard library imports
import asyncio
from pathlib import Path

# Third-party imports
import pytest

# Project imports
from services.{service_name}.src.{module} import {TargetClass}

# Shared utilities imports (standardized pattern)
from tests.shared_utilities.async_helpers import AsyncTestHelper, async_test_decorator
from tests.shared_utilities.data_generators import (
    generate_uuid_string,
    generate_hash,
    generate_timestamp,
    generate_sample_metadata
)
from tests.shared_utilities.database_helpers import (
    mock_redis_client,
    mock_async_database_session,
    DatabaseTestHelper
)
from tests.shared_utilities.fixtures import (
    mock_file_system,
    mock_message_queue,
    performance_metrics
)
from tests.shared_utilities.mock_helpers import (
    MockBuilder,
    ServiceMockFactory,
    auto_mock_dependencies
)


class Test{ServiceName}{ClassName}:
    """Test suite for {ServiceName}{ClassName} class."""

    # Fixture section - use shared utilities
    @pytest.fixture
    def mock_{dependency}(self):
        """Create mock dependency using shared utilities."""
        return MockBuilder()\
            .with_method("method_name", return_value=expected_value)\
            .with_attribute("attribute_name", value)\
            .build()

    @pytest.fixture
    def subject_under_test(self, mock_dependency):
        """Create the class under test with mocked dependencies."""
        return {TargetClass}(dependency=mock_dependency)

    # Test methods section
    def test_initialization(self):
        """Test class initialization."""
        # Test implementation
        pass

    @async_test_decorator(timeout=5.0)
    async def test_async_operation(self, subject_under_test):
        """Test async operations with proper timeout."""
        # Test implementation
        pass

    def test_with_generated_data(self, subject_under_test):
        """Test using generated test data."""
        # Use data generators
        uuid_value = generate_uuid_string()
        metadata = generate_sample_metadata(count=3)

        # Test implementation
        pass
```

## Fixture Management Standards

### 1. Mock Creation Pattern

**ALWAYS use MockBuilder for complex mocks:**

```python
@pytest.fixture
def mock_service(self):
    """Create mock service using MockBuilder pattern."""
    return MockBuilder()\
        .with_method("async_method", return_value=expected_result)\
        .with_method("sync_method", side_effect=custom_function)\
        .with_attribute("status", "ready")\
        .with_async_context_manager()\
        .build()
```

**Use ServiceMockFactory for common service patterns:**

```python
@pytest.fixture
def mock_analysis_service(self):
    return ServiceMockFactory.create_analysis_service_mock()

@pytest.fixture
def mock_storage_service(self):
    return ServiceMockFactory.create_storage_service_mock()
```

### 2. Database Testing Pattern

**For Redis testing:**

```python
def test_cache_operations(self, mock_redis_client):
    """Test cache operations with fake Redis client."""
    # Use mock_redis_client fixture - no setup needed
    pass
```

**For database testing:**

```python
async def test_database_operations(self, mock_async_database_session):
    """Test database operations with mock session."""
    # Use mock_async_database_session fixture
    pass
```

### 3. Data Generation Pattern

**Use shared data generators:**

```python
def test_with_test_data(self):
    """Test using generated data."""
    # UUID generation
    recording_id = generate_uuid_string()

    # Hash generation
    file_hash = generate_hash("test content")

    # Timestamp generation
    timestamp = generate_timestamp(offset_minutes=30)

    # Complex data generation
    metadata = generate_sample_metadata(count=5)
    tracks = generate_track_data(count=3)
```

### 4. File System Testing Pattern

**Use mock file system fixture:**

```python
async def test_file_operations(self, mock_file_system):
    """Test file operations with mock file system."""
    # mock_file_system provides pre-created test files
    test_files = list(mock_file_system.glob("*.mp3"))
    # Use test_files in test
```

### 5. Async Testing Pattern

**Use async helpers for consistent async testing:**

```python
@async_test_decorator(timeout=10.0)
async def test_async_operation_with_waiting(self):
    """Test async operation with condition waiting."""
    # Start async operation
    task = asyncio.create_task(some_async_operation())

    # Wait for condition
    await AsyncTestHelper.wait_for_condition(
        lambda: task.done(),
        timeout=5.0
    )

    result = await task
    assert result == expected_value
```

## Service-Specific Patterns

### Analysis Service
- Use `ServiceMockFactory.create_analysis_service_mock()` for analysis services
- Use `mock_file_system` fixture for audio file testing
- Use `generate_sample_metadata()` for audio metadata generation

### Tracklist Service
- Use `generate_track_data()` for track list generation
- Use `mock_async_database_session` for database operations
- Use `generate_uuid_string()` for tracklist IDs

### File Watcher Service
- Use `mock_file_system` fixture for file monitoring tests
- Use `generate_hash()` for hash validation tests
- Use `performance_metrics` fixture for timing tests

### Notification Service
- Use `ServiceMockFactory.create_notification_service_mock()` for notifications
- Use `mock_message_queue` fixture for message queue testing
- Use `generate_timestamp()` for notification timing

## Naming Conventions

### Fixtures
- `mock_{service_name}` - For service mocks
- `mock_{component}` - For component mocks
- `test_{data_type}` - For test data fixtures
- `{service}_under_test` - For the main class being tested

### Test Methods
- `test_initialization` - For constructor/setup tests
- `test_{method_name}_success` - For happy path tests
- `test_{method_name}_error` - For error condition tests
- `test_{method_name}_edge_case` - For edge case tests

### Test Classes
- `Test{ServiceName}{ClassName}` - Main test class pattern
- `Test{ServiceName}Integration` - Integration test classes
- `Test{ServiceName}Performance` - Performance test classes

## Migration Checklist

When standardizing an existing test file:

1. ✅ **Update imports** - Add shared utilities imports
2. ✅ **Replace manual mocks** - Use MockBuilder and ServiceMockFactory
3. ✅ **Replace manual data generation** - Use data_generators functions
4. ✅ **Replace database mocks** - Use database_helpers fixtures
5. ✅ **Replace file creation** - Use mock_file_system fixture
6. ✅ **Add async helpers** - Use async_test_decorator and AsyncTestHelper
7. ✅ **Standardize fixture naming** - Follow naming conventions
8. ✅ **Update test method names** - Follow naming conventions
9. ✅ **Run tests** - Ensure all tests still pass
10. ✅ **Update documentation** - Document any service-specific patterns

## Benefits of Standardization

1. **Consistency** - All tests follow the same patterns across services
2. **Maintainability** - Centralized test utilities reduce duplication
3. **Reliability** - Shared utilities are thoroughly tested
4. **Performance** - Optimized fixtures and helpers improve test speed
5. **Developer Experience** - Familiar patterns reduce cognitive load
6. **Quality** - Standardized patterns encourage best practices

## Example Service Migration

See `/tests/unit/analysis_service/test_message_consumer.py` for a complete example of how to apply these standards to an existing test file.
