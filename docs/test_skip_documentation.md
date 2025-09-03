# Skipped Tests Documentation

## Overview
This document tracks all skipped tests in the Tracktion project, documenting why they are skipped and plans for resolution.

## Skipped Tests by Category

### External Dependencies Required

#### Database Dependencies
- **File**: `tests/unit/cataloging_service/test_models.py`
  - **Reason**: Database not available
  - **Resolution**: Mock database connections or use test database fixtures
  - **Priority**: Medium

- **File**: `tests/unit/analysis_service/test_storage_handler_bpm.py`
  - **Reason**: Requires database connections
  - **Resolution**: Mock database operations or use SQLite for testing
  - **Priority**: Medium

#### Redis Dependencies
- **File**: `tests/unit/analysis_service/test_audio_cache.py`
  - **Reason**: Requires Redis server
  - **Resolution**: Use fakeredis or mock Redis operations
  - **Priority**: High (caching is critical)

- **File**: `tests/integration/test_production_cache_service.py`
  - **Reason**: Redis server not available
  - **Resolution**: Use Docker container for integration tests
  - **Priority**: High

#### RabbitMQ Dependencies
- **File**: `tests/integration/test_analysis_service.py`
  - **Reason**: Requires live RabbitMQ and database connections
  - **Resolution**: Use Docker compose for integration test environment
  - **Priority**: High (core service functionality)

#### Job Repository Dependencies
- **File**: `tests/integration/test_cue_generation_workflow.py`
  - **Count**: 4 skipped tests
  - **Reason**: Job repository not available
  - **Resolution**: Mock repository or use test fixtures
  - **Priority**: Medium

### Missing Test Files

#### BPM Integration Tests
- **File**: `tests/integration/test_bpm_integration.py`
  - **Count**: 8 skipped tests
  - **Reason**: Various test audio files not found
  - **Resolution**: Create minimal test audio files or use synthetic data
  - **Priority**: Low (can use mocked data)

- **File**: `tests/unit/analysis_service/test_bpm_detector_mock.py`
  - **Reason**: No reference test files found
  - **Resolution**: Generate synthetic test files
  - **Priority**: Low

### Complex Implementation Issues

#### Async Mocking
- **File**: `tests/unit/file_rename_service/test_feedback_loop.py`
  - **Reason**: Complex async mocking issue - existing functionality works
  - **Resolution**: Refactor to use proper AsyncMock patterns
  - **Priority**: Low (functionality works)

#### Missing Methods
- **File**: `tests/unit/file_rename_service/test_feedback_loop.py`
  - **Reason**: OnlineLearner doesn't have _train_model method
  - **Resolution**: Update test to match current implementation
  - **Priority**: Medium

## Resolution Strategy

### Phase 1: Mock External Dependencies (High Priority)
1. Implement fakeredis for Redis-dependent tests
2. Create database fixtures using SQLAlchemy test utilities
3. Mock RabbitMQ connections for unit tests

### Phase 2: Test Infrastructure (Medium Priority)
1. Set up Docker Compose for integration tests
2. Create test data fixtures
3. Implement proper async test patterns

### Phase 3: Test Data (Low Priority)
1. Generate synthetic audio files for BPM tests
2. Create minimal test datasets
3. Document test data requirements

## Metrics
- **Total Skipped Tests**: ~20
- **By Category**:
  - External Dependencies: 12
  - Missing Test Files: 6
  - Implementation Issues: 2

## Recommendations
1. **Immediate**: Add fakeredis to dependencies for Redis tests
2. **Short-term**: Create Docker Compose configuration for integration tests
3. **Long-term**: Establish test data management strategy
