# Cataloging Service Unit Tests

This directory contains comprehensive unit tests for the SQLAlchemy models in the cataloging service.

## Test Files Overview

### `test_models.py` - Complete Database Tests
Comprehensive tests for all SQLAlchemy models that require a PostgreSQL database connection. These tests cover:

- **Model creation and validation**
- **Relationship testing** (one-to-many, foreign keys)
- **Constraint validation** (unique fields, nullable fields, foreign key constraints)
- **Cascade delete operations**
- **Database indexes and performance**

**Database Requirements:**
- PostgreSQL server running on localhost:5432
- Database: `test_tracktion`
- User: `tracktion_user`
- Password: `changeme`
- Extensions: `uuid-ossp`

**Usage:**
```bash
# Run all database-dependent model tests
uv run pytest tests/unit/cataloging_service/test_models.py -v

# Run specific test classes
uv run pytest tests/unit/cataloging_service/test_models.py::TestRecordingModel -v
uv run pytest tests/unit/cataloging_service/test_models.py::TestMetadataModel -v
uv run pytest tests/unit/cataloging_service/test_models.py::TestTracklistModel -v
```

### `test_models_no_db.py` - Structure Tests (No Database Required)
Lightweight tests that validate model structure and behavior without requiring a database connection. These tests cover:

- **Model attribute validation**
- **Table name validation**
- **Type hint validation**
- **Model inheritance verification**
- **String representation (`__repr__`) methods**

**Usage:**
```bash
# Run structure tests without database
uv run pytest tests/unit/cataloging_service/test_models_no_db.py -v
```

### `conftest.py` - Test Configuration and Fixtures
Contains shared test fixtures and configuration for both test files:

- **Sample data fixtures** (recording data, metadata, track lists)
- **Mock object fixtures** for testing without database
- **Database session fixture** (commented out until database is available)

## Model Architecture

The tests cover three main SQLAlchemy models:

### Recording Model
- **Primary model** representing audio files in the catalog
- **Unique constraints** on SHA256 and XXH128 hashes
- **Indexed fields** for performance (file_path, hashes)
- **Relationships**: One-to-many with Metadata and Tracklist

### Metadata Model
- **Key-value pairs** associated with recordings
- **Foreign key** to Recording with cascade delete
- **Composite indexes** for efficient querying
- **Relationship**: Many-to-one with Recording

### Tracklist Model
- **Track information** stored as JSONB
- **Support for CUE files** and multiple sources
- **Foreign key** to Recording with cascade delete
- **Relationship**: Many-to-one with Recording

## Test Coverage

The tests achieve comprehensive coverage including:

- ✅ **Model Creation**: Valid and invalid scenarios
- ✅ **Field Validation**: Required fields, nullable fields, type validation
- ✅ **Unique Constraints**: Duplicate prevention, null handling
- ✅ **Foreign Key Constraints**: Orphan prevention, referential integrity
- ✅ **Relationships**: Bidirectional relationships, lazy loading
- ✅ **Cascade Operations**: Delete cascading, orphan cleanup
- ✅ **Database Indexes**: Performance optimization verification
- ✅ **JSONB Functionality**: Complex data storage and retrieval
- ✅ **String Representations**: Debugging and logging support

## Running Tests

### With Database (Complete Test Suite)
```bash
# Ensure PostgreSQL is running with test database
# Then run the complete test suite
uv run pytest tests/unit/cataloging_service/test_models.py -v

# Run with coverage
uv run pytest tests/unit/cataloging_service/test_models.py --cov=services.cataloging_service.src.models
```

### Without Database (Structure Tests Only)
```bash
# Run basic structure validation
uv run pytest tests/unit/cataloging_service/test_models_no_db.py -v
```

### All Tests (If Database Available)
```bash
# Run all cataloging service tests
uv run pytest tests/unit/cataloging_service/ -v
```

## Test Data

The tests use realistic sample data including:
- Music file paths and metadata
- BPM, key, and genre information
- Complex track listings with timestamps
- JSONB data with nested structures
- DJ set and tracklist information

## Notes

- Tests follow the project's standards using `uv run pytest`
- All tests pass pre-commit hooks (ruff, mypy, formatting)
- Database tests provide isolation through session-scoped fixtures
- Mock objects are used appropriately for non-database tests
- Comprehensive error scenarios are covered for robustness
