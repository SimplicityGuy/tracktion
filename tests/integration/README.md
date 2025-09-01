# Integration Tests for Feedback System

This directory contains comprehensive integration tests for the feedback system components, including end-to-end feedback flow, API endpoints, database transactions, cache consistency, A/B testing, and resource management.

## Test Structure

```
tests/integration/file_rename_service/
├── test_feedback_integration.py    # Main integration test suite
├── conftest.py                     # Service-specific fixtures
├── docker-compose.test.yml         # Test services setup
├── sql/
│   └── init.sql                   # Database initialization
└── README.md                      # This file
```

## Test Coverage

### 1. End-to-End Feedback Flow Tests
- Complete feedback submission → storage → learning → metrics pipeline
- Feedback processing with batch operations
- Learning integration and model retraining triggers
- Error handling and recovery scenarios

### 2. API Endpoints with Authentication
- Feedback approval, rejection, and modification endpoints
- Authentication and authorization verification
- Rate limiting and input validation
- Admin-only endpoints protection

### 3. Database Transactions and Rollback
- Transaction integrity during failures
- Rollback behavior on storage errors
- Concurrent submission safety
- Data consistency verification

### 4. Cache Consistency Tests
- Redis-PostgreSQL consistency
- Cache invalidation on updates
- Recovery after cache failures
- Feedback count accuracy

### 5. A/B Testing Integration
- Complete experiment lifecycle
- Traffic allocation and variant assignment
- Statistical significance calculation
- Multiple concurrent experiments

### 6. Resource Management Under Load
- Backpressure handling strategies
- Memory monitoring and warnings
- Concurrent processing safety
- System cleanup and recovery

## Prerequisites

### Local Testing (Recommended)
Ensure you have the following services running locally:

1. **PostgreSQL** (version 12+)
   ```bash
   # Using Docker
   docker run -d --name postgres-test \
     -e POSTGRES_USER=tracktion_user \
     -e POSTGRES_PASSWORD=changeme \
     -e POSTGRES_DB=test_feedback \
     -p 5433:5432 \
     postgres:15-alpine
   ```

2. **Redis** (version 6+)
   ```bash
   # Using Docker
   docker run -d --name redis-test \
     -p 6380:6379 \
     redis:7-alpine
   ```

### Docker Compose Testing
Alternatively, use the provided Docker Compose setup:

```bash
cd tests/integration
docker-compose -f docker-compose.test.yml up -d
```

## Running Tests

### Environment Variables
Set the following environment variables for local testing:

```bash
export TEST_POSTGRES_DSN="postgresql://tracktion_user:changeme@localhost:5433/test_feedback"
export TEST_REDIS_URL="redis://localhost:6380/1"
```

### Run All Integration Tests
```bash
# From project root
uv run pytest tests/integration/file_rename_service/ -v

# With coverage
uv run pytest tests/integration/file_rename_service/ --cov=services.file_rename_service.app.feedback --cov-report=html

# Run only fast tests (exclude performance tests)
uv run pytest tests/integration/file_rename_service/ -v -m "not slow"
```

### Run Specific Test Classes
```bash
# End-to-end flow tests
uv run pytest tests/integration/file_rename_service/test_feedback_integration.py::TestEndToEndFeedbackFlow -v

# API endpoint tests
uv run pytest tests/integration/file_rename_service/test_feedback_integration.py::TestAPIEndpointsWithAuthentication -v

# A/B testing tests
uv run pytest tests/integration/file_rename_service/test_feedback_integration.py::TestABTestingIntegration -v

# Resource management tests
uv run pytest tests/integration/file_rename_service/test_feedback_integration.py::TestResourceManagementUnderLoad -v
```

### Run Performance Tests
```bash
# Performance and stress tests (marked as slow)
uv run pytest tests/integration/file_rename_service/test_feedback_integration.py::TestPerformanceUnderLoad -v -s
```

## Test Features

### Isolated Test Databases
Each test creates its own isolated database and Redis instance to ensure no cross-test contamination:
- Unique database names per test
- Separate Redis databases
- Automatic cleanup after tests

### Mock External Dependencies
- ML model operations are mocked to focus on integration logic
- Authentication is mocked for testing different user scenarios
- External services are mocked while testing real database/cache interactions

### Resource Monitoring
Tests include resource monitoring to verify:
- Memory usage patterns
- Queue utilization
- Processing throughput
- Error handling under load

### Performance Benchmarks
Performance tests verify:
- Throughput targets (>50 feedbacks/second)
- Response time limits (<5 seconds for 200 feedbacks)
- Resource efficiency under load
- Proper cleanup and resource release

## Test Data Management

### Automatic Cleanup
- Databases are automatically created and destroyed per test
- Redis databases are flushed between tests
- No manual cleanup required

### Sample Data Generation
Tests include utilities for generating:
- Realistic feedback data with various actions
- Experiment configurations with proper statistical setup
- Load testing scenarios with concurrent submissions

### Consistent Test Results
- Fixed random seeds where applicable
- Deterministic timing for batch processing tests
- Proper async test handling

## Debugging Tests

### Logging
Tests include comprehensive logging at INFO level:
```bash
# Run with debug logging
uv run pytest tests/integration/file_rename_service/ -v -s --log-cli-level=DEBUG
```

### Test Isolation
Each test runs in complete isolation:
- Fresh database per test
- Clean Redis state
- Independent storage instances

### Error Investigation
For test failures:
1. Check database connectivity
2. Verify Redis accessibility
3. Review test logs for specific error details
4. Ensure proper cleanup between tests

## CI/CD Integration

### GitHub Actions
Add to your GitHub Actions workflow:

```yaml
- name: Start test services
  run: |
    cd tests/integration
    docker-compose -f docker-compose.test.yml up -d

- name: Wait for services
  run: |
    timeout 60 bash -c 'until docker-compose -f tests/integration/docker-compose.test.yml exec postgres-test pg_isready -U tracktion_user; do sleep 2; done'

- name: Run integration tests
  run: |
    export TEST_POSTGRES_DSN="postgresql://tracktion_user:changeme@localhost:5433/test_feedback"
    export TEST_REDIS_URL="redis://localhost:6380/1"
    uv run pytest tests/integration/ -v

- name: Cleanup test services
  run: |
    cd tests/integration
    docker-compose -f docker-compose.test.yml down -v
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify PostgreSQL is running on the correct port
   - Check credentials and database name
   - Ensure `uuid-ossp` extension is available

2. **Redis Connection Errors**
   - Verify Redis is running on the correct port
   - Check Redis memory configuration
   - Ensure proper database separation (different db numbers)

3. **Test Timeouts**
   - Increase timeout for slow systems
   - Check async/await patterns in tests
   - Verify proper test cleanup

4. **Permission Errors**
   - Ensure database user has proper permissions
   - Check file system permissions for test data
   - Verify Docker container access rights

### Performance Issues
- Tests are designed to be fast but thorough
- Use `-m "not slow"` to skip performance benchmarks during development
- Adjust batch sizes and timeouts for slower systems in test configuration

## Contributing

When adding new integration tests:

1. **Follow Existing Patterns**
   - Use provided fixtures for consistency
   - Include proper cleanup in test methods
   - Add appropriate test markers (`@pytest.mark.asyncio`, etc.)

2. **Test Real Integration Points**
   - Focus on component interactions, not unit-level logic
   - Test actual database and cache behavior
   - Verify end-to-end scenarios

3. **Include Performance Considerations**
   - Add resource usage assertions where relevant
   - Test error conditions and recovery
   - Verify proper cleanup and resource management

4. **Update Documentation**
   - Update this README for new test categories
   - Document any new prerequisites or setup steps
   - Include troubleshooting information for new failure modes
