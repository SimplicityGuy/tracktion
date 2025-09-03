# CI/CD Test Integration Setup Guide

## Overview

This document provides instructions for setting up CI/CD test integration for the Tracktion project using the comprehensive test suite developed in Story 11.4.

## Prerequisites

- GitHub repository with Actions enabled
- Access to configure GitHub secrets
- Docker support for service containers
- Slack webhook for notifications (optional)

## Quick Start

### 1. GitHub Actions Setup

The project includes two main workflows:

- **`.github/workflows/test-suite.yml`** - Main test execution pipeline
- **`.github/workflows/test-quality-gates.yml`** - Quality gates for pull requests

These workflows are configured to run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Daily scheduled runs at 2 AM UTC

### 2. Required GitHub Secrets

Configure the following secrets in your GitHub repository:

```bash
# Optional: Slack notifications
SLACK_WEBHOOK=https://hooks.slack.com/services/...

# Optional: Codecov integration
CODECOV_TOKEN=your-codecov-token
```

### 3. Local Test Execution

Use the provided test script for local development:

```bash
# Run all tests
./scripts/run-tests.sh all

# Run only unit tests
./scripts/run-tests.sh unit

# Run with coverage
./scripts/run-tests.sh coverage

# Performance benchmarks
./scripts/run-tests.sh performance
```

## Workflow Details

### Main Test Pipeline (`test-suite.yml`)

#### Jobs Overview:

1. **Lint and Format**
   - Runs pre-commit hooks (ruff, mypy, formatting)
   - Blocks pipeline if code quality checks fail

2. **Unit Tests (Matrix Strategy)**
   - Parallel execution across services: `analysis_service`, `tracklist_service`, `cataloging_service`, `file_watcher`
   - Coverage reporting with 80% threshold
   - Upload results to Codecov

3. **Integration Tests**
   - Runs with service containers (PostgreSQL, Redis, RabbitMQ)
   - Tests service interactions and workflows
   - Comprehensive system integration validation

4. **Performance Tests**
   - Benchmark execution for main branch pushes
   - Performance regression detection
   - Results stored for trend analysis

5. **Flaky Test Detection**
   - Multi-run execution on pull requests
   - Identifies non-deterministic test behavior
   - Prevents flaky tests from merging

6. **Coverage Reporting**
   - Combined coverage from all test runs
   - Service-specific coverage metrics
   - Historical trend tracking

### Quality Gates (`test-quality-gates.yml`)

Enforces quality standards on pull requests:

- **Coverage Threshold**: Minimum 75% coverage required
- **Execution Time**: Tests must complete within 5 minutes
- **Flakiness Detection**: Tests run 3 times to detect instability
- **Structure Validation**: Validates test file conventions

## Test Categories and Markers

The test suite uses pytest markers for organization:

```python
# Pytest markers in pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests (>1s execution time)",
    "performance: Performance benchmark tests",
    "requires_db: Tests requiring database connection",
    "requires_redis: Tests requiring Redis connection",
    "requires_external: Tests requiring external services"
]
```

### Running Specific Test Categories:

```bash
# Unit tests only
uv run pytest -m "unit"

# Integration tests excluding slow ones
uv run pytest -m "integration and not slow"

# Tests requiring database
uv run pytest -m "requires_db"

# Performance benchmarks
uv run pytest -m "performance" --benchmark-only
```

## Service Containers Configuration

Integration tests use Docker containers for external services:

### PostgreSQL
- **Image**: `postgres:15`
- **Port**: 5432
- **Credentials**: testuser/testpass
- **Database**: testdb

### Redis
- **Image**: `redis:7-alpine`
- **Port**: 6379
- **Health Check**: `redis-cli ping`

### RabbitMQ
- **Image**: `rabbitmq:3-management-alpine`
- **Ports**: 5672 (AMQP), 15672 (Management)
- **Credentials**: testuser/testpass

## Coverage Configuration

### Coverage Targets:
- **Overall Project**: 75% minimum (CI enforcement)
- **Critical API Modules**: 80% minimum
- **Error Handlers**: 100% (already achieved)
- **Core Business Logic**: 90% target

### Coverage Reporting:
- **HTML Report**: `htmlcov/index.html`
- **XML Report**: `coverage.xml` (for Codecov)
- **Terminal Summary**: Displayed in CI logs

## Performance Monitoring

### Benchmark Tracking:
- **Tool**: pytest-benchmark
- **Storage**: GitHub Pages (benchmark-action)
- **Metrics**: Execution time, memory usage, throughput
- **Regression Detection**: Automatic alerts for performance degradation

### Performance Targets:
- **Unit Tests**: <100ms each
- **Integration Tests**: <1s each
- **Full Test Suite**: <5 minutes
- **CI Pipeline**: <10 minutes total

## Flaky Test Management

### Detection Strategy:
1. **Multi-run Execution**: Tests run 3 times on PRs
2. **Timing Analysis**: Execution time variance tracking
3. **Failure Pattern Recognition**: Statistical analysis of failures
4. **Deterministic Validation**: Random seed fixing verification

### Prevention Measures:
- **Fixed Random Seeds**: All tests use seed=42
- **Mock External Services**: No real network dependencies
- **Controlled Timing**: Event-driven synchronization instead of sleeps
- **Resource Isolation**: Each test gets clean state

## Troubleshooting

### Common CI Issues:

1. **Test Timeout**
   ```bash
   # Increase timeout in workflow
   timeout-minutes: 15  # Default: 10
   ```

2. **Service Connection Failures**
   ```bash
   # Add service health check wait
   - name: Wait for services
     run: |
       sleep 10
       ./scripts/wait-for-services.sh
   ```

3. **Coverage Failures**
   ```bash
   # Check coverage threshold
   --cov-fail-under=75
   ```

4. **Memory Issues**
   ```bash
   # Reduce parallel test execution
   pytest -n 2  # Instead of -n auto
   ```

### Debugging Failed Tests:

```bash
# Verbose output
uv run pytest -vvs tests/path/to/test.py

# Debug specific test
uv run pytest --pdb tests/path/to/test.py::test_name

# Show test duration
uv run pytest --durations=10 tests/
```

## Maintenance

### Regular Tasks:

1. **Weekly**: Review flaky test reports
2. **Monthly**: Update dependency versions
3. **Quarterly**: Review performance trends
4. **Annually**: Update service container versions

### Metrics to Monitor:

- **Test Execution Time**: Should remain under 5 minutes
- **Coverage Percentage**: Should not decrease below thresholds
- **Flaky Test Count**: Should remain at zero
- **Pipeline Success Rate**: Should exceed 95%

## Integration with Development Workflow

### Pre-commit Hooks:
- **Automatic**: Runs on every commit
- **Fast**: Completes in <30 seconds
- **Comprehensive**: Includes linting, formatting, type checking

### Pull Request Process:
1. **Quality Gates**: Automatic validation on PR creation
2. **Coverage Report**: Posted as PR comment
3. **Performance Impact**: Benchmark comparison with main branch
4. **Flaky Test Check**: Multi-run validation

### Deployment Pipeline:
1. **All Tests Pass**: Required for merge to main
2. **Coverage Maintained**: No decrease in coverage percentage
3. **Performance Validated**: No significant regression detected
4. **Quality Gates**: All CI checks must pass

## Support and Resources

### Documentation:
- **Test Structure Standards**: `docs/test-structure-standards.md`
- **Shared Utilities Guide**: `tests/shared_utilities/README.md`
- **Skip Test Documentation**: `docs/test_skip_documentation.md`

### Monitoring:
- **GitHub Actions**: Repository Actions tab
- **Coverage Reports**: Codecov dashboard
- **Performance Trends**: GitHub Pages benchmark site
- **Quality Metrics**: CI job summaries

For additional support, refer to the comprehensive test suite documentation created in Story 11.4 or contact the development team.
