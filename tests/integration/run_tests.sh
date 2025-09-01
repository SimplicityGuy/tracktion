#!/bin/bash

# Integration Test Runner for Feedback System
# This script sets up the test environment and runs integration tests

set -e

echo "🚀 Starting Feedback System Integration Tests"
echo "============================================="

# Check if required services are available
echo "📋 Checking prerequisites..."

# Check PostgreSQL availability
if ! command -v psql &> /dev/null && ! command -v docker &> /dev/null; then
    echo "❌ Neither PostgreSQL nor Docker found. Please install one of them."
    exit 1
fi

# Check Redis availability
if ! command -v redis-cli &> /dev/null && ! command -v docker &> /dev/null; then
    echo "❌ Neither Redis nor Docker found. Please install one of them."
    exit 1
fi

# Set test environment variables
export TEST_POSTGRES_DSN="${TEST_POSTGRES_DSN:-postgresql://tracktion_user:changeme@localhost:5433/test_feedback}"
export TEST_REDIS_URL="${TEST_REDIS_URL:-redis://localhost:6380/1}"

echo "📊 Test Environment:"
echo "  PostgreSQL: $TEST_POSTGRES_DSN"
echo "  Redis: $TEST_REDIS_URL"
echo ""

# Function to start test services using Docker
start_docker_services() {
    echo "🐳 Starting test services with Docker Compose..."

    if [ -f "docker-compose.test.yml" ]; then
        docker-compose -f docker-compose.test.yml up -d

        echo "⏳ Waiting for services to be ready..."
        timeout 60 bash -c 'until docker-compose -f docker-compose.test.yml exec postgres-test pg_isready -U tracktion_user -d test_feedback; do sleep 2; done'
        timeout 30 bash -c 'until docker-compose -f docker-compose.test.yml exec redis-test redis-cli ping | grep PONG; do sleep 1; done'

        echo "✅ Test services are ready"
    else
        echo "❌ docker-compose.test.yml not found"
        exit 1
    fi
}

# Function to stop test services
stop_docker_services() {
    echo "🛑 Stopping test services..."
    if [ -f "docker-compose.test.yml" ]; then
        docker-compose -f docker-compose.test.yml down -v
        echo "✅ Test services stopped and cleaned up"
    fi
}

# Function to run tests
run_tests() {
    echo "🧪 Running Integration Tests..."
    echo ""

    # Run different test categories
    echo "📝 Running End-to-End Flow Tests..."
    uv run pytest file_rename_service/test_feedback_integration.py::TestEndToEndFeedbackFlow -v

    echo ""
    echo "🔐 Running API Endpoint Tests..."
    uv run pytest file_rename_service/test_feedback_integration.py::TestAPIEndpointsWithAuthentication -v

    echo ""
    echo "💾 Running Database Transaction Tests..."
    uv run pytest file_rename_service/test_feedback_integration.py::TestDatabaseTransactionsAndRollback -v

    echo ""
    echo "🔄 Running Cache Consistency Tests..."
    uv run pytest file_rename_service/test_feedback_integration.py::TestCacheConsistency -v

    echo ""
    echo "🧪 Running A/B Testing Integration Tests..."
    uv run pytest file_rename_service/test_feedback_integration.py::TestABTestingIntegration -v

    echo ""
    echo "⚡ Running Resource Management Tests..."
    uv run pytest file_rename_service/test_feedback_integration.py::TestResourceManagementUnderLoad -v

    echo ""
    echo "🏃 Running Performance Tests (if --perf flag provided)..."
    if [[ "$*" == *"--perf"* ]]; then
        uv run pytest file_rename_service/test_feedback_integration.py::TestPerformanceUnderLoad -v -s
    else
        echo "   Skipped (use --perf to run performance tests)"
    fi
}

# Function to run tests with coverage
run_tests_with_coverage() {
    echo "📊 Running Tests with Coverage..."
    uv run pytest file_rename_service/ \
        --cov=services.file_rename_service.app.feedback \
        --cov-report=html \
        --cov-report=term \
        -v

    echo ""
    echo "📋 Coverage report generated in htmlcov/"
}

# Parse command line arguments
DOCKER_SERVICES=false
WITH_COVERAGE=false
RUN_PERFORMANCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            DOCKER_SERVICES=true
            shift
            ;;
        --coverage)
            WITH_COVERAGE=true
            shift
            ;;
        --perf)
            RUN_PERFORMANCE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker      Start PostgreSQL and Redis using Docker Compose"
            echo "  --coverage    Run tests with coverage reporting"
            echo "  --perf        Include performance/stress tests"
            echo "  --help        Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  TEST_POSTGRES_DSN    PostgreSQL connection string"
            echo "  TEST_REDIS_URL       Redis connection URL"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to integration test directory
cd "$(dirname "$0")"

# Set up cleanup trap
cleanup() {
    if [ "$DOCKER_SERVICES" = true ]; then
        stop_docker_services
    fi
}
trap cleanup EXIT

# Start Docker services if requested
if [ "$DOCKER_SERVICES" = true ]; then
    start_docker_services
fi

# Run tests
echo ""
if [ "$WITH_COVERAGE" = true ]; then
    run_tests_with_coverage
else
    run_tests $@
fi

echo ""
echo "🎉 Integration tests completed successfully!"
echo ""

# Show summary
echo "📈 Test Summary:"
echo "  ✅ End-to-End Flow Tests"
echo "  ✅ API Endpoint Tests"
echo "  ✅ Database Transaction Tests"
echo "  ✅ Cache Consistency Tests"
echo "  ✅ A/B Testing Integration Tests"
echo "  ✅ Resource Management Tests"
if [[ "$*" == *"--perf"* ]]; then
    echo "  ✅ Performance Tests"
else
    echo "  ⏭️  Performance Tests (skipped)"
fi

echo ""
echo "🔧 Next steps:"
echo "  • Review test results and logs"
echo "  • Check coverage report (if --coverage was used)"
echo "  • Run with --perf for performance benchmarks"
echo "  • Integrate into your CI/CD pipeline"
