#!/bin/bash
# Test execution script for CI/CD and local development
# Usage: ./scripts/run-tests.sh [unit|integration|all|coverage|performance]

set -e

# Configuration
PYTHON_VERSION="3.11"
TEST_TIMEOUT=300  # 5 minutes
COVERAGE_THRESHOLD=75

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v uv &> /dev/null; then
        log_error "uv is required but not installed"
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi

    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ "$PYTHON_VER" != "3.11" ]]; then
        log_warning "Expected Python 3.11, found $PYTHON_VER"
    fi

    log_success "Dependencies check passed"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    uv sync --dev
    log_success "Dependencies installed"
}

# Run pre-commit checks
run_pre_commit() {
    log_info "Running pre-commit checks..."
    if uv run pre-commit run --all-files; then
        log_success "Pre-commit checks passed"
    else
        log_error "Pre-commit checks failed"
        exit 1
    fi
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."

    START_TIME=$(date +%s)

    if uv run pytest tests/unit/ \
        --tb=short \
        --durations=10 \
        --maxfail=10 \
        -v \
        --timeout=$TEST_TIMEOUT; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        log_success "Unit tests passed in ${DURATION}s"

        if [ $DURATION -gt $TEST_TIMEOUT ]; then
            log_warning "Unit tests took ${DURATION}s (exceeds ${TEST_TIMEOUT}s limit)"
        fi
    else
        log_error "Unit tests failed"
        exit 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."

    # Check if required services are available
    if ! nc -z localhost 5432; then
        log_warning "PostgreSQL not available on localhost:5432"
    fi

    if ! nc -z localhost 6379; then
        log_warning "Redis not available on localhost:6379"
    fi

    START_TIME=$(date +%s)

    if uv run pytest tests/integration/ \
        --tb=short \
        --durations=10 \
        --maxfail=5 \
        -v \
        -m "not slow" \
        --timeout=$TEST_TIMEOUT; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        log_success "Integration tests passed in ${DURATION}s"
    else
        log_error "Integration tests failed"
        exit 1
    fi
}

# Run coverage analysis
run_coverage() {
    log_info "Running coverage analysis..."

    uv run pytest tests/unit/ \
        --cov=services \
        --cov-report=html \
        --cov-report=term \
        --cov-report=xml \
        --cov-fail-under=$COVERAGE_THRESHOLD

    if [ $? -eq 0 ]; then
        log_success "Coverage analysis passed (≥${COVERAGE_THRESHOLD}%)"
        log_info "Coverage report saved to htmlcov/index.html"
    else
        log_error "Coverage analysis failed (below ${COVERAGE_THRESHOLD}% threshold)"
        exit 1
    fi
}

# Run performance tests
run_performance_tests() {
    log_info "Running performance benchmarks..."

    if uv run pytest tests/integration/test_performance_benchmarks.py \
        --benchmark-only \
        --benchmark-json=benchmark_results.json \
        -v; then

        log_success "Performance benchmarks completed"
        log_info "Results saved to benchmark_results.json"
    else
        log_error "Performance benchmarks failed"
        exit 1
    fi
}

# Run flaky test detection
run_flaky_detection() {
    log_info "Running flaky test detection (3 iterations)..."

    for i in {1..3}; do
        log_info "Flaky detection run $i/3"
        if ! uv run pytest tests/unit/analysis_service/api/ -q --tb=no; then
            log_error "Tests failed on run $i - possible flaky test detected"
            exit 1
        fi
    done

    log_success "No flaky tests detected"
}

# Generate test report
generate_report() {
    log_info "Generating test report..."

    REPORT_FILE="test_report.md"

    cat > $REPORT_FILE << EOF
# Test Execution Report

**Date:** $(date '+%Y-%m-%d %H:%M:%S')
**Environment:** $(uname -s) $(uname -r)
**Python Version:** $(python3 --version)

## Test Results Summary

EOF

    # Add test counts
    TEST_COUNT=$(uv run pytest tests/ --collect-only -q 2>/dev/null | grep -c "^tests/" || echo "Unknown")
    echo "**Total Tests:** $TEST_COUNT" >> $REPORT_FILE
    echo "" >> $REPORT_FILE

    # Add coverage if available
    if [ -f coverage.xml ]; then
        echo "**Coverage Report:** See htmlcov/index.html" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
    fi

    log_success "Test report generated: $REPORT_FILE"
}

# Main execution
main() {
    local command=${1:-all}

    log_info "Starting test execution: $command"

    case $command in
        "unit")
            check_dependencies
            install_dependencies
            run_pre_commit
            run_unit_tests
            ;;
        "integration")
            check_dependencies
            install_dependencies
            run_pre_commit
            run_integration_tests
            ;;
        "coverage")
            check_dependencies
            install_dependencies
            run_pre_commit
            run_coverage
            ;;
        "performance")
            check_dependencies
            install_dependencies
            run_performance_tests
            ;;
        "flaky")
            check_dependencies
            install_dependencies
            run_flaky_detection
            ;;
        "all")
            check_dependencies
            install_dependencies
            run_pre_commit
            run_unit_tests
            run_integration_tests
            run_flaky_detection
            generate_report
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Usage: $0 [unit|integration|all|coverage|performance|flaky]"
            exit 1
            ;;
    esac

    log_success "Test execution completed successfully"
}

# Execute main function
main "$@"
