"""
Test debugging helpers and utilities for troubleshooting test failures.

Provides structured debugging tools for identifying test issues, analyzing failures,
and monitoring test execution patterns.
"""

import contextlib
import functools
import inspect
import logging
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class TestExecutionMetrics:
    """Tracks metrics for individual test execution."""

    test_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    memory_usage: int = 0
    peak_memory: int = 0
    exception_count: int = 0
    warning_count: int = 0
    assertion_count: int = 0
    fixture_setup_time: float = 0.0
    fixture_teardown_time: float = 0.0


@dataclass
class DebugContext:
    """Context information for debugging test failures."""

    test_name: str
    test_file: str
    test_line: int
    failure_type: str
    failure_message: str
    stack_trace: str
    local_variables: dict[str, Any] = field(default_factory=dict)
    fixture_state: dict[str, Any] = field(default_factory=dict)
    execution_metrics: TestExecutionMetrics | None = None
    related_tests: list[str] = field(default_factory=list)


class TestDebugger:
    """Enhanced debugging helper for test execution and failure analysis."""

    def __init__(self, enabled: bool = True, log_level: str = "DEBUG"):
        self.enabled = enabled
        self.execution_history: list[TestExecutionMetrics] = []
        self.failure_contexts: list[DebugContext] = []
        self.test_dependencies: dict[str, set[str]] = defaultdict(set)
        self.performance_warnings: list[str] = []
        self.flaky_test_indicators: dict[str, int] = defaultdict(int)

        # Configure logging
        self.logger = logging.getLogger(f"test_debugger.{id(self)}")
        self.logger.setLevel(getattr(logging, log_level))
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s [TEST-DEBUG] %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def capture_test_metrics(self, test_name: str) -> TestExecutionMetrics:
        """Create and track metrics for a test execution."""
        metrics = TestExecutionMetrics(test_name=test_name)
        metrics.start_time = time.time()

        if self.enabled:
            self.logger.debug(f"Starting metrics capture for: {test_name}")

        return metrics

    def finalize_metrics(self, metrics: TestExecutionMetrics) -> None:
        """Finalize metrics collection and analysis."""
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time

        # Performance warning for slow tests
        if metrics.duration > 1.0:
            warning = f"Slow test detected: {metrics.test_name} ({metrics.duration:.2f}s)"
            self.performance_warnings.append(warning)
            if self.enabled:
                self.logger.warning(warning)

        self.execution_history.append(metrics)

        if self.enabled:
            self.logger.debug(f"Test {metrics.test_name} completed in {metrics.duration:.3f}s")

    def capture_failure_context(
        self, test_name: str, exception: Exception, test_locals: dict[str, Any] | None = None
    ) -> DebugContext:
        """Capture comprehensive context for test failure analysis."""

        # Extract test location information
        frame = inspect.currentframe()
        test_file = "unknown"
        test_line = 0

        try:
            # Walk up the stack to find the test function
            while frame:
                if frame.f_code.co_name.startswith("test_"):
                    test_file = frame.f_code.co_filename
                    test_line = frame.f_lineno
                    break
                frame = frame.f_back
        except Exception:
            pass  # Fallback to default values

        # Create failure context
        context = DebugContext(
            test_name=test_name,
            test_file=test_file,
            test_line=test_line,
            failure_type=type(exception).__name__,
            failure_message=str(exception),
            stack_trace=traceback.format_exc(),
            local_variables=test_locals or {},
        )

        self.failure_contexts.append(context)

        # Track flaky test indicators
        self.flaky_test_indicators[test_name] += 1
        if self.flaky_test_indicators[test_name] > 1 and self.enabled:
            self.logger.warning(
                f"Potential flaky test detected: {test_name} (failed {self.flaky_test_indicators[test_name]} times)"
            )

        if self.enabled:
            self.logger.error(f"Test failure captured: {test_name} - {context.failure_type}: {context.failure_message}")

        return context

    def analyze_test_dependencies(self, test_name: str, fixtures_used: list[str]) -> None:
        """Analyze and track test dependencies for debugging."""
        for fixture in fixtures_used:
            self.test_dependencies[test_name].add(fixture)

        if self.enabled:
            self.logger.debug(f"Test {test_name} depends on fixtures: {fixtures_used}")

    def generate_debug_report(self, output_path: Path | None = None) -> str:
        """Generate comprehensive debug report for test session."""

        report_lines = [
            "# Test Debug Report",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "## Execution Summary",
            f"Total tests tracked: {len(self.execution_history)}",
            f"Total failures: {len(self.failure_contexts)}",
            f"Performance warnings: {len(self.performance_warnings)}",
            f"Potential flaky tests: {sum(1 for count in self.flaky_test_indicators.values() if count > 1)}",
            "",
        ]

        # Performance analysis
        if self.execution_history:
            durations = [m.duration for m in self.execution_history]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)

            report_lines.extend(
                [
                    "## Performance Analysis",
                    f"Average test duration: {avg_duration:.3f}s",
                    f"Slowest test duration: {max_duration:.3f}s",
                    f"Tests over 1s: {sum(1 for d in durations if d > 1.0)}",
                    "",
                ]
            )

        # Failure analysis
        if self.failure_contexts:
            report_lines.extend(["## Failure Analysis", ""])

            failure_types = defaultdict(int)
            for context in self.failure_contexts:
                failure_types[context.failure_type] += 1

            for failure_type, count in sorted(failure_types.items()):
                report_lines.append(f"- {failure_type}: {count} occurrences")

            report_lines.extend(["", "### Recent Failures", ""])

            for context in self.failure_contexts[-5:]:  # Last 5 failures
                report_lines.extend(
                    [
                        f"**{context.test_name}**",
                        f"- Type: {context.failure_type}",
                        f"- Message: {context.failure_message}",
                        f"- File: {Path(context.test_file).name}:{context.test_line}",
                        "",
                    ]
                )

        # Flaky test detection
        flaky_tests = {name: count for name, count in self.flaky_test_indicators.items() if count > 1}

        if flaky_tests:
            report_lines.extend(["## Flaky Test Detection", ""])

            for test_name, failure_count in sorted(flaky_tests.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"- {test_name}: {failure_count} failures")

            report_lines.append("")

        # Performance warnings
        if self.performance_warnings:
            report_lines.extend(["## Performance Warnings", ""])

            report_lines.extend(f"- {warning}" for warning in self.performance_warnings[-10:])  # Last 10

            report_lines.append("")

        # Dependency analysis
        if self.test_dependencies:
            report_lines.extend(["## Test Dependencies", ""])

            for test_name, fixtures in sorted(self.test_dependencies.items()):
                if fixtures:
                    fixture_list = ", ".join(sorted(fixtures))
                    report_lines.append(f"- {test_name}: {fixture_list}")

            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            output_path.write_text(report_content, encoding="utf-8")
            if self.enabled:
                self.logger.info(f"Debug report saved to: {output_path}")

        return report_content

    def clear_history(self) -> None:
        """Clear all collected debugging data."""
        self.execution_history.clear()
        self.failure_contexts.clear()
        self.test_dependencies.clear()
        self.performance_warnings.clear()
        self.flaky_test_indicators.clear()

        if self.enabled:
            self.logger.info("Debug history cleared")


# Global debugger instance
test_debugger = TestDebugger()


def debug_test(capture_locals: bool = True, track_fixtures: bool = True, performance_threshold: float = 1.0):
    """
    Decorator to add comprehensive debugging to test functions.

    Args:
        capture_locals: Whether to capture local variables on failure
        track_fixtures: Whether to track fixture dependencies
        performance_threshold: Threshold in seconds for performance warnings

    Example:
        @debug_test(capture_locals=True)
        def test_my_function():
            assert my_function() == expected_result
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            test_name = f"{test_func.__module__}.{test_func.__name__}"
            metrics = test_debugger.capture_test_metrics(test_name)

            # Track fixture usage
            if track_fixtures:
                fixtures_used = list(kwargs.keys())
                test_debugger.analyze_test_dependencies(test_name, fixtures_used)

            try:
                # Execute the test
                result = test_func(*args, **kwargs)

                # Finalize metrics
                test_debugger.finalize_metrics(metrics)

                return result

            except Exception as e:
                # Capture failure context
                local_vars = {}
                if capture_locals:
                    frame = inspect.currentframe()
                    try:
                        if frame and frame.f_back:
                            local_vars = {
                                k: str(v)[:100] for k, v in frame.f_back.f_locals.items() if not k.startswith("_")
                            }
                    except Exception:
                        pass

                test_debugger.capture_failure_context(test_name, e, local_vars)
                test_debugger.finalize_metrics(metrics)

                # Re-raise the original exception
                raise

        return wrapper

    return decorator


@contextlib.contextmanager
def debug_context(test_name: str):
    """
    Context manager for debugging test execution.

    Example:
        def test_something():
            with debug_context("test_something"):
                # Test code here
                assert result == expected
    """
    metrics = test_debugger.capture_test_metrics(test_name)

    try:
        yield test_debugger
        test_debugger.finalize_metrics(metrics)

    except Exception as e:
        test_debugger.capture_failure_context(test_name, e)
        test_debugger.finalize_metrics(metrics)
        raise


def assert_with_debug(condition: bool, message: str = "", debug_data: dict[str, Any] | None = None):
    """
    Enhanced assertion with debugging information.

    Args:
        condition: The condition to assert
        message: Custom error message
        debug_data: Additional data to include in failure context

    Example:
        assert_with_debug(
            result == expected,
            "Function returned unexpected value",
            {"result": result, "expected": expected, "input": input_data}
        )
    """
    if not condition:
        # Prepare enhanced error message
        error_msg = f"Assertion failed: {message}" if message else "Assertion failed"

        if debug_data:
            debug_info = "\nDebug information:"
            for key, value in debug_data.items():
                debug_info += f"\n  {key}: {value!r}"
            error_msg += debug_info

        # Create and raise AssertionError with enhanced context
        raise AssertionError(error_msg)


class AsyncTestDebugger:
    """Specialized debugging for async test functions."""

    @staticmethod
    async def debug_async_test(coro, test_name: str):
        """Debug wrapper for async test execution."""
        metrics = test_debugger.capture_test_metrics(test_name)

        try:
            result = await coro
            test_debugger.finalize_metrics(metrics)
            return result

        except Exception as e:
            test_debugger.capture_failure_context(test_name, e)
            test_debugger.finalize_metrics(metrics)
            raise

    @staticmethod
    def debug_async(func):
        """Decorator for async test functions."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            test_name = f"{func.__module__}.{func.__name__}"
            return await AsyncTestDebugger.debug_async_test(func(*args, **kwargs), test_name)

        return wrapper


# Pytest integration
def pytest_runtest_setup(item):
    """Pytest hook to initialize debugging for each test."""
    if hasattr(item, "function"):
        test_name = f"{item.module.__name__}.{item.function.__name__}"
        fixtures_used = list(item.fixturenames)
        test_debugger.analyze_test_dependencies(test_name, fixtures_used)


def pytest_runtest_call(item):
    """Pytest hook to track test execution."""
    test_name = f"{item.module.__name__}.{item.function.__name__}"
    metrics = test_debugger.capture_test_metrics(test_name)

    # Store metrics in item for later finalization
    item._debug_metrics = metrics


def pytest_runtest_teardown(item):
    """Pytest hook to finalize test debugging."""
    if hasattr(item, "_debug_metrics"):
        test_debugger.finalize_metrics(item._debug_metrics)


def pytest_sessionfinish(session):
    """Generate debug report at end of test session."""
    if test_debugger.enabled and (test_debugger.failure_contexts or test_debugger.performance_warnings):
        report_path = Path("test_debug_report.md")
        test_debugger.generate_debug_report(report_path)
        print(f"\n[DEBUG] Test debug report generated: {report_path}")
