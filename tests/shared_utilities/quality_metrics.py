"""
Test quality metrics collection and analysis for test suite assessment.

Provides comprehensive metrics collection, analysis, and reporting capabilities
to measure test suite health, performance, and quality indicators.
"""

import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class TestMetrics:
    """Individual test execution metrics."""

    test_id: str
    test_name: str
    test_file: str
    duration: float
    status: str  # passed, failed, skipped, error
    memory_peak: int | None = None
    setup_time: float = 0.0
    teardown_time: float = 0.0
    fixture_count: int = 0
    assertion_count: int = 0
    lines_of_code: int = 0
    cyclomatic_complexity: int = 1
    execution_count: int = 1
    failure_count: int = 0
    skip_reason: str | None = None
    error_type: str | None = None
    markers: list[str] = None

    def __post_init__(self):
        if self.markers is None:
            self.markers = []


@dataclass
class TestSuiteMetrics:
    """Aggregate metrics for the entire test suite."""

    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0

    total_duration: float = 0.0
    average_duration: float = 0.0
    median_duration: float = 0.0
    slowest_duration: float = 0.0
    fastest_duration: float = 0.0

    coverage_percentage: float | None = None
    lines_covered: int = 0
    lines_total: int = 0

    flaky_tests: list[str] = None
    slow_tests: list[str] = None  # Tests over performance threshold
    performance_threshold: float = 1.0

    test_density: float = 0.0  # Tests per line of production code
    assertion_density: float = 0.0  # Assertions per test
    fixture_utilization: dict[str, int] = None

    quality_score: float = 0.0  # Composite quality score (0-100)
    reliability_score: float = 0.0  # Based on flakiness and failures
    performance_score: float = 0.0  # Based on execution times
    coverage_score: float = 0.0  # Based on code coverage

    def __post_init__(self):
        if self.flaky_tests is None:
            self.flaky_tests = []
        if self.slow_tests is None:
            self.slow_tests = []
        if self.fixture_utilization is None:
            self.fixture_utilization = {}


@dataclass
class TestTrendData:
    """Historical trend data for test metrics."""

    timestamp: datetime
    suite_metrics: TestSuiteMetrics
    individual_tests: list[TestMetrics]
    commit_hash: str | None = None
    branch: str | None = None
    environment: str | None = None


class TestQualityCollector:
    """Collects and analyzes test quality metrics."""

    def __init__(self, performance_threshold: float = 1.0):
        self.performance_threshold = performance_threshold
        self.test_metrics: dict[str, TestMetrics] = {}
        self.execution_history: list[TestTrendData] = []
        self.fixture_usage: dict[str, int] = defaultdict(int)
        self.test_dependencies: dict[str, set[str]] = defaultdict(set)

        # Quality thresholds
        self.quality_thresholds = {
            "min_coverage": 80.0,
            "max_failure_rate": 5.0,
            "max_skip_rate": 10.0,
            "max_flaky_tests": 5,
            "max_slow_tests": 10,
            "min_assertion_density": 2.0,
            "max_avg_duration": 0.1,
        }

    def record_test_execution(
        self, test_id: str, test_name: str, test_file: str, duration: float, status: str, **kwargs
    ) -> None:
        """Record metrics for a single test execution."""

        if test_id in self.test_metrics:
            # Update existing metrics
            metrics = self.test_metrics[test_id]
            metrics.execution_count += 1
            metrics.duration = (metrics.duration + duration) / 2  # Running average

            if status == "failed":
                metrics.failure_count += 1

        else:
            # Create new metrics
            metrics = TestMetrics(
                test_id=test_id, test_name=test_name, test_file=test_file, duration=duration, status=status, **kwargs
            )

            if status == "failed":
                metrics.failure_count = 1

        self.test_metrics[test_id] = metrics

    def record_fixture_usage(self, test_id: str, fixtures: list[str]) -> None:
        """Record fixture usage for dependency analysis."""

        for fixture in fixtures:
            self.fixture_usage[fixture] += 1
            self.test_dependencies[test_id].add(fixture)

        if test_id in self.test_metrics:
            self.test_metrics[test_id].fixture_count = len(fixtures)

    def calculate_suite_metrics(self, coverage_data: dict | None = None) -> TestSuiteMetrics:
        """Calculate comprehensive suite-level metrics."""

        if not self.test_metrics:
            return TestSuiteMetrics()

        # Basic counts
        total_tests = len(self.test_metrics)
        passed_tests = sum(1 for m in self.test_metrics.values() if m.status == "passed")
        failed_tests = sum(1 for m in self.test_metrics.values() if m.status == "failed")
        skipped_tests = sum(1 for m in self.test_metrics.values() if m.status == "skipped")
        error_tests = sum(1 for m in self.test_metrics.values() if m.status == "error")

        # Duration statistics
        durations = [m.duration for m in self.test_metrics.values()]
        total_duration = sum(durations)
        average_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        slowest_duration = max(durations)
        fastest_duration = min(durations)

        # Identify problematic tests
        flaky_tests = [
            m.test_id
            for m in self.test_metrics.values()
            if m.execution_count > 1 and m.failure_count > 0 and m.status == "passed"
        ]

        slow_tests = [m.test_id for m in self.test_metrics.values() if m.duration > self.performance_threshold]

        # Coverage data
        coverage_percentage = None
        lines_covered = 0
        lines_total = 0

        if coverage_data:
            coverage_percentage = coverage_data.get("coverage_percent", 0.0)
            lines_covered = coverage_data.get("lines_covered", 0)
            lines_total = coverage_data.get("lines_total", 0)

        # Calculate density metrics
        total_assertions = sum(m.assertion_count for m in self.test_metrics.values())
        assertion_density = total_assertions / total_tests if total_tests > 0 else 0.0

        # Quality scores
        reliability_score = self._calculate_reliability_score(total_tests, failed_tests, len(flaky_tests))
        performance_score = self._calculate_performance_score(average_duration, len(slow_tests), total_tests)
        coverage_score = self._calculate_coverage_score(coverage_percentage)

        # Composite quality score
        quality_score = (reliability_score + performance_score + coverage_score) / 3

        return TestSuiteMetrics(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_duration=total_duration,
            average_duration=average_duration,
            median_duration=median_duration,
            slowest_duration=slowest_duration,
            fastest_duration=fastest_duration,
            coverage_percentage=coverage_percentage,
            lines_covered=lines_covered,
            lines_total=lines_total,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            performance_threshold=self.performance_threshold,
            assertion_density=assertion_density,
            fixture_utilization=dict(self.fixture_usage),
            quality_score=quality_score,
            reliability_score=reliability_score,
            performance_score=performance_score,
            coverage_score=coverage_score,
        )

    def _calculate_reliability_score(self, total: int, failed: int, flaky: int) -> float:
        """Calculate reliability score (0-100)."""
        if total == 0:
            return 100.0

        failure_rate = (failed / total) * 100
        flaky_rate = (flaky / total) * 100

        # Penalize failures and flakiness
        score = 100.0 - (failure_rate * 2) - (flaky_rate * 1.5)
        return max(0.0, min(100.0, score))

    def _calculate_performance_score(self, avg_duration: float, slow_count: int, total: int) -> float:
        """Calculate performance score (0-100)."""
        if total == 0:
            return 100.0

        # Score based on average duration and slow test percentage
        duration_score = max(0, 100 - (avg_duration * 100))  # Penalize slow averages
        slow_percentage = (slow_count / total) * 100
        slow_score = max(0, 100 - (slow_percentage * 2))  # Penalize slow tests

        return (duration_score + slow_score) / 2

    def _calculate_coverage_score(self, coverage: float | None) -> float:
        """Calculate coverage score (0-100)."""
        if coverage is None:
            return 50.0  # Neutral score for unknown coverage

        # Linear score based on coverage percentage
        return min(100.0, coverage * 1.25)  # Boost score for high coverage

    def analyze_trends(self, lookback_days: int = 30) -> dict[str, Any]:
        """Analyze trends over time."""

        cutoff_date = datetime.now(UTC) - timedelta(days=lookback_days)
        recent_data = [trend for trend in self.execution_history if trend.timestamp >= cutoff_date]

        if len(recent_data) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Calculate trend metrics
        quality_scores = [trend.suite_metrics.quality_score for trend in recent_data]
        durations = [trend.suite_metrics.average_duration for trend in recent_data]
        coverage_scores = [trend.suite_metrics.coverage_percentage or 0 for trend in recent_data]

        return {
            "data_points": len(recent_data),
            "quality_trend": {
                "current": quality_scores[-1],
                "previous": quality_scores[0],
                "change": quality_scores[-1] - quality_scores[0],
                "trend": "improving" if quality_scores[-1] > quality_scores[0] else "degrading",
            },
            "performance_trend": {
                "current": durations[-1],
                "previous": durations[0],
                "change": durations[-1] - durations[0],
                "trend": "improving" if durations[-1] < durations[0] else "degrading",
            },
            "coverage_trend": {
                "current": coverage_scores[-1],
                "previous": coverage_scores[0],
                "change": coverage_scores[-1] - coverage_scores[0],
                "trend": "improving" if coverage_scores[-1] > coverage_scores[0] else "degrading",
            },
        }

    def generate_quality_report(self, coverage_data: dict | None = None, output_path: Path | None = None) -> str:
        """Generate comprehensive quality report."""

        suite_metrics = self.calculate_suite_metrics(coverage_data)

        report_lines = [
            "# Test Quality Report",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "## Executive Summary",
            f"- **Quality Score**: {suite_metrics.quality_score:.1f}/100",
            f"- **Reliability Score**: {suite_metrics.reliability_score:.1f}/100",
            f"- **Performance Score**: {suite_metrics.performance_score:.1f}/100",
            f"- **Coverage Score**: {suite_metrics.coverage_score:.1f}/100",
            "",
            "## Test Execution Summary",
            f"- Total Tests: {suite_metrics.total_tests}",
            f"- Passed: {suite_metrics.passed_tests} "
            f"({suite_metrics.passed_tests / suite_metrics.total_tests * 100:.1f}%)",
            f"- Failed: {suite_metrics.failed_tests} "
            f"({suite_metrics.failed_tests / suite_metrics.total_tests * 100:.1f}%)",
            f"- Skipped: {suite_metrics.skipped_tests} "
            f"({suite_metrics.skipped_tests / suite_metrics.total_tests * 100:.1f}%)",
            f"- Errors: {suite_metrics.error_tests}",
            "",
            "## Performance Metrics",
            f"- Total Duration: {suite_metrics.total_duration:.2f}s",
            f"- Average Duration: {suite_metrics.average_duration:.3f}s",
            f"- Median Duration: {suite_metrics.median_duration:.3f}s",
            f"- Slowest Test: {suite_metrics.slowest_duration:.3f}s",
            f"- Performance Threshold: {suite_metrics.performance_threshold:.1f}s",
            f"- Slow Tests: {len(suite_metrics.slow_tests)}",
            "",
        ]

        # Coverage section
        if suite_metrics.coverage_percentage is not None:
            report_lines.extend(
                [
                    "## Code Coverage",
                    f"- Overall Coverage: {suite_metrics.coverage_percentage:.1f}%",
                    f"- Lines Covered: {suite_metrics.lines_covered:,}",
                    f"- Total Lines: {suite_metrics.lines_total:,}",
                    "",
                ]
            )

        # Quality indicators
        report_lines.extend(
            [
                "## Quality Indicators",
                f"- Flaky Tests: {len(suite_metrics.flaky_tests)}",
                f"- Assertion Density: {suite_metrics.assertion_density:.1f} per test",
                f"- Fixture Utilization: {len(suite_metrics.fixture_utilization)} fixtures used",
                "",
            ]
        )

        # Problem areas
        if suite_metrics.flaky_tests:
            report_lines.extend(["## ⚠️ Flaky Tests", ""])
            report_lines.extend(f"- {test_id}" for test_id in suite_metrics.flaky_tests[:10])  # Top 10
            report_lines.append("")

        if suite_metrics.slow_tests:
            report_lines.extend([f"## 🐌 Slow Tests (>{suite_metrics.performance_threshold:.1f}s)", ""])
            slow_tests_with_duration = [
                (test_id, self.test_metrics[test_id].duration)
                for test_id in suite_metrics.slow_tests
                if test_id in self.test_metrics
            ]
            slow_tests_with_duration.sort(key=lambda x: x[1], reverse=True)

            for test_id, duration in slow_tests_with_duration[:10]:  # Top 10
                report_lines.append(f"- {test_id}: {duration:.3f}s")
            report_lines.append("")

        # Quality recommendations
        recommendations = self._generate_recommendations(suite_metrics)
        if recommendations:
            report_lines.extend(["## 💡 Recommendations", ""])
            report_lines.extend(f"- {rec}" for rec in recommendations)
            report_lines.append("")

        # Top fixtures
        if suite_metrics.fixture_utilization:
            report_lines.extend(["## 🔧 Most Used Fixtures", ""])
            sorted_fixtures = sorted(suite_metrics.fixture_utilization.items(), key=lambda x: x[1], reverse=True)
            for fixture, count in sorted_fixtures[:10]:
                report_lines.append(f"- {fixture}: {count} tests")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            output_path.write_text(report_content, encoding="utf-8")

        return report_content

    def _generate_recommendations(self, metrics: TestSuiteMetrics) -> list[str]:
        """Generate actionable recommendations based on metrics."""

        recommendations = []

        # Coverage recommendations
        if metrics.coverage_percentage and metrics.coverage_percentage < 80:
            recommendations.append(f"Increase test coverage from {metrics.coverage_percentage:.1f}% to at least 80%")

        # Performance recommendations
        if len(metrics.slow_tests) > 10:
            recommendations.append(
                f"Optimize {len(metrics.slow_tests)} slow tests (target: <{metrics.performance_threshold:.1f}s)"
            )

        # Reliability recommendations
        if len(metrics.flaky_tests) > 0:
            recommendations.append(f"Fix {len(metrics.flaky_tests)} flaky tests for better reliability")

        # Failure rate recommendations
        if metrics.total_tests > 0:
            failure_rate = (metrics.failed_tests / metrics.total_tests) * 100
            if failure_rate > 5:
                recommendations.append(f"Reduce test failure rate from {failure_rate:.1f}% to <5%")

        # Assertion density recommendations
        if metrics.assertion_density < 2.0:
            recommendations.append(
                f"Increase assertion density from {metrics.assertion_density:.1f} to at least 2.0 per test"
            )

        # Skip rate recommendations
        if metrics.total_tests > 0:
            skip_rate = (metrics.skipped_tests / metrics.total_tests) * 100
            if skip_rate > 10:
                recommendations.append(f"Review {metrics.skipped_tests} skipped tests (skip rate: {skip_rate:.1f}%)")

        return recommendations

    def save_trend_data(self, suite_metrics: TestSuiteMetrics, file_path: Path, commit_hash: str | None = None) -> None:
        """Save trend data to file for historical analysis."""

        trend_data = TestTrendData(
            timestamp=datetime.now(UTC),
            suite_metrics=suite_metrics,
            individual_tests=list(self.test_metrics.values()),
            commit_hash=commit_hash,
        )

        self.execution_history.append(trend_data)

        # Save to JSON file
        trend_dict = asdict(trend_data)
        trend_dict["timestamp"] = trend_data.timestamp.isoformat()

        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        history.append(trend_dict)

        # Keep only last 100 entries
        history = history[-100:]

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.test_metrics.clear()
        self.fixture_usage.clear()
        self.test_dependencies.clear()


# Global quality collector instance
quality_collector = TestQualityCollector()


def pytest_runtest_setup(item):
    """Pytest hook to track fixture usage."""
    test_id = f"{item.module.__name__}::{item.function.__name__}"
    fixtures = list(item.fixturenames)
    quality_collector.record_fixture_usage(test_id, fixtures)


def pytest_runtest_call(item):
    """Pytest hook to start timing test execution."""
    item._start_time = time.time()


def pytest_runtest_teardown(item, nextitem):
    """Pytest hook to record test execution metrics."""
    if hasattr(item, "_start_time"):
        duration = time.time() - item._start_time

        test_id = f"{item.module.__name__}::{item.function.__name__}"
        test_name = item.function.__name__
        test_file = str(item.fspath)

        # Determine status from test outcome
        status = "passed"  # Default
        if hasattr(item, "rep_call"):
            if item.rep_call.failed:
                status = "failed"
            elif item.rep_call.skipped:
                status = "skipped"

        quality_collector.record_test_execution(
            test_id=test_id,
            test_name=test_name,
            test_file=test_file,
            duration=duration,
            status=status,
            markers=[marker.name for marker in item.iter_markers()],
        )


def pytest_sessionfinish(session):
    """Generate quality report at end of test session."""

    # Generate quality report
    report_path = Path("test_quality_report.md")
    quality_collector.generate_quality_report(output_path=report_path)

    # Save trend data
    suite_metrics = quality_collector.calculate_suite_metrics()
    trends_path = Path("test_trends.json")
    quality_collector.save_trend_data(suite_metrics, trends_path)

    print(f"\n[METRICS] Test quality report: {report_path}")
    print(f"[METRICS] Quality score: {suite_metrics.quality_score:.1f}/100")

    # Print warnings for quality issues
    if suite_metrics.flaky_tests:
        print(f"[WARNING] {len(suite_metrics.flaky_tests)} flaky tests detected")

    if len(suite_metrics.slow_tests) > 10:
        print(f"[WARNING] {len(suite_metrics.slow_tests)} slow tests detected")

    if suite_metrics.quality_score < 70:
        print(f"[WARNING] Quality score below threshold: {suite_metrics.quality_score:.1f}/100")
