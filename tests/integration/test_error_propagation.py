"""
Integration tests for error propagation between services.

Tests how errors, exceptions, and failures propagate through the service mesh,
including error handling strategies, circuit breakers, and fallback mechanisms.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import pytest

from tests.shared_utilities import (
    TestDataGenerator,
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in the system."""

    TIMEOUT = "timeout"
    VALIDATION = "validation"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_ERROR = "authentication_error"
    NETWORK_ERROR = "network_error"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"


class ServiceError(Exception):
    """Base exception for service errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        service_name: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.service_name = service_name
        self.error_code = error_code or error_type.value
        self.details = details or {}
        self.timestamp = datetime.now(UTC)


class CircuitBreaker:
    """Circuit breaker implementation for service protection."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def should_allow_request(self) -> bool:
        """Check if request should be allowed through."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if self.last_failure_time and (datetime.now(UTC) - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False

        # HALF_OPEN state
        return True

    def record_success(self):
        """Record a successful request."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class MockService:
    """Mock service with error simulation capabilities."""

    def __init__(self, name: str, circuit_breaker: CircuitBreaker = None):
        self.name = name
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.error_scenarios = []
        self.call_count = 0
        self.error_count = 0
        self.response_delay = 0.0
        self.is_healthy = True

    def configure_error_scenario(
        self, error_type: ErrorType, trigger_after: int = 0, probability: float = 1.0, message: str | None = None
    ):
        """Configure an error scenario."""
        self.error_scenarios.append(
            {
                "error_type": error_type,
                "trigger_after": trigger_after,
                "probability": probability,
                "message": message or f"{error_type.value} error from {self.name}",
            }
        )

    def set_response_delay(self, delay: float):
        """Set response delay for simulating slow services."""
        self.response_delay = delay

    async def process_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process a request with potential error simulation."""
        self.call_count += 1

        # Check circuit breaker
        if not self.circuit_breaker.should_allow_request():
            self.error_count += 1
            raise ServiceError(
                f"Circuit breaker OPEN for {self.name}", ErrorType.RESOURCE_EXHAUSTED, self.name, "CIRCUIT_BREAKER_OPEN"
            )

        # Simulate response delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        # Check for configured error scenarios
        for scenario in self.error_scenarios:
            if self.call_count > scenario["trigger_after"] and (
                scenario["probability"] >= 1.0 or hash(str(request_data)) % 100 < scenario["probability"] * 100
            ):
                self.error_count += 1
                self.circuit_breaker.record_failure()

                error = ServiceError(
                    scenario["message"],
                    scenario["error_type"],
                    self.name,
                    f"{scenario['error_type'].value.upper()}_ERROR",
                )
                raise error

        # Successful processing
        self.circuit_breaker.record_success()
        return {
            "service": self.name,
            "status": "success",
            "processed_at": datetime.now(UTC).isoformat(),
            "request_id": request_data.get("id", str(uuid4())),
        }

    def reset_metrics(self):
        """Reset service metrics."""
        self.call_count = 0
        self.error_count = 0
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "CLOSED"


@pytest.fixture
def data_generator():
    """Test data generator."""
    return TestDataGenerator(seed=42)


@pytest.fixture
def service_mesh():
    """Service mesh with multiple mock services."""
    return {
        "analysis_service": MockService("analysis_service"),
        "cataloging_service": MockService("cataloging_service"),
        "tracklist_service": MockService("tracklist_service"),
        "file_watcher": MockService("file_watcher"),
        "notification_service": MockService("notification_service"),
    }


class TestBasicErrorPropagation:
    """Test basic error propagation scenarios."""

    @pytest.mark.asyncio
    async def test_single_service_error_propagation(self, service_mesh, data_generator):
        """Test error propagation from a single failing service."""

        analysis_service = service_mesh["analysis_service"]
        cataloging_service = service_mesh["cataloging_service"]

        # Configure analysis service to fail
        analysis_service.configure_error_scenario(
            ErrorType.DATABASE_ERROR, trigger_after=0, message="Database connection failed"
        )

        # Test request processing
        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Analysis service should fail
        with pytest.raises(ServiceError) as exc_info:
            await analysis_service.process_request(request_data)

        error = exc_info.value
        assert error.error_type == ErrorType.DATABASE_ERROR
        assert error.service_name == "analysis_service"
        assert "Database connection failed" in str(error)

        # Cataloging service should still work (no error propagation yet)
        result = await cataloging_service.process_request(request_data)
        assert result["status"] == "success"
        assert result["service"] == "cataloging_service"

        # Verify error counts
        assert analysis_service.error_count == 1
        assert cataloging_service.error_count == 0

        logger.info("✅ Single service error propagation test completed successfully")

    @pytest.mark.asyncio
    async def test_cascading_service_failures(self, service_mesh, data_generator):
        """Test cascading failures across multiple services."""

        # Configure failure cascade: analysis -> cataloging -> tracklist
        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Track propagation chain
        failure_chain = []

        async def process_with_dependency_chain():
            """Simulate service dependency chain."""

            try:
                # Step 1: Analysis service fails
                service_mesh["analysis_service"].configure_error_scenario(
                    ErrorType.TIMEOUT, message="Analysis timeout after 30 seconds"
                )
                await service_mesh["analysis_service"].process_request(request_data)

            except ServiceError as e:
                failure_chain.append({"service": "analysis_service", "error": e.error_type})

                # Step 2: Cataloging service fails due to missing analysis data
                try:
                    service_mesh["cataloging_service"].configure_error_scenario(
                        ErrorType.VALIDATION, message="Missing analysis data for cataloging"
                    )
                    await service_mesh["cataloging_service"].process_request(request_data)

                except ServiceError as e2:
                    failure_chain.append({"service": "cataloging_service", "error": e2.error_type})

                    # Step 3: Tracklist service fails due to missing catalog data
                    try:
                        service_mesh["tracklist_service"].configure_error_scenario(
                            ErrorType.VALIDATION, message="Missing catalog data for tracklist"
                        )
                        await service_mesh["tracklist_service"].process_request(request_data)

                    except ServiceError as e3:
                        failure_chain.append({"service": "tracklist_service", "error": e3.error_type})

        await process_with_dependency_chain()

        # Verify cascading failure
        assert len(failure_chain) == 3
        assert failure_chain[0]["service"] == "analysis_service"
        assert failure_chain[0]["error"] == ErrorType.TIMEOUT
        assert failure_chain[1]["service"] == "cataloging_service"
        assert failure_chain[1]["error"] == ErrorType.VALIDATION
        assert failure_chain[2]["service"] == "tracklist_service"
        assert failure_chain[2]["error"] == ErrorType.VALIDATION

        logger.info("✅ Cascading service failures test completed successfully")

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, service_mesh, data_generator):
        """Test circuit breaker activation and error containment."""

        analysis_service = service_mesh["analysis_service"]

        # Configure service to fail consistently
        analysis_service.configure_error_scenario(
            ErrorType.EXTERNAL_SERVICE_FAILURE, trigger_after=0, message="External API unavailable"
        )

        # Make requests until circuit breaker opens
        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}
        failure_count = 0
        circuit_breaker_errors = 0

        for _ in range(10):
            try:
                await analysis_service.process_request(request_data)
            except ServiceError as e:
                if e.error_code == "CIRCUIT_BREAKER_OPEN":
                    circuit_breaker_errors += 1
                else:
                    failure_count += 1

        # Verify circuit breaker behavior
        assert failure_count >= analysis_service.circuit_breaker.failure_threshold
        assert circuit_breaker_errors > 0
        assert analysis_service.circuit_breaker.state == "OPEN"

        # Verify error containment
        assert analysis_service.error_count == failure_count + circuit_breaker_errors

        logger.info("✅ Circuit breaker activation test completed successfully")


class TestErrorRecoveryMechanisms:
    """Test error recovery and fallback mechanisms."""

    @pytest.mark.asyncio
    async def test_service_failover_mechanism(self, service_mesh, data_generator):
        """Test automatic failover to backup service."""

        primary_service = service_mesh["analysis_service"]
        backup_service = service_mesh["cataloging_service"]  # Using as backup for simplicity

        # Configure primary service to fail
        primary_service.configure_error_scenario(
            ErrorType.NETWORK_ERROR, trigger_after=0, message="Primary service network error"
        )

        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Implement failover logic
        async def process_with_failover(request):
            """Process request with failover to backup service."""
            try:
                return await primary_service.process_request(request)
            except ServiceError as primary_error:
                logger.warning(f"Primary service failed: {primary_error}, failing over to backup")

                try:
                    result = await backup_service.process_request(request)
                    result["failover"] = True
                    result["primary_error"] = str(primary_error)
                    return result
                except ServiceError as backup_error:
                    # Both services failed
                    raise ServiceError(
                        f"Both primary and backup services failed. Primary: {primary_error}, Backup: {backup_error}",
                        ErrorType.RESOURCE_EXHAUSTED,
                        "failover_system",
                    ) from backup_error

        # Test failover
        result = await process_with_failover(request_data)

        # Verify failover success
        assert result["status"] == "success"
        assert result["failover"] is True
        assert "Primary service network error" in result["primary_error"]
        assert result["service"] == "cataloging_service"  # Backup service

        # Verify error counts
        assert primary_service.error_count == 1
        assert backup_service.error_count == 0

        logger.info("✅ Service failover mechanism test completed successfully")

    @pytest.mark.asyncio
    async def test_retry_with_backoff_strategy(self, service_mesh, data_generator):
        """Test retry mechanism with exponential backoff."""

        analysis_service = service_mesh["analysis_service"]

        # Configure intermittent failures (50% success rate)
        analysis_service.configure_error_scenario(
            ErrorType.TIMEOUT, trigger_after=0, probability=0.5, message="Intermittent timeout error"
        )

        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Implement retry with exponential backoff
        async def retry_with_backoff(request, max_retries=5):
            """Retry request with exponential backoff."""
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    result = await analysis_service.process_request(request)
                    if attempt > 0:
                        result["retry_count"] = attempt
                    return result

                except ServiceError as e:
                    last_error = e

                    if attempt < max_retries:
                        # Exponential backoff: 0.1, 0.2, 0.4, 0.8, 1.6 seconds
                        backoff_time = 0.1 * (2**attempt)
                        await asyncio.sleep(backoff_time)
                        logger.info(f"Retry attempt {attempt + 1} after {backoff_time}s")
                    else:
                        # Max retries exceeded
                        raise ServiceError(
                            f"Max retries ({max_retries}) exceeded. Last error: {last_error}",
                            last_error.error_type,
                            last_error.service_name,
                            "MAX_RETRIES_EXCEEDED",
                        ) from last_error

            # This should never be reached due to the raise in the else clause
            return None

        # Test retry mechanism
        start_time = time.time()
        try:
            result = await retry_with_backoff(request_data)
            execution_time = time.time() - start_time

            # Verify successful retry
            assert result["status"] == "success"

            # If retries occurred, verify timing
            if "retry_count" in result:
                assert execution_time > 0.1  # At least one backoff occurred
                assert result["retry_count"] <= 5

            logger.info(f"✅ Retry succeeded after {result.get('retry_count', 0)} attempts")

        except ServiceError as e:
            # If all retries failed, verify error handling
            assert e.error_code == "MAX_RETRIES_EXCEEDED"
            execution_time = time.time() - start_time
            assert execution_time > 3.0  # Should have taken time for all backoffs

            logger.info("✅ Retry mechanism correctly exhausted all attempts")

        logger.info("✅ Retry with backoff strategy test completed successfully")

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, service_mesh, data_generator):
        """Test graceful degradation when non-critical services fail."""

        # Core services (critical)
        analysis_service = service_mesh["analysis_service"]
        cataloging_service = service_mesh["cataloging_service"]

        # Optional services (non-critical)
        notification_service = service_mesh["notification_service"]

        # Configure non-critical service to fail
        notification_service.configure_error_scenario(
            ErrorType.EXTERNAL_SERVICE_FAILURE, message="Email service unavailable"
        )

        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Implement graceful degradation
        async def process_with_degradation(request):
            """Process request with graceful degradation for non-critical services."""
            result = {"status": "success", "warnings": []}

            # Critical services (must succeed)
            try:
                analysis_result = await analysis_service.process_request(request)
                result["analysis"] = analysis_result
            except ServiceError as e:
                # Critical service failure - propagate error
                raise e

            try:
                catalog_result = await cataloging_service.process_request(request)
                result["cataloging"] = catalog_result
            except ServiceError as e:
                # Critical service failure - propagate error
                raise e

            # Non-critical services (can fail gracefully)
            try:
                notification_result = await notification_service.process_request(request)
                result["notifications"] = notification_result
            except ServiceError as e:
                # Non-critical service failure - degrade gracefully
                result["warnings"].append(
                    {"service": e.service_name, "error": str(e), "impact": "Notifications will not be sent"}
                )
                result["notifications"] = {"status": "skipped", "reason": "service_unavailable"}

            return result

        # Test graceful degradation
        result = await process_with_degradation(request_data)

        # Verify core functionality succeeded
        assert result["status"] == "success"
        assert result["analysis"]["status"] == "success"
        assert result["cataloging"]["status"] == "success"

        # Verify non-critical service degraded gracefully
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["service"] == "notification_service"
        assert result["notifications"]["status"] == "skipped"

        # Verify error counts
        assert analysis_service.error_count == 0
        assert cataloging_service.error_count == 0
        assert notification_service.error_count == 1

        logger.info("✅ Graceful degradation test completed successfully")


class TestComplexErrorScenarios:
    """Test complex error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_error_storm_handling(self, service_mesh, data_generator):
        """Test handling of error storms across multiple services."""

        # Configure all services to fail simultaneously
        for service_name, service in service_mesh.items():
            service.configure_error_scenario(
                ErrorType.RESOURCE_EXHAUSTED, trigger_after=0, message=f"Resource exhaustion in {service_name}"
            )

        # Generate multiple concurrent requests
        concurrent_requests = 10
        request_data_list = [
            {"id": data_generator.generate_uuid_string(), "file_path": f"/test_{i}.mp3"}
            for i in range(concurrent_requests)
        ]

        # Track error storm metrics
        error_storm_metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "error_types": {},
            "services_affected": set(),
            "circuit_breakers_opened": 0,
        }

        async def process_request_with_metrics(request_data, service_name):
            """Process request and collect error storm metrics."""
            error_storm_metrics["total_requests"] += 1

            try:
                service = service_mesh[service_name]
                result = await service.process_request(request_data)
                return {"service": service_name, "result": result}

            except ServiceError as e:
                error_storm_metrics["total_errors"] += 1
                error_storm_metrics["services_affected"].add(e.service_name)

                if e.error_type not in error_storm_metrics["error_types"]:
                    error_storm_metrics["error_types"][e.error_type] = 0
                error_storm_metrics["error_types"][e.error_type] += 1

                if e.error_code == "CIRCUIT_BREAKER_OPEN":
                    error_storm_metrics["circuit_breakers_opened"] += 1

                return {"service": service_name, "error": e}

        # Execute error storm simulation
        tasks = []
        for request_data in request_data_list:
            tasks.extend(process_request_with_metrics(request_data, service_name) for service_name in service_mesh)

        await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze error storm impact
        # Note: successful_requests and failed_requests variables removed as they were unused

        # Verify error storm was handled
        assert error_storm_metrics["total_requests"] == concurrent_requests * len(service_mesh)
        assert error_storm_metrics["total_errors"] > 0
        assert len(error_storm_metrics["services_affected"]) == len(service_mesh)

        # Verify circuit breakers activated under load
        total_circuit_breaker_errors = sum(service.circuit_breaker.failure_count for service in service_mesh.values())
        assert total_circuit_breaker_errors > 0

        logger.info("✅ Error storm handling test completed successfully")
        logger.info(f"Error storm metrics: {error_storm_metrics}")

    @pytest.mark.asyncio
    async def test_partial_system_recovery(self, service_mesh, data_generator):
        """Test system recovery when some services recover before others."""

        # Configure staggered recovery times
        service_recovery_times = {
            "analysis_service": 0.1,  # Recovers quickly
            "cataloging_service": 0.3,  # Recovers moderately
            "tracklist_service": 0.5,  # Recovers slowly
        }

        # Configure initial failures for all services
        for service_name, recovery_time in service_recovery_times.items():
            service = service_mesh[service_name]
            service.configure_error_scenario(
                ErrorType.DATABASE_ERROR, trigger_after=0, message=f"Database maintenance in {service_name}"
            )
            service.set_response_delay(recovery_time)

        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Track recovery timeline
        recovery_timeline = []

        async def test_service_recovery(service_name: str, test_time: float):
            """Test if a service has recovered at a given time."""
            await asyncio.sleep(test_time)

            try:
                service = service_mesh[service_name]
                # Clear error scenarios to simulate recovery
                service.error_scenarios = []

                result = await service.process_request(request_data)
                recovery_timeline.append({"service": service_name, "recovered_at": test_time, "status": "recovered"})
                return result

            except ServiceError:
                recovery_timeline.append(
                    {"service": service_name, "recovered_at": test_time, "status": "still_failing"}
                )
                return None

        # Test recovery at different time intervals
        recovery_tasks = []
        for service_name, recovery_time in service_recovery_times.items():
            # Test just before expected recovery
            recovery_tasks.append(test_service_recovery(service_name, recovery_time - 0.05))
            # Test just after expected recovery
            recovery_tasks.append(test_service_recovery(service_name, recovery_time + 0.05))

        await asyncio.gather(*recovery_tasks)

        # Analyze recovery pattern
        recovered_services = [entry for entry in recovery_timeline if entry["status"] == "recovered"]
        # Note: still_failing_services variable removed as it was unused

        # Verify staggered recovery
        assert len(recovered_services) > 0
        assert len(recovery_timeline) == len(service_recovery_times) * 2

        # Verify recovery order (analysis should recover first)
        analysis_recoveries = [entry for entry in recovered_services if entry["service"] == "analysis_service"]
        if analysis_recoveries:
            earliest_recovery = min(entry["recovered_at"] for entry in recovered_services)
            analysis_recovery_time = min(entry["recovered_at"] for entry in analysis_recoveries)
            assert analysis_recovery_time <= earliest_recovery + 0.1  # Within tolerance

        logger.info("✅ Partial system recovery test completed successfully")
        logger.info(f"Recovery timeline: {recovery_timeline}")

    @pytest.mark.asyncio
    async def test_error_correlation_and_root_cause_analysis(self, service_mesh, data_generator):
        """Test error correlation and root cause analysis across services."""

        # Simulate a root cause that affects multiple services
        root_cause_id = data_generator.generate_uuid_string()
        affected_services = ["analysis_service", "cataloging_service", "tracklist_service"]

        # Configure correlated errors
        for service_name in affected_services:
            service = service_mesh[service_name]
            service.configure_error_scenario(
                ErrorType.DATABASE_ERROR,
                trigger_after=0,
                message=f"Database connection pool exhausted - correlation_id: {root_cause_id}",
            )

        request_data = {"id": data_generator.generate_uuid_string(), "file_path": "/test.mp3"}

        # Collect error correlation data
        error_correlation_data = {"errors": [], "correlation_patterns": {}, "timeline": []}

        async def collect_correlated_errors():
            """Collect errors for correlation analysis."""

            for service_name in affected_services:
                try:
                    service = service_mesh[service_name]
                    await service.process_request(request_data)

                except ServiceError as e:
                    error_event = {
                        "service": e.service_name,
                        "error_type": e.error_type.value,
                        "message": str(e),
                        "timestamp": e.timestamp.isoformat(),
                        "correlation_id": None,
                    }

                    # Extract correlation ID from error message
                    if "correlation_id:" in str(e):
                        correlation_part = str(e).split("correlation_id:")[1].strip()
                        error_event["correlation_id"] = correlation_part

                    error_correlation_data["errors"].append(error_event)
                    error_correlation_data["timeline"].append(
                        {"timestamp": e.timestamp, "service": e.service_name, "event": "error_occurred"}
                    )

        await collect_correlated_errors()

        # Analyze error correlation
        # Group errors by correlation ID
        for error in error_correlation_data["errors"]:
            if error["correlation_id"]:
                correlation_id = error["correlation_id"]
                if correlation_id not in error_correlation_data["correlation_patterns"]:
                    error_correlation_data["correlation_patterns"][correlation_id] = []
                error_correlation_data["correlation_patterns"][correlation_id].append(error)

        # Verify error correlation
        assert len(error_correlation_data["errors"]) == len(affected_services)
        assert root_cause_id in error_correlation_data["correlation_patterns"]

        # Verify all errors share the same root cause
        correlated_errors = error_correlation_data["correlation_patterns"][root_cause_id]
        assert len(correlated_errors) == len(affected_services)

        # Verify error timing correlation (should occur close together)
        error_timestamps = [error["timestamp"] for error in correlated_errors]
        earliest = min(error_timestamps)
        latest = max(error_timestamps)
        time_span = (datetime.fromisoformat(latest) - datetime.fromisoformat(earliest)).total_seconds()
        assert time_span < 1.0  # All errors within 1 second

        # Verify same error type across services (indicating shared root cause)
        error_types = {error["error_type"] for error in correlated_errors}
        assert len(error_types) == 1  # All same error type
        assert ErrorType.DATABASE_ERROR.value in error_types

        logger.info("✅ Error correlation and root cause analysis test completed successfully")
        logger.info(f"Correlation analysis: {error_correlation_data['correlation_patterns']}")
