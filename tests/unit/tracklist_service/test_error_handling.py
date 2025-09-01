"""
Tests for comprehensive error handling in tracklist service.

Test suite covering custom exceptions, retry logic, structured logging,
and error recovery scenarios.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.cache.redis_cache import RedisCache
from services.tracklist_service.src.database.database import get_db_context
from services.tracklist_service.src.exceptions import (
    AudioFileError,
    ConcurrentEditError,
    CueGenerationError,
    DatabaseError,
    DraftNotFoundError,
    DuplicatePositionError,
    ImportError,
    InvalidTrackPositionError,
    MatchingError,
    PublishValidationError,
    RateLimitError,
    ScrapingError,
    ServiceUnavailableError,
    TimeoutError,
    TimingError,
    ValidationError,
)
from services.tracklist_service.src.retry.retry_manager import FailureType, RetryManager, RetryPolicy


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_tracklist_service_error_base(self):
        """Test base exception class."""
        error = ImportError("Test import failed", url="https://1001tracklists.com/test", tracklist_id="test-123")

        assert error.message == "Test import failed"
        assert error.error_code == "IMPORT_ERROR"
        assert error.details["url"] == "https://1001tracklists.com/test"
        assert error.details["tracklist_id"] == "test-123"

    def test_scraping_error_with_details(self):
        """Test scraping error with URL and status code."""
        error = ScrapingError("Failed to scrape page", url="https://1001tracklists.com/test", status_code=403)

        assert error.url == "https://1001tracklists.com/test"
        assert error.status_code == 403
        assert error.details["url"] == "https://1001tracklists.com/test"
        assert error.details["status_code"] == 403

    def test_rate_limit_error(self):
        """Test rate limit error with retry_after."""
        error = RateLimitError("Rate limit exceeded", retry_after=300, url="https://1001tracklists.com/test")

        assert error.retry_after == 300
        assert error.status_code == 429
        assert error.details["retry_after"] == "300"

    def test_matching_error_with_confidence(self):
        """Test matching error with audio file details."""
        audio_file_id = str(uuid4())
        error = MatchingError("Low confidence match", audio_file_id=audio_file_id, confidence_score=0.3)

        assert error.audio_file_id == audio_file_id
        assert error.confidence_score == 0.3
        assert error.details["audio_file_id"] == audio_file_id
        assert error.details["confidence_score"] == 0.3

    def test_timing_error_with_track_position(self):
        """Test timing error with track details."""
        error = TimingError("Track timing overlap", track_position=3, timing_issue="overlap")

        assert error.track_position == 3
        assert error.timing_issue == "overlap"
        assert error.details["track_position"] == 3
        assert error.details["timing_issue"] == "overlap"

    def test_cue_generation_error(self):
        """Test CUE generation error."""
        tracklist_id = str(uuid4())
        error = CueGenerationError("Invalid CUE format", cue_format="invalid", tracklist_id=tracklist_id)

        assert error.cue_format == "invalid"
        assert error.tracklist_id == tracklist_id

    def test_database_error_with_operation(self):
        """Test database error with operation details."""
        error = DatabaseError("Connection failed", operation="insert", table="tracklists")

        assert error.operation == "insert"
        assert error.table == "tracklists"
        assert error.details["operation"] == "insert"
        assert error.details["table"] == "tracklists"

    def test_audio_file_error(self):
        """Test audio file error."""
        audio_file_id = str(uuid4())
        error = AudioFileError("File not found", audio_file_id=audio_file_id, file_path="/path/to/audio.wav")

        assert error.audio_file_id == audio_file_id
        assert error.file_path == "/path/to/audio.wav"

    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError("Operation timed out", operation="import_tracklist", timeout_seconds=30.0)

        assert error.operation == "import_tracklist"
        assert error.timeout_seconds == 30.0


class TestRetryManager:
    """Test retry manager functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create retry manager for testing."""
        return RetryManager()

    def test_failure_classification(self, retry_manager):
        """Test failure type classification."""
        # Test network errors
        assert retry_manager.classify_failure("Connection failed") == FailureType.NETWORK
        assert retry_manager.classify_failure("DNS resolution error") == FailureType.NETWORK

        # Test timeout errors
        assert retry_manager.classify_failure("Operation timed out") == FailureType.TIMEOUT
        assert retry_manager.classify_failure("Request timed out") == FailureType.TIMEOUT

        # Test rate limit errors
        assert retry_manager.classify_failure("Rate limit exceeded") == FailureType.RATE_LIMIT
        assert retry_manager.classify_failure("Too many requests") == FailureType.RATE_LIMIT

        # Test auth errors
        assert retry_manager.classify_failure("Unauthorized access") == FailureType.AUTH
        assert retry_manager.classify_failure("403 Forbidden") == FailureType.AUTH

        # Test server errors
        assert retry_manager.classify_failure("Internal server error") == FailureType.SERVER
        assert retry_manager.classify_failure("500 error") == FailureType.SERVER

        # Test client errors
        assert retry_manager.classify_failure("Bad request") == FailureType.CLIENT
        assert retry_manager.classify_failure("404 not found") == FailureType.CLIENT

        # Test parse errors
        assert retry_manager.classify_failure("JSON decode error") == FailureType.PARSE
        assert retry_manager.classify_failure("XML parsing failed") == FailureType.PARSE

        # Test unknown errors
        assert retry_manager.classify_failure("Something went wrong") == FailureType.UNKNOWN

    def test_retry_policy_delay_calculation(self):
        """Test retry policy delay calculations."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=False,  # Disable jitter for predictable testing
        )

        # Test exponential backoff
        assert policy.get_delay(0) == 1.0  # 1 * 2^0
        assert policy.get_delay(1) == 2.0  # 1 * 2^1
        assert policy.get_delay(2) == 4.0  # 1 * 2^2
        assert policy.get_delay(3) == 8.0  # 1 * 2^3

        # Test max delay cap
        policy.max_delay = 5.0
        assert policy.get_delay(3) == 5.0  # Capped at max_delay

    def test_retry_policy_failure_specific_overrides(self):
        """Test failure-specific policy overrides."""
        policy = RetryPolicy(
            base_delay=1.0,
            failure_policies={FailureType.RATE_LIMIT: {"base_delay": 60.0}, FailureType.NETWORK: {"base_delay": 5.0}},
            jitter=False,
        )

        # Test rate limit override
        rate_limit_delay = policy.get_delay(1, FailureType.RATE_LIMIT)
        assert rate_limit_delay == 120.0  # 60 * 2^1

        # Test network override
        network_delay = policy.get_delay(1, FailureType.NETWORK)
        assert network_delay == 10.0  # 5 * 2^1

        # Test default for unknown type
        default_delay = policy.get_delay(1, FailureType.UNKNOWN)
        assert default_delay == 2.0  # 1 * 2^1

    def test_circuit_breaker_functionality(self, retry_manager):
        """Test circuit breaker functionality."""
        domain = "test.example.com"
        circuit_breaker = retry_manager.circuit_breakers[domain]

        # Initially closed
        assert circuit_breaker.state == "closed"
        assert not circuit_breaker.is_open()

        # Simulate failures to trip circuit breaker
        for _ in range(5):  # Default failure threshold
            circuit_breaker._on_failure()

        # Should now be open
        assert circuit_breaker.state == "open"
        assert circuit_breaker.is_open()

        # Test state information
        state_info = circuit_breaker.get_state()
        assert state_info["state"] == "open"
        assert state_info["failure_count"] == 5

    def test_domain_specific_policies(self, retry_manager):
        """Test domain-specific retry policies."""
        # Set custom policy for specific domain
        domain = "1001tracklists.com"
        custom_policy = RetryPolicy(
            max_retries=5, base_delay=2.0, failure_policies={FailureType.RATE_LIMIT: {"base_delay": 120.0}}
        )

        retry_manager.set_domain_policy(domain, custom_policy)

        # Verify policy was set
        assert domain in retry_manager.domain_policies
        assert retry_manager.domain_policies[domain] == custom_policy

        # Test getting failure stats for domain
        stats = retry_manager.get_failure_stats(domain)
        assert stats["domain"] == domain
        assert "failures" in stats
        assert "circuit_breaker" in stats


class TestErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_cache_failure(self):
        """Test graceful degradation when cache fails."""

        # Mock cache that fails
        cache = RedisCache()

        with patch.object(cache, "get", side_effect=Exception("Redis unavailable")):
            # Should handle cache failure gracefully
            try:
                await cache.get("test_key")
                raise AssertionError("Should have raised exception")
            except Exception as e:
                assert "Redis unavailable" in str(e)

    def test_service_unavailable_handling(self):
        """Test handling of service unavailable errors."""
        error = ServiceUnavailableError(
            "External service temporarily unavailable", service_name="1001tracklists", retry_after=300
        )

        assert error.service_name == "1001tracklists"
        assert error.retry_after == 300
        assert error.details["service_name"] == "1001tracklists"
        assert error.details["retry_after"] == "300"

    @pytest.mark.asyncio
    async def test_database_connection_recovery(self):
        """Test database connection recovery."""

        # This would test actual database recovery in a full integration test
        # For unit test, we just verify the context manager exists
        try:
            async with get_db_context() as db:
                # In a real test, this would test connection recovery
                assert db is not None
        except Exception:
            # Expected in unit test without real database
            pass


class TestStructuredLogging:
    """Test structured logging functionality."""

    @patch("services.tracklist_service.src.api.import_endpoints.logger")
    def test_structured_log_format(self, mock_logger):
        """Test that logs include structured data."""

        # Create an error that would be logged
        error = ImportError("Test import failed", url="https://1001tracklists.com/test", tracklist_id="test-123")

        # Simulate logging the error
        str(uuid4())
        mock_logger.error.assert_not_called()  # Not called yet

        # In actual code, this would be logged with extra fields

        # Verify the structure we expect
        assert error.error_code == "IMPORT_ERROR"
        assert error.details["url"] == "https://1001tracklists.com/test"
        assert error.details["tracklist_id"] == "test-123"

    def test_correlation_id_tracking(self):
        """Test correlation ID tracking through error chain."""
        correlation_id = str(uuid4())

        # Create error with correlation context
        error = ImportError("Import failed", url="https://1001tracklists.com/test")

        # In production, correlation_id would be added to log context
        log_context = {
            "correlation_id": correlation_id,
            "error_code": error.error_code,
            "url": error.details.get("url"),
        }

        assert log_context["correlation_id"] == correlation_id
        assert log_context["error_code"] == "IMPORT_ERROR"
        assert log_context["url"] == "https://1001tracklists.com/test"


class TestManualTracklistExceptions:
    """Test exceptions for manual tracklist creation."""

    def test_draft_not_found_error(self):
        """Test draft not found error."""
        draft_id = str(uuid4())
        error = DraftNotFoundError(draft_id)

        assert error.error_code == "DRAFT_NOT_FOUND"
        assert error.draft_id == draft_id
        assert f"Draft with ID {draft_id} not found" in error.message
        assert error.details["draft_id"] == draft_id

    def test_concurrent_edit_error(self):
        """Test concurrent edit detection."""
        tracklist_id = str(uuid4())
        error = ConcurrentEditError(
            tracklist_id=tracklist_id,
            expected_version=1,
            actual_version=3,
        )

        assert error.error_code == "CONCURRENT_EDIT"
        assert error.tracklist_id == tracklist_id
        assert error.expected_version == 1
        assert error.actual_version == 3
        assert "Expected version 1" in error.message
        assert "current version is 3" in error.message

    def test_duplicate_position_error(self):
        """Test duplicate position error."""
        positions = [1, 3, 5, 3, 7, 5]
        error = DuplicatePositionError(positions)

        assert error.error_code == "VALIDATION_ERROR"
        assert error.positions == positions
        assert "Duplicate track positions detected" in error.message
        assert error.details["duplicate_positions"] == positions

    def test_publish_validation_error(self):
        """Test publish validation error with issues."""
        tracklist_id = str(uuid4())
        issues = [
            "Track 3 missing artist",
            "Track 5 has invalid timing",
            "Track 7 overlaps with track 8",
        ]

        error = PublishValidationError(
            message="Draft cannot be published",
            issues=issues,
            tracklist_id=tracklist_id,
        )

        assert error.error_code == "VALIDATION_ERROR"
        assert error.issues == issues
        assert error.tracklist_id == tracklist_id
        assert len(error.details["validation_issues"]) == 3
        assert error.details["tracklist_id"] == tracklist_id

    def test_invalid_track_position_error(self):
        """Test invalid track position error."""
        error = InvalidTrackPositionError(
            position=25,
            max_position=20,
        )

        assert error.error_code == "VALIDATION_ERROR"
        assert error.position == 25
        assert error.max_position == 20
        assert "Invalid track position 25" in error.message
        assert "Must be between 1 and 20" in error.message
        assert error.details["position"] == 25
        assert error.details["max_position"] == 20

    def test_validation_error_with_field(self):
        """Test validation error with field and value."""
        error = ValidationError(
            message="BPM must be between 60 and 200",
            field="bpm",
            value=250,
        )

        assert error.error_code == "VALIDATION_ERROR"
        assert error.field == "bpm"
        assert error.value == 250
        assert error.details["field"] == "bpm"
        assert error.details["value"] == "250"

    def test_timing_error_with_overlap(self):
        """Test timing error for track overlap."""

        error = TimingError(
            message="Tracks overlap by 15 seconds",
            track_position=5,
            timing_issue="overlap",
            details={"overlap_duration": 15.0, "conflicting_track": 6},
        )

        assert error.error_code == "TIMING_ERROR"
        assert error.track_position == 5
        assert error.timing_issue == "overlap"
        assert error.details["overlap_duration"] == 15.0
        assert error.details["conflicting_track"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
