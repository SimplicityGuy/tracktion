"""
Custom exceptions for the tracklist service.

Provides specific exception types for different error scenarios.
"""

from typing import Any


class TracklistServiceError(Exception):
    """Base exception for all tracklist service errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ScrapingError(TracklistServiceError):
    """Raised when web scraping fails."""

    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize scraping error.

        Args:
            message: Error message
            url: URL that failed to scrape
            status_code: HTTP status code if applicable
            details: Additional error details
        """
        error_details = details or {}
        if url:
            error_details["url"] = url
        if status_code:
            error_details["status_code"] = status_code

        super().__init__(message, "SCRAPING_ERROR", error_details)
        self.url = url
        self.status_code = status_code


class RateLimitError(ScrapingError):
    """Raised when rate limiting is detected."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        url: str | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            url: URL that triggered rate limiting
        """
        details = {}
        if retry_after:
            details["retry_after"] = str(retry_after)

        super().__init__(message, url=url, status_code=429, details=details)
        self.retry_after = retry_after


class ParsingError(TracklistServiceError):
    """Raised when HTML parsing fails."""

    def __init__(
        self,
        message: str,
        element: str | None = None,
        html_snippet: str | None = None,
    ) -> None:
        """Initialize parsing error.

        Args:
            message: Error message
            element: Element that failed to parse
            html_snippet: Snippet of HTML that caused the error
        """
        details = {}
        if element:
            details["element"] = element
        if html_snippet:
            details["html_snippet"] = html_snippet[:500]  # Limit snippet size

        super().__init__(message, "PARSING_ERROR", details)
        self.element = element
        self.html_snippet = html_snippet


class CacheError(TracklistServiceError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        key: str | None = None,
    ) -> None:
        """Initialize cache error.

        Args:
            message: Error message
            operation: Cache operation that failed
            key: Cache key involved in the error
        """
        details = {}
        if operation:
            details["operation"] = operation
        if key:
            details["key"] = key

        super().__init__(message, "CACHE_ERROR", details)
        self.operation = operation
        self.key = key


class MessageQueueError(TracklistServiceError):
    """Raised when message queue operations fail."""

    def __init__(
        self,
        message: str,
        queue_name: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Initialize message queue error.

        Args:
            message: Error message
            queue_name: Queue name involved in the error
            correlation_id: Message correlation ID
        """
        details = {}
        if queue_name:
            details["queue_name"] = queue_name
        if correlation_id:
            details["correlation_id"] = correlation_id

        super().__init__(message, "MESSAGE_QUEUE_ERROR", details)
        self.queue_name = queue_name
        self.correlation_id = correlation_id


class ValidationError(TracklistServiceError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class ConfigurationError(TracklistServiceError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that is invalid
            config_value: Invalid configuration value
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)

        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key
        self.config_value = config_value


class ServiceUnavailableError(TracklistServiceError):
    """Raised when a required service is unavailable."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        """Initialize service unavailable error.

        Args:
            message: Error message
            service_name: Name of the unavailable service
            retry_after: Seconds to wait before retrying
        """
        details = {}
        if service_name:
            details["service_name"] = service_name
        if retry_after:
            details["retry_after"] = str(retry_after)

        super().__init__(message, "SERVICE_UNAVAILABLE", details)
        self.service_name = service_name
        self.retry_after = retry_after


class ImportError(TracklistServiceError):
    """Raised when tracklist import operations fail."""

    def __init__(
        self,
        message: str,
        url: str | None = None,
        tracklist_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize import error.

        Args:
            message: Error message
            url: URL that failed to import
            tracklist_id: ID of tracklist being imported
            details: Additional error details
        """
        error_details = details or {}
        if url:
            error_details["url"] = url
        if tracklist_id:
            error_details["tracklist_id"] = tracklist_id

        super().__init__(message, "IMPORT_ERROR", error_details)
        self.url = url
        self.tracklist_id = tracklist_id


class MatchingError(TracklistServiceError):
    """Raised when audio file matching fails."""

    def __init__(
        self,
        message: str,
        audio_file_id: str | None = None,
        confidence_score: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize matching error.

        Args:
            message: Error message
            audio_file_id: ID of audio file that failed to match
            confidence_score: Confidence score at time of failure
            details: Additional error details
        """
        error_details = details or {}
        if audio_file_id:
            error_details["audio_file_id"] = audio_file_id
        if confidence_score is not None:
            error_details["confidence_score"] = confidence_score

        super().__init__(message, "MATCHING_ERROR", error_details)
        self.audio_file_id = audio_file_id
        self.confidence_score = confidence_score


class TimingError(TracklistServiceError):
    """Raised when timing adjustment operations fail."""

    def __init__(
        self,
        message: str,
        track_position: int | None = None,
        timing_issue: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize timing error.

        Args:
            message: Error message
            track_position: Position of track with timing issue
            timing_issue: Type of timing issue (overlap, gap, invalid, etc.)
            details: Additional error details
        """
        error_details = details or {}
        if track_position is not None:
            error_details["track_position"] = track_position
        if timing_issue:
            error_details["timing_issue"] = timing_issue

        super().__init__(message, "TIMING_ERROR", error_details)
        self.track_position = track_position
        self.timing_issue = timing_issue


class CueGenerationError(TracklistServiceError):
    """Raised when CUE file generation fails."""

    def __init__(
        self,
        message: str,
        cue_format: str | None = None,
        tracklist_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize CUE generation error.

        Args:
            message: Error message
            cue_format: CUE format being generated
            tracklist_id: ID of tracklist for CUE generation
            details: Additional error details
        """
        error_details = details or {}
        if cue_format:
            error_details["cue_format"] = cue_format
        if tracklist_id:
            error_details["tracklist_id"] = tracklist_id

        super().__init__(message, "CUE_GENERATION_ERROR", error_details)
        self.cue_format = cue_format
        self.tracklist_id = tracklist_id


class DatabaseError(TracklistServiceError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        table: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize database error.

        Args:
            message: Error message
            operation: Database operation that failed (insert, update, delete, select)
            table: Database table involved in the error
            details: Additional error details
        """
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if table:
            error_details["table"] = table

        super().__init__(message, "DATABASE_ERROR", error_details)
        self.operation = operation
        self.table = table


class AudioFileError(TracklistServiceError):
    """Raised when audio file operations fail."""

    def __init__(
        self,
        message: str,
        audio_file_id: str | None = None,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize audio file error.

        Args:
            message: Error message
            audio_file_id: ID of audio file
            file_path: Path to audio file
            details: Additional error details
        """
        error_details = details or {}
        if audio_file_id:
            error_details["audio_file_id"] = audio_file_id
        if file_path:
            error_details["file_path"] = file_path

        super().__init__(message, "AUDIO_FILE_ERROR", error_details)
        self.audio_file_id = audio_file_id
        self.file_path = file_path


class TimeoutError(TracklistServiceError):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Error message
            operation: Operation that timed out
            timeout_seconds: Timeout value in seconds
            details: Additional error details
        """
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if timeout_seconds is not None:
            error_details["timeout_seconds"] = timeout_seconds

        super().__init__(message, "TIMEOUT_ERROR", error_details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class DraftNotFoundError(TracklistServiceError):
    """Raised when a draft tracklist is not found."""

    def __init__(
        self,
        draft_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize draft not found error.

        Args:
            draft_id: ID of the missing draft
            details: Additional error details
        """
        message = f"Draft with ID {draft_id} not found"
        error_details = details or {}
        error_details["draft_id"] = draft_id

        super().__init__(message, "DRAFT_NOT_FOUND", error_details)
        self.draft_id = draft_id


class ConcurrentEditError(TracklistServiceError):
    """Raised when concurrent edits are detected."""

    def __init__(
        self,
        tracklist_id: str,
        expected_version: int,
        actual_version: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize concurrent edit error.

        Args:
            tracklist_id: ID of the tracklist
            expected_version: Version that was expected
            actual_version: Actual current version
            details: Additional error details
        """
        message = (
            f"Concurrent edit detected for tracklist {tracklist_id}. "
            f"Expected version {expected_version}, but current version is {actual_version}"
        )
        error_details = details or {}
        error_details.update(
            {
                "tracklist_id": tracklist_id,
                "expected_version": expected_version,
                "actual_version": actual_version,
            }
        )

        super().__init__(message, "CONCURRENT_EDIT", error_details)
        self.tracklist_id = tracklist_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class DuplicatePositionError(ValidationError):
    """Raised when duplicate track positions are detected."""

    def __init__(
        self,
        positions: list[int],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize duplicate position error.

        Args:
            positions: List of duplicate positions
            details: Additional error details
        """
        message = f"Duplicate track positions detected: {positions}"
        error_details = details or {}
        error_details["duplicate_positions"] = positions

        super().__init__(message, field="position")
        self.details.update(error_details)
        self.positions = positions


class PublishValidationError(ValidationError):
    """Raised when a draft fails publishing validation."""

    def __init__(
        self,
        message: str,
        issues: list[str],
        tracklist_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize publish validation error.

        Args:
            message: Error message
            issues: List of validation issues
            tracklist_id: ID of the tracklist
            details: Additional error details
        """
        error_details = details or {}
        error_details["validation_issues"] = issues
        if tracklist_id:
            error_details["tracklist_id"] = tracklist_id

        super().__init__(message)
        self.details.update(error_details)
        self.issues = issues
        self.tracklist_id = tracklist_id


class InvalidTrackPositionError(ValidationError):
    """Raised when an invalid track position is provided."""

    def __init__(
        self,
        position: int,
        max_position: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize invalid track position error.

        Args:
            position: Invalid position
            max_position: Maximum valid position
            details: Additional error details
        """
        message = f"Invalid track position {position}. Must be between 1 and {max_position}"
        error_details = details or {}
        error_details.update(
            {
                "position": position,
                "max_position": max_position,
            }
        )

        super().__init__(message, field="position", value=position)
        self.details.update(error_details)
        self.position = position
        self.max_position = max_position
