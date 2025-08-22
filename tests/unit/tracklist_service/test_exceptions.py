"""Tests for custom exceptions."""

from services.tracklist_service.src.exceptions import (
    CacheError,
    ConfigurationError,
    MessageQueueError,
    ParsingError,
    RateLimitError,
    ScrapingError,
    ServiceUnavailableError,
    TracklistServiceError,
    ValidationError,
)


class TestTracklistServiceError:
    """Test base exception class."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        error = TracklistServiceError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "TracklistServiceError"
        assert error.details == {}

    def test_exception_with_details(self):
        """Test exception with error code and details."""
        error = TracklistServiceError("Test error", error_code="TEST_ERROR", details={"key": "value"})

        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}


class TestScrapingError:
    """Test scraping error exception."""

    def test_basic_scraping_error(self):
        """Test basic scraping error."""
        error = ScrapingError("Scraping failed")

        assert error.message == "Scraping failed"
        assert error.error_code == "SCRAPING_ERROR"
        assert error.url is None
        assert error.status_code is None

    def test_scraping_error_with_url_and_status(self):
        """Test scraping error with URL and status code."""
        error = ScrapingError("Page not found", url="https://example.com", status_code=404)

        assert error.message == "Page not found"
        assert error.url == "https://example.com"
        assert error.status_code == 404
        assert error.details["url"] == "https://example.com"
        assert error.details["status_code"] == 404

    def test_scraping_error_with_custom_details(self):
        """Test scraping error with custom details."""
        error = ScrapingError("Timeout", url="https://example.com", details={"timeout": 30, "retries": 3})

        assert error.details["url"] == "https://example.com"
        assert error.details["timeout"] == 30
        assert error.details["retries"] == 3


class TestRateLimitError:
    """Test rate limit error exception."""

    def test_basic_rate_limit_error(self):
        """Test basic rate limit error."""
        error = RateLimitError()

        assert error.message == "Rate limit exceeded"
        assert error.error_code == "SCRAPING_ERROR"
        assert error.status_code == 429
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """Test rate limit error with retry after."""
        error = RateLimitError(message="Too many requests", retry_after=60, url="https://example.com")

        assert error.message == "Too many requests"
        assert error.retry_after == 60
        assert error.url == "https://example.com"
        assert error.details["retry_after"] == "60"


class TestParsingError:
    """Test parsing error exception."""

    def test_basic_parsing_error(self):
        """Test basic parsing error."""
        error = ParsingError("Failed to parse")

        assert error.message == "Failed to parse"
        assert error.error_code == "PARSING_ERROR"
        assert error.element is None
        assert error.html_snippet is None

    def test_parsing_error_with_element(self):
        """Test parsing error with element."""
        error = ParsingError("Element not found", element="div.content")

        assert error.element == "div.content"
        assert error.details["element"] == "div.content"

    def test_parsing_error_with_html_snippet(self):
        """Test parsing error with HTML snippet."""
        long_html = "x" * 1000
        error = ParsingError("Invalid HTML", html_snippet=long_html)

        assert error.html_snippet == long_html
        # Should truncate to 500 chars in details
        assert len(error.details["html_snippet"]) == 500


class TestCacheError:
    """Test cache error exception."""

    def test_basic_cache_error(self):
        """Test basic cache error."""
        error = CacheError("Cache operation failed")

        assert error.message == "Cache operation failed"
        assert error.error_code == "CACHE_ERROR"
        assert error.operation is None
        assert error.key is None

    def test_cache_error_with_operation_and_key(self):
        """Test cache error with operation and key."""
        error = CacheError("Failed to set cache", operation="SET", key="search:dj:test")

        assert error.operation == "SET"
        assert error.key == "search:dj:test"
        assert error.details["operation"] == "SET"
        assert error.details["key"] == "search:dj:test"


class TestMessageQueueError:
    """Test message queue error exception."""

    def test_basic_message_queue_error(self):
        """Test basic message queue error."""
        error = MessageQueueError("Queue connection failed")

        assert error.message == "Queue connection failed"
        assert error.error_code == "MESSAGE_QUEUE_ERROR"
        assert error.queue_name is None
        assert error.correlation_id is None

    def test_message_queue_error_with_details(self):
        """Test message queue error with details."""
        error = MessageQueueError("Failed to publish", queue_name="search_queue", correlation_id="123-456")

        assert error.queue_name == "search_queue"
        assert error.correlation_id == "123-456"
        assert error.details["queue_name"] == "search_queue"
        assert error.details["correlation_id"] == "123-456"


class TestValidationError:
    """Test validation error exception."""

    def test_basic_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Invalid data")

        assert error.message == "Invalid data"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field is None
        assert error.value is None

    def test_validation_error_with_field_and_value(self):
        """Test validation error with field and value."""
        error = ValidationError("Invalid email format", field="email", value="not-an-email")

        assert error.field == "email"
        assert error.value == "not-an-email"
        assert error.details["field"] == "email"
        assert error.details["value"] == "not-an-email"


class TestConfigurationError:
    """Test configuration error exception."""

    def test_basic_configuration_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid configuration")

        assert error.message == "Invalid configuration"
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.config_key is None
        assert error.config_value is None

    def test_configuration_error_with_key_and_value(self):
        """Test configuration error with key and value."""
        error = ConfigurationError("Invalid port number", config_key="api.port", config_value=99999)

        assert error.config_key == "api.port"
        assert error.config_value == 99999
        assert error.details["config_key"] == "api.port"
        assert error.details["config_value"] == "99999"


class TestServiceUnavailableError:
    """Test service unavailable error exception."""

    def test_basic_service_unavailable_error(self):
        """Test basic service unavailable error."""
        error = ServiceUnavailableError("Service is down")

        assert error.message == "Service is down"
        assert error.error_code == "SERVICE_UNAVAILABLE"
        assert error.service_name is None
        assert error.retry_after is None

    def test_service_unavailable_error_with_details(self):
        """Test service unavailable error with details."""
        error = ServiceUnavailableError("Redis connection failed", service_name="redis", retry_after=30)

        assert error.service_name == "redis"
        assert error.retry_after == 30
        assert error.details["service_name"] == "redis"
        assert error.details["retry_after"] == "30"
