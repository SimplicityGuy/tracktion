"""Comprehensive unit tests for error handlers."""

from unittest.mock import Mock, patch

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import IntegrityError, OperationalError

from services.analysis_service.src.api.error_handlers import (
    AnalysisError,
    # Custom exceptions
    AnalysisServiceError,
    DatabaseError,
    FileAccessError,
    FileNotFoundError,
    MessageQueueError,
    RecordingNotFoundError,
    RequestValidationError,
    # Error handlers
    analysis_service_error_handler,
    # Response formatter
    create_error_response,
    database_error_handler,
    file_error_handler,
    generic_error_handler,
    # Registration function
    register_error_handlers,
    validation_error_handler,
)


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_analysis_service_error_base(self):
        """Test base AnalysisServiceError."""
        error = AnalysisServiceError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code == "ANALYSIS_SERVICE_ERROR"

    def test_analysis_service_error_custom_code(self):
        """Test AnalysisServiceError with custom error code."""
        error = AnalysisServiceError("Custom error", "CUSTOM_ERROR_CODE")

        assert error.message == "Custom error"
        assert error.error_code == "CUSTOM_ERROR_CODE"

    def test_recording_not_found_error(self):
        """Test RecordingNotFoundError."""
        recording_id = "550e8400-e29b-41d4-a716-446655440000"
        error = RecordingNotFoundError(recording_id)

        assert f"Recording not found: {recording_id}" in error.message
        assert error.error_code == "RECORDING_NOT_FOUND"
        assert isinstance(error, AnalysisServiceError)

    def test_file_not_found_error(self):
        """Test FileNotFoundError."""
        file_path = "/path/to/missing/file.wav"
        error = FileNotFoundError(file_path)

        assert f"Audio file not found: {file_path}" in error.message
        assert error.error_code == "FILE_NOT_FOUND"
        assert isinstance(error, AnalysisServiceError)

    def test_file_access_error(self):
        """Test FileAccessError."""
        file_path = "/path/to/restricted/file.wav"
        operation = "read"
        error = FileAccessError(file_path, operation)

        assert f"Cannot {operation} file: {file_path}" in error.message
        assert error.error_code == "FILE_ACCESS_ERROR"
        assert isinstance(error, AnalysisServiceError)

    def test_database_error(self):
        """Test DatabaseError."""
        operation = "create"
        details = "Connection timeout"
        error = DatabaseError(operation, details)

        assert f"Database {operation} failed: {details}" in error.message
        assert error.error_code == "DATABASE_ERROR"
        assert isinstance(error, AnalysisServiceError)

    def test_database_error_no_details(self):
        """Test DatabaseError without details."""
        operation = "update"
        error = DatabaseError(operation)

        assert f"Database {operation} failed" in error.message
        assert ":" not in error.message  # No details separator

    def test_message_queue_error(self):
        """Test MessageQueueError."""
        operation = "publish"
        details = "RabbitMQ connection failed"
        error = MessageQueueError(operation, details)

        assert f"Message queue {operation} failed: {details}" in error.message
        assert error.error_code == "MESSAGE_QUEUE_ERROR"
        assert isinstance(error, AnalysisServiceError)

    def test_message_queue_error_no_details(self):
        """Test MessageQueueError without details."""
        operation = "consume"
        error = MessageQueueError(operation)

        assert f"Message queue {operation} failed" in error.message
        assert ":" not in error.message

    def test_analysis_error(self):
        """Test AnalysisError."""
        analysis_type = "BPM"
        details = "Audio format not supported"
        error = AnalysisError(analysis_type, details)

        assert f"{analysis_type} analysis failed: {details}" in error.message
        assert error.error_code == "ANALYSIS_ERROR"
        assert isinstance(error, AnalysisServiceError)

    def test_analysis_error_no_details(self):
        """Test AnalysisError without details."""
        analysis_type = "key detection"
        error = AnalysisError(analysis_type)

        assert f"{analysis_type} analysis failed" in error.message
        assert ":" not in error.message

    def test_request_validation_error(self):
        """Test RequestValidationError."""
        field = "recording_id"
        details = "Invalid UUID format"
        error = RequestValidationError(field, details)

        assert f"Validation error for {field}: {details}" in error.message
        assert error.error_code == "VALIDATION_ERROR"
        assert isinstance(error, AnalysisServiceError)


class TestErrorResponseFormatter:
    """Test create_error_response function."""

    def test_create_error_response_basic(self):
        """Test basic error response creation."""
        response = create_error_response(
            error_code="TEST_ERROR",
            message="Test error message",
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        # Parse response content
        content = response.body.decode()
        assert "TEST_ERROR" in content
        assert "Test error message" in content

    def test_create_error_response_with_details(self):
        """Test error response with details."""
        details = {"field": "value", "context": "test"}
        response = create_error_response(
            error_code="DETAILED_ERROR",
            message="Error with details",
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        content = response.body.decode()
        assert "DETAILED_ERROR" in content
        assert "Error with details" in content
        assert "field" in content
        assert "value" in content

    def test_create_error_response_custom_status(self):
        """Test error response with custom status code."""
        response = create_error_response(
            error_code="CUSTOM_ERROR",
            message="Custom error",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_error_response_no_details(self):
        """Test error response without details."""
        response = create_error_response(
            error_code="SIMPLE_ERROR",
            message="Simple error",
            details=None,
        )

        content = response.body.decode()
        assert "SIMPLE_ERROR" in content
        assert "Simple error" in content
        # Should not contain details field
        assert '"details"' not in content


class TestErrorHandlers:
    """Test error handler functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_request = Mock(spec=Request)
        self.mock_request.url = "http://test.com/api/test"

    @pytest.mark.asyncio
    async def test_analysis_service_error_handler(self):
        """Test analysis service error handler."""
        error = RecordingNotFoundError("test-recording-id")

        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            response = await analysis_service_error_handler(self.mock_request, error)

            # Verify logging
            mock_logger.error.assert_called_once()
            log_args = mock_logger.error.call_args
            assert "Analysis service error" in log_args[0][0]
            assert log_args[1]["extra"]["error_code"] == "RECORDING_NOT_FOUND"
            assert "test-recording-id" in log_args[1]["extra"]["message"]

            # Verify response
            assert isinstance(response, JSONResponse)
            assert response.status_code == status.HTTP_404_NOT_FOUND

            content = response.body.decode()
            assert "RECORDING_NOT_FOUND" in content
            assert "test-recording-id" in content

    @pytest.mark.asyncio
    async def test_analysis_service_error_handler_status_mapping(self):
        """Test status code mapping for different error types."""
        test_cases = [
            (RecordingNotFoundError("id"), status.HTTP_404_NOT_FOUND),
            (FileNotFoundError("/path"), status.HTTP_404_NOT_FOUND),
            (FileAccessError("/path", "read"), status.HTTP_403_FORBIDDEN),
            (RequestValidationError("field", "details"), status.HTTP_400_BAD_REQUEST),
            (DatabaseError("operation", "details"), status.HTTP_503_SERVICE_UNAVAILABLE),
            (MessageQueueError("operation", "details"), status.HTTP_503_SERVICE_UNAVAILABLE),
            (AnalysisError("type", "details"), status.HTTP_422_UNPROCESSABLE_ENTITY),
            (AnalysisServiceError("generic", "UNKNOWN_ERROR"), status.HTTP_500_INTERNAL_SERVER_ERROR),
        ]

        for error, expected_status in test_cases:
            with patch("services.analysis_service.src.api.error_handlers.logger"):
                response = await analysis_service_error_handler(self.mock_request, error)
                assert response.status_code == expected_status, f"Failed for {type(error).__name__}"

    @pytest.mark.asyncio
    async def test_validation_error_handler(self):
        """Test validation error handler."""
        # Create a mock ValidationError
        mock_validation_error = Mock(spec=PydanticValidationError)
        mock_validation_error.__str__ = Mock(return_value="Validation failed: field required")

        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            response = await validation_error_handler(self.mock_request, mock_validation_error)

            # Verify logging
            mock_logger.warning.assert_called_once()
            log_args = mock_logger.warning.call_args
            assert "Validation error" in log_args[0][0]
            assert "Validation failed: field required" in log_args[1]["extra"]["errors"]

            # Verify response
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            content = response.body.decode()
            assert "VALIDATION_ERROR" in content
            assert "Request validation failed" in content

    @pytest.mark.asyncio
    async def test_database_error_handler_integrity_error(self):
        """Test database error handler with IntegrityError."""
        integrity_error = IntegrityError("statement", "params", "orig")

        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            response = await database_error_handler(self.mock_request, integrity_error)

            # Verify logging
            mock_logger.error.assert_called_once()
            log_args = mock_logger.error.call_args
            assert log_args[1]["extra"]["error_code"] == "DATABASE_CONSTRAINT_ERROR"

            # Verify response
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            content = response.body.decode()
            assert "DATABASE_CONSTRAINT_ERROR" in content
            assert "Database constraint violation" in content

    @pytest.mark.asyncio
    async def test_database_error_handler_operational_error(self):
        """Test database error handler with OperationalError."""
        operational_error = OperationalError("statement", "params", "orig")

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await database_error_handler(self.mock_request, operational_error)

            # Verify response
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            content = response.body.decode()
            assert "DATABASE_CONNECTION_ERROR" in content
            assert "Database connection error" in content

    @pytest.mark.asyncio
    async def test_database_error_handler_generic(self):
        """Test database error handler with generic exception."""
        generic_error = Exception("Generic database error")

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await database_error_handler(self.mock_request, generic_error)

            # Verify response
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            content = response.body.decode()
            assert "DATABASE_ERROR" in content
            assert "Database operation failed" in content

    @pytest.mark.asyncio
    async def test_file_error_handler_file_not_found(self):
        """Test file error handler with FileNotFoundError."""
        file_error = FileNotFoundError("File not found")

        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            response = await file_error_handler(self.mock_request, file_error)

            # Verify logging
            mock_logger.error.assert_called_once()
            log_args = mock_logger.error.call_args
            assert log_args[1]["extra"]["error_code"] == "FILE_NOT_FOUND"

            # Verify response
            assert response.status_code == status.HTTP_404_NOT_FOUND
            content = response.body.decode()
            assert "FILE_NOT_FOUND" in content

    @pytest.mark.asyncio
    async def test_file_error_handler_permission_error(self):
        """Test file error handler with PermissionError."""
        permission_error = PermissionError("Permission denied")

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await file_error_handler(self.mock_request, permission_error)

            # Verify response
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            content = response.body.decode()
            assert "FILE_ACCESS_DENIED" in content
            assert "File access denied" in content

    @pytest.mark.asyncio
    async def test_file_error_handler_os_error(self):
        """Test file error handler with OSError."""
        os_error = OSError("OS file system error")

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await file_error_handler(self.mock_request, os_error)

            # Verify response
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            content = response.body.decode()
            assert "FILE_SYSTEM_ERROR" in content
            assert "File system error" in content

    @pytest.mark.asyncio
    async def test_file_error_handler_generic(self):
        """Test file error handler with generic exception."""
        generic_error = Exception("Generic file error")

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await file_error_handler(self.mock_request, generic_error)

            # Verify response
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            content = response.body.decode()
            assert "FILE_ERROR" in content
            assert "File operation failed" in content

    @pytest.mark.asyncio
    async def test_generic_error_handler(self):
        """Test generic error handler."""
        generic_error = Exception("Unexpected error")

        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            response = await generic_error_handler(self.mock_request, generic_error)

            # Verify logging with exception
            mock_logger.exception.assert_called_once()
            log_args = mock_logger.exception.call_args
            assert "Unexpected error" in log_args[0][0]
            assert log_args[1]["extra"]["error_type"] == "Exception"

            # Verify response
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            content = response.body.decode()
            assert "INTERNAL_SERVER_ERROR" in content
            assert "An unexpected error occurred" in content


class TestErrorHandlerRegistration:
    """Test error handler registration function."""

    def test_register_error_handlers(self):
        """Test registering all error handlers with FastAPI app."""
        mock_app = Mock()

        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            register_error_handlers(mock_app)

            # Verify all exception handlers were registered
            expected_calls = [
                # Analysis Service specific errors
                (AnalysisServiceError, analysis_service_error_handler),
                # Validation errors
                (PydanticValidationError, validation_error_handler),
                # Database errors
                (IntegrityError, database_error_handler),
                (OperationalError, database_error_handler),
                # File system errors
                (FileNotFoundError, file_error_handler),
                (PermissionError, file_error_handler),
                (OSError, file_error_handler),
                # Generic error handler (catch-all)
                (Exception, generic_error_handler),
            ]

            # Verify add_exception_handler was called for each
            assert mock_app.add_exception_handler.call_count == len(expected_calls)

            # Verify the calls were made with correct arguments
            actual_calls = mock_app.add_exception_handler.call_args_list
            for i, (exception_type, handler_func) in enumerate(expected_calls):
                call_args = actual_calls[i][0]  # Get positional arguments
                assert call_args[0] == exception_type
                assert call_args[1] == handler_func

            # Verify success logging
            mock_logger.info.assert_called_once_with("Error handlers registered successfully")

    def test_register_error_handlers_preserves_order(self):
        """Test that error handlers are registered in correct order."""
        mock_app = Mock()

        register_error_handlers(mock_app)

        # Get all the exception types that were registered
        registered_exceptions = [call[0][0] for call in mock_app.add_exception_handler.call_args_list]

        # Verify specific exceptions come before generic Exception handler
        analysis_service_index = registered_exceptions.index(AnalysisServiceError)
        generic_exception_index = registered_exceptions.index(Exception)

        assert analysis_service_index < generic_exception_index, (
            "AnalysisServiceError should be registered before generic Exception handler"
        )

        # Verify Exception handler is last
        assert registered_exceptions[-1] is Exception, "Exception handler should be registered last as catch-all"


class TestErrorHandlerIntegration:
    """Test error handlers with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_request = Mock(spec=Request)
        self.mock_request.url = "http://test.com/api/analysis/123"

    @pytest.mark.asyncio
    async def test_error_handler_chain(self):
        """Test that specific error handlers take precedence over generic ones."""
        # Test that RecordingNotFoundError is handled by analysis_service_error_handler
        recording_error = RecordingNotFoundError("test-id")

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await analysis_service_error_handler(self.mock_request, recording_error)

        # Should get 404 for recording not found, not 500 for generic error
        assert response.status_code == status.HTTP_404_NOT_FOUND
        content = response.body.decode()
        assert "RECORDING_NOT_FOUND" in content

    @pytest.mark.asyncio
    async def test_error_response_format_consistency(self):
        """Test that all error handlers produce consistent response format."""
        # Create a mock ValidationError
        mock_validation_error = Mock(spec=PydanticValidationError)
        mock_validation_error.__str__ = Mock(return_value="Validation failed")

        test_cases = [
            (RecordingNotFoundError("test-id"), analysis_service_error_handler),
            (mock_validation_error, validation_error_handler),
            (IntegrityError("stmt", "params", "orig"), database_error_handler),
            (FileNotFoundError("test file"), file_error_handler),
            (Exception("generic error"), generic_error_handler),
        ]

        for error, handler in test_cases:
            with patch("services.analysis_service.src.api.error_handlers.logger"):
                response = await handler(self.mock_request, error)

                # All responses should be JSONResponse
                assert isinstance(response, JSONResponse)

                # All responses should have proper status codes
                assert 400 <= response.status_code < 600

                # All responses should have consistent JSON structure
                content = response.body.decode()
                assert '"error"' in content
                assert '"code"' in content
                assert '"message"' in content

    @pytest.mark.asyncio
    async def test_error_logging_consistency(self):
        """Test that all error handlers log consistently."""
        with patch("services.analysis_service.src.api.error_handlers.logger") as mock_logger:
            # Test analysis service error logging
            error = DatabaseError("test operation", "test details")
            await analysis_service_error_handler(self.mock_request, error)

            # Verify structured logging
            log_call = mock_logger.error.call_args
            assert "extra" in log_call[1]
            extra = log_call[1]["extra"]
            assert "error_code" in extra
            assert "message" in extra
            assert "path" in extra
            assert str(self.mock_request.url) in extra["path"]


class TestErrorHandlerEdgeCases:
    """Test edge cases and error conditions in error handlers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_request = Mock(spec=Request)
        self.mock_request.url = "http://test.com/api/test"

    @pytest.mark.asyncio
    async def test_error_handler_with_none_details(self):
        """Test error handlers when details are None."""
        error = DatabaseError("operation")  # No details

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await analysis_service_error_handler(self.mock_request, error)

        content = response.body.decode()
        # Should not have trailing colon when no details
        assert "Database operation failed:" not in content
        assert "Database operation failed" in content

    @pytest.mark.asyncio
    async def test_error_handler_with_empty_string_details(self):
        """Test error handlers when details are empty string."""
        error = MessageQueueError("publish", "")  # Empty details

        with patch("services.analysis_service.src.api.error_handlers.logger"):
            response = await analysis_service_error_handler(self.mock_request, error)

        content = response.body.decode()
        # Empty details should be handled gracefully
        assert "MESSAGE_QUEUE_ERROR" in content

    @pytest.mark.asyncio
    async def test_error_response_with_complex_details(self):
        """Test error response creation with complex details object."""
        complex_details = {
            "validation_errors": [
                {"field": "recording_id", "message": "Invalid UUID"},
                {"field": "priority", "message": "Must be between 1 and 10"},
            ],
            "request_data": {"some": "data"},
            "timestamp": "2024-01-01T00:00:00Z",
        }

        response = create_error_response(
            error_code="COMPLEX_VALIDATION_ERROR",
            message="Multiple validation failures",
            details=complex_details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

        content = response.body.decode()
        assert "COMPLEX_VALIDATION_ERROR" in content
        assert "Multiple validation failures" in content
        assert "validation_errors" in content
        assert "Invalid UUID" in content

    def test_custom_exception_inheritance(self):
        """Test that all custom exceptions properly inherit from AnalysisServiceError."""
        custom_exceptions = [
            RecordingNotFoundError("test"),
            FileNotFoundError("/path"),
            FileAccessError("/path", "read"),
            DatabaseError("op"),
            MessageQueueError("op"),
            AnalysisError("type"),
            RequestValidationError("field", "details"),
        ]

        for exc in custom_exceptions:
            assert isinstance(exc, AnalysisServiceError)
            assert isinstance(exc, Exception)
            assert hasattr(exc, "message")
            assert hasattr(exc, "error_code")
            assert exc.message is not None
            assert exc.error_code is not None
