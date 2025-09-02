"""Comprehensive error handling for Analysis Service API."""

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import IntegrityError, OperationalError

from services.analysis_service.src.structured_logging import get_logger

logger = get_logger(__name__)


class AnalysisServiceError(Exception):
    """Base exception for Analysis Service."""

    def __init__(self, message: str, error_code: str = "ANALYSIS_SERVICE_ERROR") -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class RecordingNotFoundError(AnalysisServiceError):
    """Recording not found error."""

    def __init__(self, recording_id: str) -> None:
        super().__init__(f"Recording not found: {recording_id}", "RECORDING_NOT_FOUND")


class FileNotFoundError(AnalysisServiceError):
    """Audio file not found error."""

    def __init__(self, file_path: str) -> None:
        super().__init__(f"Audio file not found: {file_path}", "FILE_NOT_FOUND")


class FileAccessError(AnalysisServiceError):
    """File access error."""

    def __init__(self, file_path: str, operation: str) -> None:
        super().__init__(f"Cannot {operation} file: {file_path}", "FILE_ACCESS_ERROR")


class DatabaseError(AnalysisServiceError):
    """Database operation error."""

    def __init__(self, operation: str, details: str = "") -> None:
        message = f"Database {operation} failed"
        if details:
            message += f": {details}"
        super().__init__(message, "DATABASE_ERROR")


class MessageQueueError(AnalysisServiceError):
    """Message queue operation error."""

    def __init__(self, operation: str, details: str = "") -> None:
        message = f"Message queue {operation} failed"
        if details:
            message += f": {details}"
        super().__init__(message, "MESSAGE_QUEUE_ERROR")


class AnalysisError(AnalysisServiceError):
    """Analysis operation error."""

    def __init__(self, analysis_type: str, details: str = "") -> None:
        message = f"{analysis_type} analysis failed"
        if details:
            message += f": {details}"
        super().__init__(message, "ANALYSIS_ERROR")


class RequestValidationError(AnalysisServiceError):
    """Request validation error."""

    def __init__(self, field: str, details: str) -> None:
        super().__init__(f"Validation error for {field}: {details}", "VALIDATION_ERROR")


# Error response formatters
def create_error_response(
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> JSONResponse:
    """Create standardized error response.

    Args:
        error_code: Unique error code
        message: Human-readable error message
        details: Additional error details
        status_code: HTTP status code

    Returns:
        JSONResponse with error information
    """
    response_data: dict[str, Any] = {
        "error": {
            "code": error_code,
            "message": message,
        }
    }

    if details:
        response_data["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=response_data,
    )


# Exception handlers
async def analysis_service_error_handler(request: Request, exc: AnalysisServiceError) -> JSONResponse:
    """Handle Analysis Service specific errors."""
    logger.error(
        "Analysis service error",
        extra={
            "error_code": exc.error_code,
            "message": exc.message,
            "path": str(request.url),
        },
    )

    # Map error types to HTTP status codes
    status_map = {
        "RECORDING_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "FILE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "FILE_ACCESS_ERROR": status.HTTP_403_FORBIDDEN,
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "DATABASE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "MESSAGE_QUEUE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "ANALYSIS_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
    }

    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=status_map.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR),
    )


async def validation_error_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        "Validation error",
        extra={
            "path": str(request.url),
            "errors": str(exc),
        },
    )

    return create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"validation_errors": str(exc)},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle database-related errors."""
    error_message = "Database operation failed"
    error_code = "DATABASE_ERROR"

    if isinstance(exc, IntegrityError):
        error_message = "Database constraint violation"
        error_code = "DATABASE_CONSTRAINT_ERROR"
    elif isinstance(exc, OperationalError):
        error_message = "Database connection error"
        error_code = "DATABASE_CONNECTION_ERROR"

    logger.error(
        "Database error",
        extra={
            "error_code": error_code,
            "message": str(exc),
            "path": str(request.url),
        },
    )

    return create_error_response(
        error_code=error_code,
        message=error_message,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


async def file_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle file system errors."""
    error_message = "File operation failed"
    error_code = "FILE_ERROR"

    if isinstance(exc, FileNotFoundError):
        error_message = str(exc)
        error_code = "FILE_NOT_FOUND"
    elif isinstance(exc, PermissionError):
        error_message = "File access denied"
        error_code = "FILE_ACCESS_DENIED"
    elif isinstance(exc, OSError):
        error_message = "File system error"
        error_code = "FILE_SYSTEM_ERROR"

    logger.error(
        "File error",
        extra={
            "error_code": error_code,
            "message": str(exc),
            "path": str(request.url),
        },
    )

    return create_error_response(
        error_code=error_code,
        message=error_message,
        status_code=status.HTTP_404_NOT_FOUND
        if error_code == "FILE_NOT_FOUND"
        else status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected errors."""
    logger.exception(
        "Unexpected error",
        extra={
            "path": str(request.url),
            "error_type": type(exc).__name__,
        },
    )

    return create_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# Error handler registry
def register_error_handlers(app: Any) -> None:
    """Register all error handlers with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Analysis Service specific errors
    app.add_exception_handler(AnalysisServiceError, analysis_service_error_handler)

    # Validation errors
    app.add_exception_handler(PydanticValidationError, validation_error_handler)

    # Database errors
    app.add_exception_handler(IntegrityError, database_error_handler)
    app.add_exception_handler(OperationalError, database_error_handler)

    # File system errors
    app.add_exception_handler(FileNotFoundError, file_error_handler)
    app.add_exception_handler(PermissionError, file_error_handler)
    app.add_exception_handler(OSError, file_error_handler)

    # Generic error handler (catch-all)
    app.add_exception_handler(Exception, generic_error_handler)

    logger.info("Error handlers registered successfully")
