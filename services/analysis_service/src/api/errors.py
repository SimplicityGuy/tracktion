"""Error handling and custom exceptions for API."""

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from services.analysis_service.src.structured_logging import get_logger

logger = get_logger(__name__)


class APIError(HTTPException):
    """Base API error class."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Initialize API error.

        Args:
            status_code: HTTP status code
            detail: Error message
            error_code: Application-specific error code
            headers: Optional response headers
        """
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(self, resource: str, resource_id: str):
        """Initialize not found error.

        Args:
            resource: Type of resource
            resource_id: ID of resource
        """
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with ID {resource_id} not found",
            error_code="RESOURCE_NOT_FOUND",
        )


class ValidationError(APIError):
    """Request validation error."""

    def __init__(self, detail: str, field: str | None = None):
        """Initialize validation error.

        Args:
            detail: Error details
            field: Field that failed validation
        """
        message = f"Validation error in field '{field}': {detail}" if field else detail
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="VALIDATION_ERROR",
        )


class AuthenticationError(APIError):
    """Authentication error."""

    def __init__(self, detail: str = "Authentication required"):
        """Initialize authentication error.

        Args:
            detail: Error details
        """
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_REQUIRED",
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(APIError):
    """Authorization error."""

    def __init__(self, detail: str = "Insufficient permissions"):
        """Initialize authorization error.

        Args:
            detail: Error details
        """
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="INSUFFICIENT_PERMISSIONS",
        )


class RateLimitError(APIError):
    """Rate limit exceeded error."""

    def __init__(self, retry_after: int = 60):
        """Initialize rate limit error.

        Args:
            retry_after: Seconds until retry allowed
        """
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": str(retry_after)},
        )


class ServiceUnavailableError(APIError):
    """Service unavailable error."""

    def __init__(self, service: str, retry_after: int | None = None):
        """Initialize service unavailable error.

        Args:
            service: Name of unavailable service
            retry_after: Optional seconds until retry
        """
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service '{service}' is temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE",
            headers=headers,
        )


class ConflictError(APIError):
    """Resource conflict error."""

    def __init__(self, detail: str):
        """Initialize conflict error.

        Args:
            detail: Conflict details
        """
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="RESOURCE_CONFLICT",
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle API errors with structured response.

    Args:
        request: Request that caused error
        exc: API error exception

    Returns:
        JSON error response
    """
    request_id = getattr(request.state, "request_id", None)

    # Log error
    logger.error(
        f"API error: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Build error response
    error_response = {
        "error": {
            "code": exc.error_code or "API_ERROR",
            "message": exc.detail,
            "status": exc.status_code,
        }
    }

    if request_id:
        error_response["error"]["request_id"] = request_id

    return JSONResponse(status_code=exc.status_code, content=error_response, headers=exc.headers)


async def validation_exception_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: Request that caused error
        exc: Validation error

    Returns:
        JSON error response
    """
    request_id = getattr(request.state, "request_id", None)

    # Log validation error
    logger.warning(
        "Request validation failed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "errors": str(exc),
        },
    )

    # Parse validation errors
    errors = [
        {
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    # Build error response
    error_response = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "status": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "details": errors,
        }
    }

    if request_id:
        error_response["error"]["request_id"] = request_id

    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: Request that caused error
        exc: Exception

    Returns:
        JSON error response
    """
    request_id = getattr(request.state, "request_id", None)

    # Log unexpected error
    logger.error(
        "Unexpected error occurred",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
        },
        exc_info=True,
    )

    # Build generic error response
    error_response = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
    }

    if request_id:
        error_response["error"]["request_id"] = request_id

    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response)


def register_error_handlers(app: Any) -> None:
    """Register error handlers with FastAPI app.

    Args:
        app: FastAPI application
    """
    # Register custom error handlers
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(PydanticValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Override default HTTP exception handler
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException) -> Any:
        """Custom HTTP exception handler with request ID."""
        request_id = getattr(request.state, "request_id", None)

        # Log HTTP exception
        logger.warning(
            f"HTTP exception: {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        # Add request ID to response
        if request_id:
            headers = dict(exc.headers) if exc.headers else {}
            headers["X-Request-ID"] = request_id
            exc.headers = headers

        return await http_exception_handler(request, exc)
