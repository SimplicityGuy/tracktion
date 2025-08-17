"""Custom exceptions for the analysis service."""


class AnalysisServiceError(Exception):
    """Base exception for all analysis service errors."""

    pass


class InvalidAudioFileError(AnalysisServiceError):
    """Raised when an audio file is invalid or cannot be processed."""

    pass


class MetadataExtractionError(AnalysisServiceError):
    """Raised when metadata extraction fails."""

    pass


class StorageError(AnalysisServiceError):
    """Raised when database storage operations fail."""

    pass


class MessageProcessingError(AnalysisServiceError):
    """Raised when message processing fails."""

    pass


class ConnectionError(AnalysisServiceError):
    """Raised when connection to external services fails."""

    pass


class ConfigurationError(AnalysisServiceError):
    """Raised when service configuration is invalid."""

    pass


class UnsupportedFormatError(InvalidAudioFileError):
    """Raised when an audio format is not supported."""

    pass


class CorruptedFileError(InvalidAudioFileError):
    """Raised when an audio file is corrupted."""

    pass


class RetryableError(AnalysisServiceError):
    """Base class for errors that should trigger a retry."""

    pass


class TransientStorageError(StorageError, RetryableError):
    """Raised when a storage error is likely transient and should be retried."""

    pass


class TransientConnectionError(ConnectionError, RetryableError):
    """Raised when a connection error is likely transient and should be retried."""

    pass
