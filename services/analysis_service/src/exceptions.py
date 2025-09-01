"""Custom exceptions for the analysis service."""


class AnalysisServiceError(Exception):
    """Base exception for all analysis service errors."""


class InvalidAudioFileError(AnalysisServiceError):
    """Raised when an audio file is invalid or cannot be processed."""


class MetadataExtractionError(AnalysisServiceError):
    """Raised when metadata extraction fails."""


class StorageError(AnalysisServiceError):
    """Raised when database storage operations fail."""


class MessageProcessingError(AnalysisServiceError):
    """Raised when message processing fails."""


class ConnectionError(AnalysisServiceError):
    """Raised when connection to external services fails."""


class ConfigurationError(AnalysisServiceError):
    """Raised when service configuration is invalid."""


class UnsupportedFormatError(InvalidAudioFileError):
    """Raised when an audio format is not supported."""


class CorruptedFileError(InvalidAudioFileError):
    """Raised when an audio file is corrupted."""


class RetryableError(AnalysisServiceError):
    """Base class for errors that should trigger a retry."""


class TransientStorageError(StorageError, RetryableError):
    """Raised when a storage error is likely transient and should be retried."""


class TransientConnectionError(ConnectionError, RetryableError):
    """Raised when a connection error is likely transient and should be retried."""
