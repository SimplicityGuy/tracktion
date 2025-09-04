"""Custom exceptions for the analysis service."""


class AnalysisServiceError(Exception):
    """Base exception for all analysis service errors.

    This is the root exception class for all custom exceptions in the
    analysis service. All service-specific exceptions should inherit
    from this class to allow for consistent error handling.

    Example:
        >>> try:
        ...     # some analysis operation
        ...     pass
        ... except AnalysisServiceError as e:
        ...     logger.error(f"Analysis service error: {e}")
    """


class InvalidAudioFileError(AnalysisServiceError):
    """Raised when an audio file is invalid or cannot be processed.

    This exception is raised when the analysis service encounters an audio file
    that cannot be processed due to corruption, unsupported format, or other
    file-related issues.

    Example:
        >>> detector = BPMDetector()
        >>> try:
        ...     detector.detect_bpm("corrupted.mp3")
        ... except InvalidAudioFileError as e:
        ...     logger.error(f"Invalid audio file: {e}")
    """


class MetadataExtractionError(AnalysisServiceError):
    """Raised when metadata extraction fails.

    This exception occurs when the metadata extractor cannot read or parse
    metadata from an audio file, typically due to missing tags, corrupted
    metadata, or unsupported metadata formats.

    Example:
        >>> extractor = MetadataExtractor()
        >>> try:
        ...     metadata = extractor.extract("file.mp3")
        ... except MetadataExtractionError as e:
        ...     logger.error(f"Metadata extraction failed: {e}")
    """


class StorageError(AnalysisServiceError):
    """Raised when database storage operations fail.

    This exception is thrown when the storage handler encounters errors
    during database operations such as saving metadata, updating recording
    status, or querying data.

    Example:
        >>> storage = StorageHandler()
        >>> try:
        ...     storage.store_metadata(recording_id, metadata)
        ... except StorageError as e:
        ...     logger.error(f"Storage operation failed: {e}")
    """


class MessageProcessingError(AnalysisServiceError):
    """Raised when message processing fails.

    This exception occurs when the message consumer encounters errors
    while processing incoming messages from the message queue, such as
    invalid message format, missing required fields, or processing failures.

    Example:
        >>> consumer = MessageConsumer()
        >>> try:
        ...     consumer.process_message(invalid_message)
        ... except MessageProcessingError as e:
        ...     logger.error(f"Message processing failed: {e}")
    """


class ConnectionError(AnalysisServiceError):
    """Raised when connection to external services fails.

    This exception is raised when the service cannot establish or maintain
    connections to external dependencies like RabbitMQ, Redis, or database
    servers.

    Example:
        >>> try:
        ...     connect_to_rabbitmq()
        ... except ConnectionError as e:
        ...     logger.error(f"Failed to connect to RabbitMQ: {e}")
    """


class ConfigurationError(AnalysisServiceError):
    """Raised when service configuration is invalid.

    This exception occurs when the service configuration contains invalid
    values, missing required settings, or inconsistent parameter combinations.

    Example:
        >>> config = ServiceConfig.from_env()
        >>> errors = config.validate()
        >>> if errors:
        ...     raise ConfigurationError(f"Invalid configuration: {errors}")
    """


class UnsupportedFormatError(InvalidAudioFileError):
    """Raised when an audio format is not supported.

    This specific exception is raised when the analysis service encounters
    an audio file with a format that is not supported by the current
    configuration or available codecs.

    Example:
        >>> detector = BPMDetector()
        >>> try:
        ...     detector.detect_bpm("file.xyz")
        ... except UnsupportedFormatError as e:
        ...     logger.warning(f"Unsupported format: {e}")
    """


class CorruptedFileError(InvalidAudioFileError):
    """Raised when an audio file is corrupted.

    This specific exception is raised when the analysis service detects
    that an audio file is corrupted and cannot be read or processed.

    Example:
        >>> try:
        ...     analyze_audio("corrupted.mp3")
        ... except CorruptedFileError as e:
        ...     logger.error(f"Corrupted file detected: {e}")
    """


class RetryableError(AnalysisServiceError):
    """Base class for errors that should trigger a retry.

    This exception class is used to mark errors that are likely transient
    and should be retried by the message processing system. Errors that
    inherit from this class will trigger the retry mechanism.

    Example:
        >>> try:
        ...     process_file()
        ... except RetryableError:
        ...     # This will trigger automatic retry
        ...     raise
    """


class TransientStorageError(StorageError, RetryableError):
    """Raised when a storage error is likely transient and should be retried.

    This exception combines StorageError and RetryableError to indicate
    that a storage operation failed due to a transient issue (e.g., network
    timeout, temporary database unavailability) and should be retried.

    Example:
        >>> try:
        ...     storage.save_data(data)
        ... except TransientStorageError as e:
        ...     logger.warning(f"Transient storage error: {e}")
        ...     # Will be automatically retried
        ...     raise
    """


class TransientConnectionError(ConnectionError, RetryableError):
    """Raised when a connection error is likely transient and should be retried.

    This exception combines ConnectionError and RetryableError to indicate
    that a connection failure was likely due to a transient issue (e.g.,
    network hiccup, temporary service unavailability) and should be retried.

    Example:
        >>> try:
        ...     connect_to_service()
        ... except TransientConnectionError as e:
        ...     logger.warning(f"Transient connection error: {e}")
        ...     # Will be automatically retried
        ...     raise
    """
