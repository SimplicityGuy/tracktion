"""
Message schemas for RabbitMQ CUE generation operations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class MessageType(str, Enum):
    """Message types for CUE generation operations."""

    CUE_GENERATION = "cue_generation"
    CUE_GENERATION_COMPLETE = "cue_generation_complete"
    BATCH_CUE_GENERATION = "batch_cue_generation"
    BATCH_CUE_GENERATION_COMPLETE = "batch_cue_generation_complete"
    CUE_VALIDATION = "cue_validation"
    CUE_CONVERSION = "cue_conversion"


class BaseMessage(BaseModel):
    """Base message schema with common fields."""

    message_id: UUID = Field(description="Unique message identifier")
    message_type: MessageType = Field(description="Type of message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[UUID] = Field(None, description="Correlation ID for request tracking")
    retry_count: int = Field(0, description="Number of retry attempts", ge=0, le=5)
    priority: int = Field(5, description="Message priority (1-10)", ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "BaseMessage":
        """Create message from JSON string."""
        return cls.model_validate_json(json_str)


class CueGenerationMessage(BaseMessage):
    """Message schema for single CUE file generation requests."""

    message_type: MessageType = Field(MessageType.CUE_GENERATION, frozen=True)

    # Request data
    tracklist_id: UUID = Field(description="Source tracklist ID")
    format: str = Field(description="Target CUE format")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")
    validate_audio: bool = Field(True, description="Whether to validate against audio file")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file for validation")

    # Job tracking
    job_id: UUID = Field(description="Generation job ID")
    requested_by: Optional[str] = Field(None, description="User or service that requested generation")

    @validator("format")
    def validate_format(cls, v: str) -> str:
        """Validate CUE format."""
        valid_formats = ["standard", "cdj", "traktor", "serato", "rekordbox", "kodi"]
        if v not in valid_formats:
            raise ValueError(f"Invalid CUE format: {v}. Must be one of {valid_formats}")
        return v


class CueGenerationCompleteMessage(BaseMessage):
    """Message schema for CUE generation completion notifications."""

    message_type: MessageType = Field(MessageType.CUE_GENERATION_COMPLETE, frozen=True)

    # Original request reference
    original_message_id: UUID = Field(description="ID of the original generation request message")
    job_id: UUID = Field(description="Generation job ID")
    tracklist_id: UUID = Field(description="Source tracklist ID")

    # Result data
    success: bool = Field(description="Whether generation was successful")
    cue_file_id: Optional[UUID] = Field(None, description="Generated CUE file ID")
    file_path: Optional[str] = Field(None, description="Path to generated CUE file")
    file_size: Optional[int] = Field(None, description="Size of generated file in bytes")
    checksum: Optional[str] = Field(None, description="SHA256 checksum of generated file")

    # Validation results
    validation_report: Optional[Dict[str, Any]] = Field(None, description="Validation results")

    # Error information
    error: Optional[str] = Field(None, description="Error message if generation failed")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")

    # Performance metrics
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    queue_time_ms: Optional[float] = Field(None, description="Time spent in queue in milliseconds")


class BatchCueGenerationMessage(BaseMessage):
    """Message schema for batch CUE file generation requests."""

    message_type: MessageType = Field(MessageType.BATCH_CUE_GENERATION, frozen=True)

    # Request data
    tracklist_id: UUID = Field(description="Source tracklist ID")
    formats: List[str] = Field(description="List of target CUE formats")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")
    validate_audio: bool = Field(False, description="Whether to validate against audio file")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file for validation")

    # Job tracking
    batch_job_id: UUID = Field(description="Batch generation job ID")
    requested_by: Optional[str] = Field(None, description="User or service that requested generation")

    @validator("formats")
    def validate_formats(cls, v: List[str]) -> List[str]:
        """Validate CUE formats."""
        if not v:
            raise ValueError("At least one format must be specified")
        if len(v) > 6:
            raise ValueError("Too many formats specified (maximum 6)")

        valid_formats = {"standard", "cdj", "traktor", "serato", "rekordbox", "kodi"}
        for format_name in v:
            if format_name not in valid_formats:
                raise ValueError(f"Invalid CUE format: {format_name}")

        return v


class BatchCueGenerationCompleteMessage(BaseMessage):
    """Message schema for batch CUE generation completion notifications."""

    message_type: MessageType = Field(MessageType.BATCH_CUE_GENERATION_COMPLETE, frozen=True)

    # Original request reference
    original_message_id: UUID = Field(description="ID of the original batch generation request message")
    batch_job_id: UUID = Field(description="Batch generation job ID")
    tracklist_id: UUID = Field(description="Source tracklist ID")

    # Result summary
    total_files: int = Field(description="Total number of files requested")
    successful_files: int = Field(description="Number of successfully generated files")
    failed_files: int = Field(description="Number of failed file generations")

    # Detailed results
    results: List[Dict[str, Any]] = Field(description="Individual generation results")

    # Overall status
    success: bool = Field(description="Whether batch generation was overall successful")
    error: Optional[str] = Field(None, description="Overall error message if batch failed")

    # Performance metrics
    total_processing_time_ms: Optional[float] = Field(None, description="Total processing time")
    average_processing_time_ms: Optional[float] = Field(None, description="Average per-file processing time")


class CueValidationMessage(BaseMessage):
    """Message schema for CUE file validation requests."""

    message_type: MessageType = Field(MessageType.CUE_VALIDATION, frozen=True)

    # Request data
    cue_file_id: UUID = Field(description="CUE file ID to validate")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file for validation")
    validation_options: Dict[str, Any] = Field(default_factory=dict, description="Validation options")

    # Job tracking
    validation_job_id: UUID = Field(description="Validation job ID")
    requested_by: Optional[str] = Field(None, description="User or service that requested validation")


class CueConversionMessage(BaseMessage):
    """Message schema for CUE file format conversion requests."""

    message_type: MessageType = Field(MessageType.CUE_CONVERSION, frozen=True)

    # Request data
    source_cue_file_id: UUID = Field(description="Source CUE file ID")
    target_format: str = Field(description="Target CUE format")
    preserve_metadata: bool = Field(True, description="Whether to preserve metadata during conversion")
    conversion_options: Dict[str, Any] = Field(default_factory=dict, description="Conversion options")

    # Job tracking
    conversion_job_id: UUID = Field(description="Conversion job ID")
    requested_by: Optional[str] = Field(None, description="User or service that requested conversion")

    @validator("target_format")
    def validate_target_format(cls, v: str) -> str:
        """Validate target CUE format."""
        valid_formats = ["standard", "cdj", "traktor", "serato", "rekordbox", "kodi"]
        if v not in valid_formats:
            raise ValueError(f"Invalid target CUE format: {v}. Must be one of {valid_formats}")
        return v


class MessageBatch(BaseModel):
    """Container for multiple messages to be sent as a batch."""

    batch_id: UUID = Field(description="Unique batch identifier")
    messages: List[BaseMessage] = Field(description="List of messages in the batch")
    batch_size: int = Field(description="Number of messages in the batch")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator("batch_size")
    def validate_batch_size(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate batch size matches message count."""
        messages = values.get("messages", [])
        if v != len(messages):
            raise ValueError("Batch size must match number of messages")
        return v

    def to_json(self) -> str:
        """Convert batch to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "MessageBatch":
        """Create batch from JSON string."""
        return cls.model_validate_json(json_str)


# Message routing configuration
MESSAGE_ROUTING: Dict[MessageType, Dict[str, Any]] = {
    MessageType.CUE_GENERATION: {
        "queue": "cue.generation",
        "routing_key": "cue.generation.single",
        "exchange": "cue.direct",
        "durable": True,
    },
    MessageType.CUE_GENERATION_COMPLETE: {
        "queue": "cue.generation.complete",
        "routing_key": "cue.generation.complete",
        "exchange": "cue.direct",
        "durable": True,
    },
    MessageType.BATCH_CUE_GENERATION: {
        "queue": "cue.generation.batch",
        "routing_key": "cue.generation.batch",
        "exchange": "cue.direct",
        "durable": True,
    },
    MessageType.BATCH_CUE_GENERATION_COMPLETE: {
        "queue": "cue.generation.batch.complete",
        "routing_key": "cue.generation.batch.complete",
        "exchange": "cue.direct",
        "durable": True,
    },
    MessageType.CUE_VALIDATION: {
        "queue": "cue.validation",
        "routing_key": "cue.validation",
        "exchange": "cue.direct",
        "durable": True,
    },
    MessageType.CUE_CONVERSION: {
        "queue": "cue.conversion",
        "routing_key": "cue.conversion",
        "exchange": "cue.direct",
        "durable": True,
    },
}
