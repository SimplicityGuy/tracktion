"""
CUE file models for tracklist CUE file generation and management.

This module defines the data models for managing CUE files generated from tracklists,
including support for multiple formats and storage strategies.
"""

from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Text, Boolean, Integer, BigInteger
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .tracklist import Base


class CueFormat(str, Enum):
    """Supported CUE file formats."""

    STANDARD = "standard"
    CDJ = "cdj"
    TRAKTOR = "traktor"
    SERATO = "serato"
    REKORDBOX = "rekordbox"
    KODI = "kodi"


class CueGenerationStatus(str, Enum):
    """CUE generation job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationResult(BaseModel):
    """Result of CUE file validation."""

    valid: bool = Field(description="Whether validation passed")
    error: Optional[str] = Field(None, description="Error message if invalid")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    audio_duration: Optional[float] = Field(None, description="Audio duration in seconds")
    tracklist_duration: Optional[float] = Field(None, description="Tracklist duration in seconds")
    metadata: Dict = Field(default_factory=dict, description="Additional validation metadata")


class CueFile(BaseModel):
    """CUE file model."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    tracklist_id: UUID = Field(description="Links to Tracklist")
    file_path: str = Field(description="Storage location")
    format: CueFormat = Field(description="CUE format type")
    file_size: int = Field(description="File size in bytes")
    checksum: str = Field(description="SHA256 hash for integrity")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    version: int = Field(default=1, description="Version number for updates")
    is_active: bool = Field(default=True, description="Current active version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Format-specific metadata")

    model_config = {"json_encoders": {UUID: str}}

    @field_validator("file_size")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Validate file size is reasonable for CUE files."""
        if v < 0:
            raise ValueError("File size cannot be negative")
        if v > 1024 * 1024:  # 1MB limit
            raise ValueError("CUE file size cannot exceed 1MB")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        """Validate version is positive."""
        if v < 1:
            raise ValueError("Version must be positive")
        return v


class CueGenerationJob(BaseModel):
    """CUE generation job model."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    tracklist_id: UUID = Field(description="Target tracklist ID")
    format: CueFormat = Field(description="Target format")
    status: CueGenerationStatus = Field(default=CueGenerationStatus.PENDING, description="Job status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    cue_file_id: Optional[UUID] = Field(None, description="Result CUE file ID")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    validation_report: Optional[ValidationResult] = Field(None, description="Validation report")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")

    model_config = {"json_encoders": {UUID: str}}


class CueFileDB(Base):
    """SQLAlchemy model for CUE file storage."""

    __tablename__ = "cue_files"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    tracklist_id = Column(PostgresUUID(as_uuid=True), ForeignKey("tracklists.id"), nullable=False)
    file_path = Column(Text, nullable=False)
    format = Column(String(20), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA256 hash
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    format_metadata = Column(JSON, nullable=False, default=dict)

    # Relationships
    # tracklist = relationship("TracklistDB", backref="cue_files")

    def to_model(self) -> CueFile:
        """Convert to Pydantic model."""
        return CueFile(
            id=self.id,  # type: ignore[arg-type]
            tracklist_id=self.tracklist_id,  # type: ignore[arg-type]
            file_path=self.file_path,  # type: ignore[arg-type]
            format=CueFormat(self.format),
            file_size=self.file_size,  # type: ignore[arg-type]
            checksum=self.checksum,  # type: ignore[arg-type]
            created_at=self.created_at,  # type: ignore[arg-type]
            updated_at=self.updated_at,  # type: ignore[arg-type]
            version=self.version,  # type: ignore[arg-type]
            is_active=self.is_active,  # type: ignore[arg-type]
            metadata=self.format_metadata if isinstance(self.format_metadata, dict) else {},
        )

    @classmethod
    def from_model(cls, model: CueFile) -> "CueFileDB":
        """Create from Pydantic model."""
        return cls(
            id=model.id,
            tracklist_id=model.tracklist_id,
            file_path=model.file_path,
            format=model.format.value,
            file_size=model.file_size,
            checksum=model.checksum,
            created_at=model.created_at,
            updated_at=model.updated_at,
            version=model.version,
            is_active=model.is_active,
            format_metadata=model.metadata,
        )


class CueGenerationJobDB(Base):
    """SQLAlchemy model for CUE generation job storage."""

    __tablename__ = "cue_generation_jobs"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    tracklist_id = Column(PostgresUUID(as_uuid=True), ForeignKey("tracklists.id"), nullable=False)
    format = Column(String(20), nullable=False)
    status = Column(String(20), default=CueGenerationStatus.PENDING.value, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cue_file_id = Column(PostgresUUID(as_uuid=True), ForeignKey("cue_files.id"), nullable=True)
    error_message = Column(Text, nullable=True)
    validation_report = Column(JSON, nullable=True)
    options = Column(JSON, nullable=False, default=dict)
    progress = Column(Integer, default=0, nullable=False)

    # Relationships
    # tracklist = relationship("TracklistDB", backref="cue_generation_jobs")
    # cue_file = relationship("CueFileDB", backref="generation_job")

    def to_model(self) -> CueGenerationJob:
        """Convert to Pydantic model."""
        validation_report = None
        if self.validation_report:
            try:
                validation_report = ValidationResult(**self.validation_report)
            except Exception:
                # Handle cases where stored data doesn't match current model
                validation_report = None

        return CueGenerationJob(
            id=self.id,  # type: ignore[arg-type]
            tracklist_id=self.tracklist_id,  # type: ignore[arg-type]
            format=CueFormat(self.format),
            status=CueGenerationStatus(self.status),
            created_at=self.created_at,  # type: ignore[arg-type]
            started_at=self.started_at,  # type: ignore[arg-type]
            completed_at=self.completed_at,  # type: ignore[arg-type]
            cue_file_id=self.cue_file_id,  # type: ignore[arg-type]
            error_message=self.error_message,  # type: ignore[arg-type]
            validation_report=validation_report,
            options=self.options if isinstance(self.options, dict) else {},
            progress=self.progress,  # type: ignore[arg-type]
        )

    @classmethod
    def from_model(cls, model: CueGenerationJob) -> "CueGenerationJobDB":
        """Create from Pydantic model."""
        validation_report = None
        if model.validation_report:
            validation_report = model.validation_report.model_dump()

        return cls(
            id=model.id,
            tracklist_id=model.tracklist_id,
            format=model.format.value,
            status=model.status.value,
            created_at=model.created_at,
            started_at=model.started_at,
            completed_at=model.completed_at,
            cue_file_id=model.cue_file_id,
            error_message=model.error_message,
            validation_report=validation_report,
            options=model.options,
            progress=model.progress,
        )


# Request/Response models for API
class GenerateCueRequest(BaseModel):
    """Request model for CUE generation."""

    format: CueFormat = Field(description="Target CUE format")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")
    validate_audio: bool = Field(default=True, description="Whether to validate against audio file")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file for validation")


class BatchGenerateCueRequest(BaseModel):
    """Request model for batch CUE generation."""

    formats: List[CueFormat] = Field(description="Target CUE formats")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")
    validate_audio: bool = Field(default=True, description="Whether to validate against audio files")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file for validation")

    @field_validator("formats")
    @classmethod
    def validate_formats_not_empty(cls, v: List[CueFormat]) -> List[CueFormat]:
        """Validate formats list is not empty."""
        if not v:
            raise ValueError("At least one format must be specified")
        if len(v) > 6:  # All supported formats
            raise ValueError("Too many formats specified")
        return v


class CueGenerationResponse(BaseModel):
    """Response model for CUE generation."""

    success: bool = Field(description="Whether generation was successful")
    job_id: UUID = Field(description="Generation job ID")
    cue_file_id: Optional[UUID] = Field(None, description="Generated CUE file ID")
    file_path: Optional[str] = Field(None, description="CUE file path")
    validation_report: Optional[ValidationResult] = Field(None, description="Validation report")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")

    model_config = {"json_encoders": {UUID: str}}


class BatchCueGenerationResponse(BaseModel):
    """Response model for batch CUE generation."""

    success: bool = Field(description="Overall batch success")
    results: List[CueGenerationResponse] = Field(description="Individual generation results")
    total_files: int = Field(description="Total files requested")
    successful_files: int = Field(description="Successfully generated files")
    failed_files: int = Field(description="Failed generations")
    processing_time_ms: Optional[int] = Field(None, description="Total processing time")

    model_config = {"json_encoders": {UUID: str}}


class ConvertCueRequest(BaseModel):
    """Request model for CUE format conversion."""

    target_format: CueFormat = Field(description="Target format for conversion")
    options: Dict = Field(default_factory=dict, description="Conversion options")
    preserve_metadata: bool = Field(default=True, description="Whether to preserve metadata")


class ConvertCueResponse(BaseModel):
    """Response model for CUE format conversion."""

    success: bool = Field(description="Whether conversion was successful")
    cue_file_id: UUID = Field(description="New CUE file ID")
    file_path: str = Field(description="Converted CUE file path")
    conversion_report: Dict = Field(default_factory=dict, description="Conversion details")
    warnings: List[str] = Field(default_factory=list, description="Conversion warnings")
    error: Optional[str] = Field(None, description="Error message if failed")

    model_config = {"json_encoders": {UUID: str}}
