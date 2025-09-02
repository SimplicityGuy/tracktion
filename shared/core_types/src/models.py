"""SQLAlchemy models for Tracktion data entities."""

import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import ARRAY, DECIMAL, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


class Recording(Base):
    """Model for music recording files."""

    __tablename__ = "recordings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sha256_hash: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    xxh128_hash: Mapped[str | None] = mapped_column(String(32), unique=True, nullable=True)
    processing_status: Mapped[str | None] = mapped_column(String(50), nullable=True, default="pending")
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(UTC),
        nullable=True,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    # Relationships
    metadata_items: Mapped[list["Metadata"]] = relationship(
        "Metadata", back_populates="recording", cascade="all, delete-orphan"
    )
    tracklist: Mapped[Optional["Tracklist"]] = relationship(
        "Tracklist",
        back_populates="recording",
        uselist=False,
        cascade="all, delete-orphan",
    )
    rename_proposals: Mapped[list["RenameProposal"]] = relationship(
        "RenameProposal", back_populates="recording", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """String representation of Recording."""
        return f"<Recording(id={self.id}, file_name={self.file_name})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert Recording to dictionary.

        Returns:
            Dictionary representation of the recording
        """
        return {
            "id": str(self.id),
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "sha256_hash": self.sha256_hash,
            "xxh128_hash": self.xxh128_hash,
            "processing_status": self.processing_status,
            "processing_error": self.processing_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
        }


class Metadata(Base):
    """Model for recording metadata key-value pairs."""

    __tablename__ = "metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="metadata_items")

    def __repr__(self) -> str:
        """String representation of Metadata."""
        return f"<Metadata(id={self.id}, key={self.key}, value={self.value})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert Metadata to dictionary.

        Returns:
            Dictionary representation of the metadata
        """
        return {
            "id": str(self.id),
            "recording_id": str(self.recording_id),
            "key": self.key,
            "value": self.value,
        }


class Tracklist(Base):
    """Model for recording tracklists."""

    __tablename__ = "tracklists"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False, unique=True
    )
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    tracks: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    cue_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="tracklist")

    def __repr__(self) -> str:
        """String representation of Tracklist."""
        return f"<Tracklist(id={self.id}, source={self.source})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert Tracklist to dictionary.

        Returns:
            Dictionary representation of the tracklist
        """
        return {
            "id": str(self.id),
            "recording_id": str(self.recording_id),
            "source": self.source,
            "tracks": self.tracks,
            "cue_file_path": self.cue_file_path,
        }

    def validate_tracks(self) -> bool:
        """Validate the tracks JSONB structure.

        Returns:
            True if tracks structure is valid, False otherwise
        """
        if not self.tracks:
            return True  # Empty tracks is valid

        if not isinstance(self.tracks, list):
            return False

        required_keys = {"title", "artist", "start_time"}
        for track in self.tracks:
            if not isinstance(track, dict):
                return False
            if not required_keys.issubset(track.keys()):
                return False

        return True


class AnalysisResult(Base):
    """Model for analysis results."""

    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False, index=True
    )
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    result_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(DECIMAL(5, 4), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(UTC),
        nullable=True,
    )

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording")

    def __repr__(self) -> str:
        """String representation of AnalysisResult."""
        return f"<AnalysisResult(id={self.id}, type={self.analysis_type}, status={self.status})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert AnalysisResult to dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return {
            "id": str(self.id),
            "recording_id": str(self.recording_id),
            "analysis_type": self.analysis_type,
            "result_data": self.result_data,
            "confidence_score": float(self.confidence_score) if self.confidence_score else None,
            "status": self.status,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RenameProposal(Base):
    """Model for file rename proposals."""

    __tablename__ = "rename_proposals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False, index=True
    )
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_filename: Mapped[str] = mapped_column(Text, nullable=False)
    full_proposed_path: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(DECIMAL(3, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    conflicts: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    warnings: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.current_timestamp(),
    )

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="rename_proposals")

    def __repr__(self) -> str:
        """String representation of RenameProposal."""
        return f"<RenameProposal(id={self.id}, status={self.status}, proposed={self.proposed_filename})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert RenameProposal to dictionary.

        Returns:
            Dictionary representation of the rename proposal
        """
        return {
            "id": str(self.id),
            "recording_id": str(self.recording_id),
            "original_path": self.original_path,
            "original_filename": self.original_filename,
            "proposed_filename": self.proposed_filename,
            "full_proposed_path": self.full_proposed_path,
            "confidence_score": (float(self.confidence_score) if self.confidence_score else None),
            "status": self.status,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
