"""Metadata model for the cataloging service."""

import uuid

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Metadata(Base):
    """Metadata model for storing key-value pairs associated with recordings."""

    __tablename__ = "metadata"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recording_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="metadata_items")

    # Indexes
    __table_args__ = (
        Index("idx_metadata_recording_id", "recording_id"),
        Index("idx_metadata_key", "key"),
        Index("idx_metadata_recording_key", "recording_id", "key"),
    )

    def __repr__(self) -> str:
        """String representation of Metadata."""
        return f"<Metadata(id={self.id}, key='{self.key}', value='{self.value}')>"
