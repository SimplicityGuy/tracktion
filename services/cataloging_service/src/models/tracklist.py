"""Tracklist model for the cataloging service."""

import uuid
from typing import Any

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Tracklist(Base):
    """Tracklist model for storing track information associated with recordings."""

    __tablename__ = "tracklists"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recording_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    cue_file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    tracks: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="tracklists")

    def __repr__(self) -> str:
        """String representation of Tracklist."""
        return f"<Tracklist(id={self.id}, source='{self.source}', tracks={len(self.tracks)})>"
