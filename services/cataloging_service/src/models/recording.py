"""Recording model for the cataloging service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (  # type: ignore[attr-defined]  # SQLAlchemy 2.0 features; project uses 2.0.43 but type stubs are 1.4.x
    Mapped,
    mapped_column,
    relationship,
)

from .base import Base

if TYPE_CHECKING:
    from .metadata import Metadata
    from .tracklist import Tracklist


class Recording(Base):
    """Recording model representing a music file in the catalog."""

    __tablename__ = "recordings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    sha256_hash: Mapped[str | None] = mapped_column(String, unique=True, nullable=True, index=True)
    xxh128_hash: Mapped[str | None] = mapped_column(String, unique=True, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    # Relationships
    metadata_items: Mapped[list[Metadata]] = relationship(
        "Metadata", back_populates="recording", cascade="all, delete-orphan"
    )
    tracklists: Mapped[list[Tracklist]] = relationship(
        "Tracklist", back_populates="recording", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """String representation of Recording."""
        return f"<Recording(id={self.id}, file_name='{self.file_name}')>"
