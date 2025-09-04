"""Base model for SQLAlchemy models."""

from sqlalchemy.orm import (
    DeclarativeBase,  # type: ignore[attr-defined]  # SQLAlchemy 2.0 feature; project uses 2.0.43 but type stubs are 1.4.x
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
