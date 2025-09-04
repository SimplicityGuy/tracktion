"""Database models for File Rename Service."""

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,  # type: ignore[attr-defined]  # SQLAlchemy 2.0 feature not recognized by mypy type stubs
    Session,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import NullPool

from services.file_rename_service.app.config import settings


# Create base class for models
class Base(DeclarativeBase):
    """Base class for all models."""


class PatternType(str, Enum):
    """Types of filename patterns."""

    REGEX = "regex"
    TOKEN = "token"
    TEMPLATE = "template"
    LEARNED = "learned"


class ModelStatus(str, Enum):
    """ML model status."""

    TRAINING = "training"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class Pattern(Base):
    """Filename pattern storage."""

    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(50), nullable=False)
    pattern_value = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(String(100), index=True)
    frequency = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    extra_metadata = Column(JSON, default={})

    # Relationships
    rename_histories = relationship("RenameHistory", back_populates="pattern")

    __table_args__ = (
        Index("idx_pattern_type_category", "pattern_type", "category"),
        Index("idx_pattern_confidence", "confidence_score"),
    )


class MLModel(Base):
    """ML model metadata storage."""

    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), unique=True, nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)  # e.g., "pattern_recognition", "categorization"
    status = Column(String(50), default=ModelStatus.TRAINING.value)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)
    model_path = Column(Text)  # Path to stored model file
    parameters = Column(JSON, default={})
    training_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at = Column(DateTime)
    deprecated_at = Column(DateTime)

    # Relationships
    rename_histories = relationship("RenameHistory", back_populates="ml_model")

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_name_version"),
        Index("idx_model_status", "status"),
        Index("idx_model_type", "model_type"),
    )


class RenameHistory(Base):
    """History of file rename operations."""

    __tablename__ = "rename_history"

    id = Column(Integer, primary_key=True, index=True)
    original_name = Column(Text, nullable=False)
    proposed_name = Column(Text, nullable=False)
    final_name = Column(Text)
    file_path = Column(Text)
    file_extension = Column(String(50))
    file_size_bytes = Column(Integer)
    pattern_id = Column(Integer, ForeignKey("patterns.id"))
    ml_model_id = Column(Integer, ForeignKey("ml_models.id"))
    confidence_score = Column(Float)
    was_accepted = Column(Boolean)
    user_feedback = Column(Text)
    feedback_rating = Column(Integer)  # 1-5 rating
    processing_time_ms = Column(Float)
    error_message = Column(Text)
    extra_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pattern = relationship("Pattern", back_populates="rename_histories")
    ml_model = relationship("MLModel", back_populates="rename_histories")
    feedback = relationship("UserFeedback", back_populates="rename_history", uselist=False)

    __table_args__ = (
        Index("idx_rename_created", "created_at"),
        Index("idx_rename_accepted", "was_accepted"),
        Index("idx_rename_confidence", "confidence_score"),
    )


class UserFeedback(Base):
    """User feedback on rename operations."""

    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    rename_history_id = Column(Integer, ForeignKey("rename_history.id"), unique=True)
    feedback_type = Column(String(50))  # "correction", "rating", "comment"
    corrected_name = Column(Text)
    rating = Column(Integer)  # 1-5
    comment = Column(Text)
    is_helpful = Column(Boolean)
    user_id = Column(String(255))  # Optional user identifier
    session_id = Column(String(255))  # Session identifier
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    rename_history = relationship("RenameHistory", back_populates="feedback")

    __table_args__ = (
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_type", "feedback_type"),
        Index("idx_feedback_rating", "rating"),
    )


class PatternCategory(Base):
    """Categories for organizing patterns."""

    __tablename__ = "pattern_categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    parent_category_id = Column(Integer, ForeignKey("pattern_categories.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Self-referential relationship for hierarchical categories
    parent = relationship("PatternCategory", remote_side=[id], backref="subcategories")

    __table_args__ = (
        Index("idx_category_name", "name"),
        Index("idx_category_parent", "parent_category_id"),
    )


# Database connection management
def get_engine(database_url: str | None = None) -> Engine:
    """Create database engine."""
    url = database_url or settings.database_url
    return create_engine(
        url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.database_echo,
        poolclass=NullPool if settings.environment == "test" else None,
    )


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:  # type: ignore[type-arg]  # SQLAlchemy 2.0 generic sessionmaker typing not supported in 1.4.x type stubs
    """Create session factory."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def create_tables(engine: Engine | None = None) -> None:
    """Create all database tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_tables(engine: Engine | None = None) -> None:
    """Drop all database tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.drop_all(bind=engine)
