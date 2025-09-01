"""Data models for feedback learning loop."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class FeedbackAction(str, Enum):
    """User feedback actions."""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class Feedback(BaseModel):
    """Feedback data model for user actions on rename proposals."""

    id: str = Field(..., description="Unique feedback identifier")
    proposal_id: str = Field(..., description="Associated proposal ID")
    original_filename: str = Field(..., description="Original filename")
    proposed_filename: str = Field(..., description="System proposed filename")
    user_action: FeedbackAction = Field(..., description="User action taken")
    user_filename: str | None = Field(None, description="User-provided filename if modified")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Feedback timestamp")
    model_version: str = Field(..., description="Model version that generated proposal")
    processing_time_ms: float | None = Field(None, description="Processing time in milliseconds")
    context_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @field_validator("user_filename")
    @classmethod
    def validate_user_filename(cls, v: str | None, info: Any) -> str | None:
        """Validate user filename is provided when action is modified."""
        if info.data.get("user_action") == FeedbackAction.MODIFIED and not v:
            raise ValueError("user_filename required when action is 'modified'")
        return v

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return timestamp.isoformat()

    model_config = ConfigDict()


class LearningMetrics(BaseModel):
    """Metrics for tracking model learning and performance."""

    model_version: str = Field(..., description="Model version identifier")
    total_feedback: int = Field(0, ge=0, description="Total feedback count")
    approval_rate: float = Field(0.0, ge=0.0, le=1.0, description="Approval rate")
    rejection_rate: float = Field(0.0, ge=0.0, le=1.0, description="Rejection rate")
    modification_rate: float = Field(0.0, ge=0.0, le=1.0, description="Modification rate")
    accuracy_trend: list[float] = Field(default_factory=list, description="Accuracy trend over time")
    last_retrained: datetime | None = Field(None, description="Last retraining timestamp")
    next_retrain_at: datetime | None = Field(None, description="Next scheduled retraining")
    performance_metrics: dict[str, Any] = Field(default_factory=dict, description="Additional performance metrics")

    @field_validator("approval_rate", "rejection_rate", "modification_rate")
    @classmethod
    def validate_rates(cls, v: float) -> float:
        """Ensure rates are valid percentages."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Rate must be between 0.0 and 1.0, got {v}")
        return v

    def calculate_rates(self, feedbacks: list[Feedback]) -> None:
        """Calculate approval/rejection/modification rates from feedback list."""
        if not feedbacks:
            return

        total = len(feedbacks)
        approved = sum(1 for f in feedbacks if f.user_action == FeedbackAction.APPROVED)
        rejected = sum(1 for f in feedbacks if f.user_action == FeedbackAction.REJECTED)
        modified = sum(1 for f in feedbacks if f.user_action == FeedbackAction.MODIFIED)

        self.total_feedback = total
        self.approval_rate = approved / total
        self.rejection_rate = rejected / total
        self.modification_rate = modified / total

    @field_serializer("last_retrained", "next_retrain_at")
    def serialize_datetime(self, dt: datetime | None) -> str | None:
        """Serialize datetime fields to ISO format."""
        return dt.isoformat() if dt else None

    model_config = ConfigDict()


class ExperimentStatus(str, Enum):
    """A/B experiment status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CONCLUDED = "concluded"


class ABExperiment(BaseModel):
    """A/B testing experiment model."""

    id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., description="Experiment name")
    description: str | None = Field(None, description="Experiment description")
    variant_a: str = Field(..., description="Control model version")
    variant_b: str = Field(..., description="Treatment model version")
    traffic_split: float = Field(0.5, ge=0.0, le=1.0, description="Traffic to variant B")
    start_date: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Start date")
    end_date: datetime | None = Field(None, description="End date")
    metrics_a: dict[str, Any] = Field(default_factory=dict, description="Variant A metrics")
    metrics_b: dict[str, Any] = Field(default_factory=dict, description="Variant B metrics")
    status: ExperimentStatus = Field(ExperimentStatus.PENDING, description="Experiment status")
    sample_size_a: int = Field(0, ge=0, description="Sample size for variant A")
    sample_size_b: int = Field(0, ge=0, description="Sample size for variant B")
    statistical_significance: float | None = Field(
        None, ge=0.0, le=1.0, description="Statistical significance (p-value)"
    )
    winner: str | None = Field(None, description="Winning variant")

    @field_validator("traffic_split")
    @classmethod
    def validate_traffic_split(cls, v: float) -> float:
        """Validate traffic split is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Traffic split must be between 0.0 and 1.0, got {v}")
        return v

    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        return self.status == ExperimentStatus.RUNNING

    def allocate_variant(self, random_value: float) -> str:
        """Allocate traffic to variant based on split."""
        if random_value > self.traffic_split:
            return self.variant_a
        return self.variant_b

    @field_serializer("start_date", "end_date")
    def serialize_datetime(self, dt: datetime | None) -> str | None:
        """Serialize datetime fields to ISO format."""
        return dt.isoformat() if dt else None

    model_config = ConfigDict()


class FeedbackBatch(BaseModel):
    """Batch of feedback for processing."""

    feedbacks: list[Feedback] = Field(..., min_length=1, description="Feedback items")
    batch_id: str = Field(..., description="Batch identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Batch creation time")
    processed: bool = Field(False, description="Processing status")
    processed_at: datetime | None = Field(None, description="Processing timestamp")
    error: str | None = Field(None, description="Processing error if any")

    def mark_processed(self) -> None:
        """Mark batch as processed."""
        self.processed = True
        self.processed_at = datetime.now(UTC)

    @field_serializer("created_at", "processed_at")
    def serialize_datetime(self, dt: datetime | None) -> str | None:
        """Serialize datetime fields to ISO format."""
        return dt.isoformat() if dt else None

    model_config = ConfigDict()
