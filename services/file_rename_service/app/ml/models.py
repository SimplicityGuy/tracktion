"""ML model data structures and interfaces."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ModelAlgorithm(Enum):
    """Supported ML algorithms."""

    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"


class ModelStatus(Enum):
    """Model lifecycle status."""

    TRAINING = "training"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class MLModel:
    """Machine learning model metadata and configuration."""

    id: str
    version: str
    algorithm: ModelAlgorithm
    created_at: datetime
    training_metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    feature_config: dict[str, Any]
    status: ModelStatus
    file_path: str
    training_duration: float | None = None
    sample_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary for serialization."""
        return {
            "id": self.id,
            "version": self.version,
            "algorithm": self.algorithm.value,
            "created_at": self.created_at.isoformat(),
            "training_metrics": self.training_metrics,
            "hyperparameters": self.hyperparameters,
            "feature_config": self.feature_config,
            "status": self.status.value,
            "file_path": self.file_path,
            "training_duration": self.training_duration,
            "sample_count": self.sample_count,
        }


@dataclass
class TrainingData:
    """Training data sample for ML model."""

    filename_original: str
    filename_renamed: str
    tokens: list[dict[str, Any]]  # Token objects from tokenizer
    user_approved: bool
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    features: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "filename_original": self.filename_original,
            "filename_renamed": self.filename_renamed,
            "tokens": self.tokens,
            "user_approved": self.user_approved,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
        }


@dataclass
class FeedbackData:
    """User feedback data for online learning."""

    prediction_id: str
    filename_original: str
    suggested_name: str
    actual_name: str
    user_approved: bool
    timestamp: datetime = field(default_factory=datetime.now)
    weight: float = 1.0  # Weight for importance in training

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "prediction_id": self.prediction_id,
            "filename_original": self.filename_original,
            "suggested_name": self.suggested_name,
            "actual_name": self.actual_name,
            "user_approved": self.user_approved,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
        }


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    per_category_metrics: dict[str, dict[str, float]]
    validation_samples: int
    test_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
            "per_category_metrics": self.per_category_metrics,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
        }
