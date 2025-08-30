"""ML model training pipeline."""

import json
import logging
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .features import FeatureExtractor
from .models import MLModel, ModelAlgorithm, ModelMetrics, ModelStatus, TrainingData

logger = logging.getLogger(__name__)


class Trainer:
    """ML model training pipeline with versioning and evaluation."""

    def __init__(self, model_dir: str = "models/"):
        """Initialize trainer with model storage directory."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.current_model_metadata: MLModel | None = None

    def train(
        self,
        training_data: list[TrainingData],
        algorithm: ModelAlgorithm = ModelAlgorithm.RANDOM_FOREST,
        hyperparameters: dict[str, Any] | None = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> MLModel:
        """Train ML model with given data and configuration."""
        logger.info(f"Starting training with {len(training_data)} samples using {algorithm.value}")
        start_time = time.time()

        # Prepare data
        X, y = self._prepare_training_data(training_data)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size / (1 - test_size), random_state=42
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Initialize model
        if algorithm == ModelAlgorithm.RANDOM_FOREST:
            self.model = self._create_random_forest(hyperparameters)
        else:
            raise NotImplementedError(f"Algorithm {algorithm.value} not yet implemented")

        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate model
        logger.info("Evaluating model...")
        self._evaluate_model(X_val, y_val, "validation")
        test_metrics = self._evaluate_model(X_test, y_test, "test")

        # Combine metrics
        combined_metrics = ModelMetrics(
            accuracy=test_metrics["accuracy"],
            precision=test_metrics["precision"],
            recall=test_metrics["recall"],
            f1_score=test_metrics["f1_score"],
            confusion_matrix=test_metrics["confusion_matrix"],
            per_category_metrics=test_metrics.get("per_category_metrics", {}),
            validation_samples=len(X_val),
            test_samples=len(X_test),
        )

        training_duration = time.time() - start_time
        logger.info(f"Training completed in {training_duration:.2f} seconds")

        # Save model
        model_metadata = self._save_model(
            algorithm=algorithm,
            hyperparameters=hyperparameters or {},
            metrics=combined_metrics,
            training_duration=training_duration,
            sample_count=len(training_data),
        )

        self.current_model_metadata = model_metadata
        return model_metadata

    def _prepare_training_data(self, training_data: list[TrainingData]) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model input."""
        # Convert to format for feature extractor
        samples = []
        labels = []

        for data in training_data:
            samples.append(
                {
                    "tokens": data.tokens,
                    "filename_original": data.filename_original,
                }
            )
            labels.append(data.filename_renamed)

        # Fit feature extractor
        self.feature_extractor.fit(samples)

        # Extract features
        batch_features = self.feature_extractor.transform_batch(samples)

        # For Random Forest, we'll use statistical features
        # In production, we'd combine multiple feature types
        X = batch_features["statistics"]

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        return X, y

    def _create_random_forest(self, hyperparameters: dict[str, Any] | None) -> RandomForestClassifier:
        """Create Random Forest classifier with hyperparameters."""
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,  # Use all cores
        }

        if hyperparameters:
            default_params.update(hyperparameters)

        return RandomForestClassifier(**default_params)

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> dict[str, Any]:
        """Evaluate model performance on dataset."""
        y_pred = self.model.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        # Per-category metrics if we have multiple classes
        if len(np.unique(y)) > 2:
            per_class_precision = precision_score(y, y_pred, average=None, zero_division=0)
            per_class_recall = recall_score(y, y_pred, average=None, zero_division=0)
            per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0)

            metrics["per_category_metrics"] = {}
            for i, label in enumerate(np.unique(y)):
                original_label = self.label_encoder.inverse_transform([label])[0]
                metrics["per_category_metrics"][original_label] = {
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "f1_score": float(per_class_f1[i]),
                }

        logger.info(f"{dataset_name} metrics - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")

        return metrics

    def _save_model(
        self,
        algorithm: ModelAlgorithm,
        hyperparameters: dict[str, Any],
        metrics: ModelMetrics,
        training_duration: float,
        sample_count: int,
    ) -> MLModel:
        """Save trained model to disk with metadata."""
        model_id = str(uuid.uuid4())
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model pickle
        model_filename = f"model_{algorithm.value}_{version}.pkl"
        model_path = self.model_dir / model_filename

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_extractor": self.feature_extractor,
                    "label_encoder": self.label_encoder,
                },
                f,
            )

        # Create metadata
        model_metadata = MLModel(
            id=model_id,
            version=version,
            algorithm=algorithm,
            created_at=datetime.now(),
            training_metrics=metrics.to_dict(),
            hyperparameters=hyperparameters,
            feature_config=self.feature_extractor.config,
            status=ModelStatus.TRAINING,
            file_path=str(model_path),
            training_duration=training_duration,
            sample_count=sample_count,
        )

        # Save metadata
        metadata_path = self.model_dir / f"metadata_{version}.json"
        with open(metadata_path, "w") as f:
            json.dump(model_metadata.to_dict(), f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return model_metadata

    def load_model(self, model_path: str) -> MLModel:
        """Load model from disk."""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_extractor = model_data["feature_extractor"]
        self.label_encoder = model_data["label_encoder"]

        # Load metadata
        version = Path(model_path).stem.split("_")[-1]
        metadata_path = self.model_dir / f"metadata_{version}.json"

        with open(metadata_path) as f:
            metadata_dict = json.load(f)

        self.current_model_metadata = MLModel(
            id=metadata_dict["id"],
            version=metadata_dict["version"],
            algorithm=ModelAlgorithm(metadata_dict["algorithm"]),
            created_at=datetime.fromisoformat(metadata_dict["created_at"]),
            training_metrics=metadata_dict["training_metrics"],
            hyperparameters=metadata_dict["hyperparameters"],
            feature_config=metadata_dict["feature_config"],
            status=ModelStatus(metadata_dict["status"]),
            file_path=metadata_dict["file_path"],
            training_duration=metadata_dict.get("training_duration"),
            sample_count=metadata_dict.get("sample_count"),
        )

        logger.info(f"Model loaded from {model_path}")
        return self.current_model_metadata

    def update_with_feedback(self, feedback_data: list[dict[str, Any]], learning_rate: float = 0.1) -> None:
        """Update model with user feedback (incremental learning)."""
        if not self.model:
            raise ValueError("No model loaded for updating")

        # For Random Forest, we can't do true online learning
        # Instead, we'll retrain periodically with weighted samples
        # This is a placeholder for the feedback mechanism
        logger.info(f"Received {len(feedback_data)} feedback samples for future retraining")

        # In production, we would:
        # 1. Store feedback in database
        # 2. Periodically retrain with original + feedback data
        # 3. Weight feedback samples based on recency and confidence
        pass
