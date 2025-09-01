"""ML model inference and prediction module."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .models import FeedbackData, MLModel
from .trainer import Trainer

logger = logging.getLogger(__name__)


class Predictor:
    """Handle model predictions with caching and optimization."""

    def __init__(self, model_dir: str = "models/"):
        """Initialize predictor with model directory."""
        self.model_dir = Path(model_dir)
        self.trainer = Trainer(model_dir)
        self.model_loaded = False
        self.prediction_cache: dict[str, dict[str, Any]] = {}
        self.cache_size = 1000  # Maximum cache entries
        self.feedback_buffer: list[FeedbackData] = []
        self.feedback_buffer_size = 100

    def load_latest_model(self) -> MLModel:
        """Load the most recent deployed model."""
        # Find latest model file
        model_files = list(self.model_dir.glob("model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No trained models found")

        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        model_metadata = self.trainer.load_model(str(latest_model))
        self.model_loaded = True
        logger.info(f"Loaded model version {model_metadata.version}")
        return model_metadata

    def load_specific_model(self, version: str) -> MLModel:
        """Load a specific model version."""
        model_path = self.model_dir / f"model_random_forest_{version}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")

        model_metadata = self.trainer.load_model(str(model_path))
        self.model_loaded = True
        logger.info(f"Loaded model version {model_metadata.version}")
        return model_metadata

    def predict(
        self,
        filename: str,
        tokens: list[dict[str, Any]],
        use_cache: bool = True,
        return_probabilities: bool = False,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Generate rename predictions for a filename."""
        if not self.model_loaded:
            self.load_latest_model()

        # Check cache
        cache_key = self._generate_cache_key(filename)
        if use_cache and cache_key in self.prediction_cache:
            logger.debug(f"Cache hit for {filename}")
            return self.prediction_cache[cache_key]

        start_time = time.time()

        # Extract features
        features = self.trainer.feature_extractor.extract_features(tokens, filename)

        # Get statistical features for prediction
        stats = [
            features[key]
            for key in sorted(features.keys())
            if key.startswith(("num_", "has_")) or key in {"filename_length", "avg_token_length"}
        ]

        x = np.array([stats])

        # Get predictions
        if return_probabilities:
            assert self.trainer.model is not None, "Model not loaded"
            probabilities = self.trainer.model.predict_proba(x)[0]
            top_indices = np.argsort(probabilities)[-top_k:][::-1]

            predictions = []
            for idx in top_indices:
                label = self.trainer.label_encoder.inverse_transform([idx])[0]
                predictions.append(
                    {
                        "suggested_name": label,
                        "confidence": float(probabilities[idx]),
                    }
                )
        else:
            assert self.trainer.model is not None, "Model not loaded"
            prediction = self.trainer.model.predict(x)[0]
            suggested_name = self.trainer.label_encoder.inverse_transform([prediction])[0]

            # Get confidence from prediction probability
            probabilities = self.trainer.model.predict_proba(x)[0]
            confidence = float(probabilities[prediction])

            predictions = [
                {
                    "suggested_name": suggested_name,
                    "confidence": confidence,
                }
            ]

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        result = {
            "filename_original": filename,
            "predictions": predictions,
            "inference_time_ms": inference_time,
            "model_version": (
                self.trainer.current_model_metadata.version if self.trainer.current_model_metadata else "unknown"
            ),
        }

        # Update cache
        if use_cache:
            self._update_cache(cache_key, result)

        logger.info(f"Prediction for '{filename}' completed in {inference_time:.2f}ms")
        return result

    def predict_batch(
        self,
        samples: list[tuple[str, list[dict[str, Any]]]],
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate predictions for multiple filenames."""
        if not self.model_loaded:
            self.load_latest_model()

        results: list[Any] = []
        uncached_samples = []
        cached_results = {}

        # Check cache for all samples
        if use_cache:
            for filename, tokens in samples:
                cache_key = self._generate_cache_key(filename)
                if cache_key in self.prediction_cache:
                    cached_results[filename] = self.prediction_cache[cache_key]
                else:
                    uncached_samples.append((filename, tokens))
        else:
            uncached_samples = samples

        # Process uncached samples
        if uncached_samples:
            start_time = time.time()

            # Extract features for all samples
            x_batch_list = []
            for filename, tokens in uncached_samples:
                features = self.trainer.feature_extractor.extract_features(tokens, filename)
                stats = [
                    features[key]
                    for key in sorted(features.keys())
                    if key.startswith(("num_", "has_")) or key in {"filename_length", "avg_token_length"}
                ]
                x_batch_list.append(stats)

            x_batch = np.array(x_batch_list)

            # Get batch predictions
            assert self.trainer.model is not None, "Model not loaded"
            predictions = self.trainer.model.predict(x_batch)
            probabilities = self.trainer.model.predict_proba(x_batch)

            batch_time = (time.time() - start_time) * 1000
            avg_time = batch_time / len(uncached_samples)

            # Process results
            for i, (filename, _tokens) in enumerate(uncached_samples):
                suggested_name = self.trainer.label_encoder.inverse_transform([predictions[i]])[0]
                confidence = float(probabilities[i][predictions[i]])

                result = {
                    "filename_original": filename,
                    "predictions": [
                        {
                            "suggested_name": suggested_name,
                            "confidence": confidence,
                        }
                    ],
                    "inference_time_ms": avg_time,
                    "model_version": (
                        self.trainer.current_model_metadata.version
                        if self.trainer.current_model_metadata
                        else "unknown"
                    ),
                }

                # Update cache
                if use_cache:
                    cache_key = self._generate_cache_key(filename)
                    self._update_cache(cache_key, result)

                results.append(result)

        # Add cached results
        results.extend(cached_results[filename] for filename, _ in samples if filename in cached_results)

        return results

    def add_feedback(self, feedback: FeedbackData) -> None:
        """Add user feedback for model improvement."""
        self.feedback_buffer.append(feedback)

        # Trigger retraining if buffer is full
        if len(self.feedback_buffer) >= self.feedback_buffer_size:
            logger.info("Feedback buffer full, triggering retraining")
            # In production, this would trigger an async retraining job
            self._process_feedback_buffer()

    def _process_feedback_buffer(self) -> None:
        """Process accumulated feedback for model update."""
        if not self.feedback_buffer:
            return

        # Convert feedback to training data format
        feedback_samples = []
        for fb in self.feedback_buffer:
            # Weight samples based on user approval
            weight = fb.weight * (2.0 if fb.user_approved else 0.5)
            feedback_samples.append(
                {
                    "feedback": fb.to_dict(),
                    "weight": weight,
                }
            )

        # In production, this would trigger retraining with weighted samples
        self.trainer.update_with_feedback(feedback_samples)

        # Clear buffer
        self.feedback_buffer.clear()
        logger.info("Processed feedback buffer")

    def _generate_cache_key(self, filename: str) -> str:
        """Generate cache key for filename."""
        version = self.trainer.current_model_metadata.version if self.trainer.current_model_metadata else "unknown"
        return f"{version}:{filename}"

    def _update_cache(self, key: str, value: dict[str, Any]) -> None:
        """Update prediction cache with LRU eviction."""
        # Simple LRU: remove oldest if cache is full
        if len(self.prediction_cache) >= self.cache_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]

        self.prediction_cache[key] = value

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")

    def get_model_info(self) -> dict[str, Any] | None:
        """Get information about the loaded model."""
        if not self.model_loaded or not self.trainer.current_model_metadata:
            return None

        return self.trainer.current_model_metadata.to_dict()
