"""
Mood and genre analysis module using TensorFlow models.

Implements mood dimension analysis, genre classification, and danceability
scoring using pre-trained models from Essentia.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from services.analysis_service.src.model_manager import ModelManager

try:
    import essentia.standard as es

    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False
    es = None

logger = logging.getLogger(__name__)


@dataclass
class MoodAnalysisResult:
    """Results from mood and genre analysis."""

    # Mood dimensions (0.0 to 1.0)
    mood_scores: dict[str, float] = field(default_factory=dict)

    # Genre predictions with confidence
    genres: list[dict[str, Any]] = field(default_factory=list)
    primary_genre: str | None = None
    genre_confidence: float = 0.0

    # Additional attributes
    danceability: float = 0.0
    energy: float = 0.0
    valence: float = 0.0  # Musical positivity
    arousal: float = 0.0  # Musical intensity

    # Voice/instrumental classification
    voice_instrumental: str = "unknown"  # "voice", "instrumental", or "unknown"
    voice_confidence: float = 0.0

    # Overall confidence
    overall_confidence: float = 0.0
    needs_review: bool = False


class MoodAnalyzer:
    """
    Mood and genre analysis using pre-trained TensorFlow models.

    Supports multiple mood dimensions, genre classification,
    and ensemble voting for improved accuracy.
    """

    # Mood dimension models
    MOOD_MODELS: ClassVar[list[str]] = [
        "mood_happy",
        "mood_sad",
        "mood_aggressive",
        "mood_relaxed",
    ]

    # Genre models
    GENRE_MODELS: ClassVar[list[str]] = [
        "genre_discogs_effnet",
    ]

    # Additional feature models
    FEATURE_MODELS: ClassVar[list[str]] = [
        "danceability",
    ]

    # Genre labels for Discogs EffNet model
    DISCOGS_GENRES: ClassVar[list[str]] = [
        "Blues",
        "Brass & Military",
        "Children's",
        "Classical",
        "Electronic",
        "Folk, World, & Country",
        "Funk / Soul",
        "Hip Hop",
        "Jazz",
        "Latin",
        "Non-Music",
        "Pop",
        "Reggae",
        "Rock",
        "Stage & Screen",
    ]

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        enable_mood_detection: bool = True,
        enable_genre_detection: bool = True,
        enable_danceability: bool = True,
        ensemble_voting_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        mood_dimensions: list[str] | None = None,
    ):
        """
        Initialize the mood analyzer.

        Args:
            model_manager: ModelManager instance for loading models
            enable_mood_detection: Enable mood dimension analysis
            enable_genre_detection: Enable genre classification
            enable_danceability: Enable danceability scoring
            ensemble_voting_threshold: Threshold for ensemble voting
            confidence_threshold: Minimum confidence for reliable detection
            mood_dimensions: List of mood dimensions to analyze
        """
        self.model_manager = model_manager or ModelManager()
        self.enable_mood_detection = enable_mood_detection
        self.enable_genre_detection = enable_genre_detection
        self.enable_danceability = enable_danceability
        self.ensemble_voting_threshold = ensemble_voting_threshold
        self.confidence_threshold = confidence_threshold
        self.mood_dimensions = mood_dimensions or [
            "happy",
            "sad",
            "aggressive",
            "relaxed",
            "acoustic",
            "electronic",
            "party",
        ]

        # Cache for loaded models
        self._loaded_models: dict[str, Any] = {}

    def analyze_mood(self, audio_file: str) -> MoodAnalysisResult | None:
        """
        Analyze mood and genre characteristics of an audio file.

        Args:
            audio_file: Path to the audio file

        Returns:
            MoodAnalysisResult with all analyzed features
        """
        if not HAS_ESSENTIA:
            logger.error("Essentia not installed. Install with: uv pip install essentia")
            return None

        try:
            # Load audio file (16kHz for most models)
            logger.info(f"Loading audio file for mood analysis: {audio_file}")
            audio = es.MonoLoader(filename=audio_file, sampleRate=16000)()

            result = MoodAnalysisResult()

            # Analyze mood dimensions
            if self.enable_mood_detection:
                mood_scores = self._analyze_mood_dimensions(audio)
                result.mood_scores = mood_scores

                # Calculate valence and arousal from mood dimensions
                result.valence = self._calculate_valence(mood_scores)
                result.arousal = self._calculate_arousal(mood_scores)

            # Analyze genre
            if self.enable_genre_detection:
                genre_results = self._analyze_genre(audio)
                result.genres = genre_results
                if genre_results:
                    result.primary_genre = genre_results[0]["genre"]
                    result.genre_confidence = genre_results[0]["confidence"]

            # Analyze danceability
            if self.enable_danceability:
                result.danceability = self._analyze_danceability(audio)

            # Calculate energy from various features
            result.energy = self._calculate_energy(result)

            # Determine voice/instrumental
            result.voice_instrumental, result.voice_confidence = self._detect_voice_instrumental(result.mood_scores)

            # Calculate overall confidence
            result.overall_confidence = self._calculate_overall_confidence(result)
            result.needs_review = result.overall_confidence < self.confidence_threshold

            logger.info(
                f"Mood analysis complete: {result.primary_genre or 'Unknown'} "
                f"(confidence: {result.overall_confidence:.2f})"
            )

            return result

        except ImportError:
            logger.error("Essentia not installed. Install with: uv pip install essentia")
            return None
        except Exception as e:
            logger.error(f"Error analyzing mood for {audio_file}: {e!s}")
            return None

    def _analyze_mood_dimensions(self, audio: np.ndarray) -> dict[str, float]:
        """
        Analyze mood dimensions using multiple models.

        Args:
            audio: Audio signal array

        Returns:
            Dictionary of mood dimension scores
        """
        mood_scores = {}

        for dimension in self.mood_dimensions:
            model_id = f"mood_{dimension}"

            # Check if we have a model for this dimension
            if model_id not in self.model_manager.get_all_models():
                logger.debug(f"No model available for mood dimension: {dimension}")
                continue

            try:
                # Load and run model
                score = self._run_mood_model(model_id, audio)
                if score is not None:
                    mood_scores[dimension] = score
                    logger.debug(f"Mood {dimension}: {score:.3f}")
            except Exception as e:
                logger.warning(f"Failed to analyze mood dimension {dimension}: {e!s}")

        return mood_scores

    def _run_mood_model(self, model_id: str, audio: np.ndarray) -> float | None:
        """
        Run a mood model on audio.

        Args:
            model_id: Model identifier
            audio: Audio signal array

        Returns:
            Mood score (0.0 to 1.0) or None if failed
        """
        try:
            # Load model if not cached
            if model_id not in self._loaded_models:
                model = self.model_manager.load_model(model_id)
                if model is None:
                    return None
                self._loaded_models[model_id] = model

            model = self._loaded_models[model_id]

            # Prepare audio for model (add batch dimension)
            audio_input = np.expand_dims(audio, axis=0)

            # Run inference
            predictions = model(audio_input)

            # Extract score (models typically output probabilities)
            score = float(predictions.numpy()[0]) if hasattr(predictions, "numpy") else float(predictions[0])

            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]

        except Exception as e:
            logger.error(f"Error running mood model {model_id}: {e!s}")
            return None

    def _analyze_genre(self, audio: np.ndarray) -> list[dict[str, Any]]:
        """
        Analyze genre using classification models.

        Args:
            audio: Audio signal array

        Returns:
            List of genre predictions with confidence scores
        """
        genre_predictions = []

        for model_id in self.GENRE_MODELS:
            if model_id not in self.model_manager.get_all_models():
                continue

            try:
                predictions = self._run_genre_model(model_id, audio)
                if predictions:
                    genre_predictions.extend(predictions)
            except Exception as e:
                logger.warning(f"Failed to run genre model {model_id}: {e!s}")

        # Sort by confidence and return top predictions
        genre_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return genre_predictions[:5]  # Return top 5 genres

    def _run_genre_model(self, model_id: str, audio: np.ndarray) -> list[dict[str, Any]]:
        """
        Run a genre classification model.

        Args:
            model_id: Model identifier
            audio: Audio signal array

        Returns:
            List of genre predictions
        """
        try:
            # Load model if not cached
            if model_id not in self._loaded_models:
                model = self.model_manager.load_model(model_id)
                if model is None:
                    return []
                self._loaded_models[model_id] = model

            model = self._loaded_models[model_id]

            # Prepare audio for model
            audio_input = np.expand_dims(audio, axis=0)

            # Run inference
            predictions = model(audio_input)

            # Convert to numpy if needed
            if hasattr(predictions, "numpy"):
                probs = predictions.numpy()
                if len(probs.shape) > 1:
                    probs = probs[0]  # Remove batch dimension
            else:
                probs = np.array(predictions)
                if len(probs.shape) > 1:
                    probs = probs[0]  # Remove batch dimension

            # Create genre predictions
            genre_predictions = []
            for i, prob in enumerate(probs):
                if i < len(self.DISCOGS_GENRES) and prob > 0.1:  # Only include significant predictions
                    genre_predictions.append(
                        {
                            "genre": self.DISCOGS_GENRES[i],
                            "confidence": float(prob),
                            "model": model_id,
                        }
                    )

            return genre_predictions

        except Exception as e:
            logger.error(f"Error running genre model {model_id}: {e!s}")
            return []

    def _analyze_danceability(self, audio: np.ndarray) -> float:
        """
        Analyze danceability of the audio.

        Args:
            audio: Audio signal array

        Returns:
            Danceability score (0.0 to 1.0)
        """
        model_id = "danceability"

        if model_id not in self.model_manager.get_all_models():
            return 0.5  # Default to neutral

        try:
            score = self._run_mood_model(model_id, audio)
            return score if score is not None else 0.5
        except Exception as e:
            logger.warning(f"Failed to analyze danceability: {e!s}")
            return 0.5

    def _calculate_valence(self, mood_scores: dict[str, float]) -> float:
        """
        Calculate musical valence (positivity) from mood scores.

        Args:
            mood_scores: Dictionary of mood dimension scores

        Returns:
            Valence score (0.0 to 1.0)
        """
        positive_moods = ["happy", "party", "energetic"]
        negative_moods = ["sad", "aggressive", "dark"]

        positive_sum = sum(mood_scores.get(m, 0) for m in positive_moods)
        negative_sum = sum(mood_scores.get(m, 0) for m in negative_moods)

        # Calculate valence as balance between positive and negative
        return positive_sum / (positive_sum + negative_sum) if positive_sum + negative_sum > 0 else 0.5

    def _calculate_arousal(self, mood_scores: dict[str, float]) -> float:
        """
        Calculate musical arousal (intensity) from mood scores.

        Args:
            mood_scores: Dictionary of mood dimension scores

        Returns:
            Arousal score (0.0 to 1.0)
        """
        high_arousal = ["aggressive", "party", "energetic"]
        low_arousal = ["relaxed", "sad", "acoustic"]

        high_sum = sum(mood_scores.get(m, 0) for m in high_arousal)
        low_sum = sum(mood_scores.get(m, 0) for m in low_arousal)

        # Calculate arousal as balance between high and low energy
        return high_sum / (high_sum + low_sum) if high_sum + low_sum > 0 else 0.5

    def _calculate_energy(self, result: MoodAnalysisResult) -> float:
        """
        Calculate overall energy level from various features.

        Args:
            result: MoodAnalysisResult with analyzed features

        Returns:
            Energy score (0.0 to 1.0)
        """
        # Combine danceability, arousal, and specific mood dimensions
        energy_components = [
            result.danceability,
            result.arousal,
            result.mood_scores.get("aggressive", 0) * 0.8,
            result.mood_scores.get("party", 0) * 0.9,
            (1.0 - result.mood_scores.get("relaxed", 0.5)) * 0.7,
        ]

        # Weight and average the components
        energy = sum(energy_components) / len(energy_components)
        return min(max(energy, 0.0), 1.0)

    def _detect_voice_instrumental(self, mood_scores: dict[str, float]) -> tuple[str, float]:
        """
        Detect if track is vocal or instrumental.

        Args:
            mood_scores: Dictionary of mood dimension scores

        Returns:
            Tuple of (classification, confidence)
        """
        # Simple heuristic based on acoustic score
        # In a real implementation, we'd use a dedicated model
        acoustic_score = mood_scores.get("acoustic", 0.5)

        if acoustic_score > 0.7:
            return "voice", 0.6  # Acoustic often indicates vocals
        if acoustic_score < 0.3:
            return "instrumental", 0.6  # Electronic often instrumental
        return "unknown", 0.3

    def _calculate_overall_confidence(self, result: MoodAnalysisResult) -> float:
        """
        Calculate overall confidence from all analysis results.

        Args:
            result: MoodAnalysisResult with all features

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        confidences = []

        # Add genre confidence if available
        if result.genre_confidence > 0:
            confidences.append(result.genre_confidence)

        # Add mood dimension confidences (using score as proxy)
        if result.mood_scores:
            # High or low scores indicate confidence
            for score in result.mood_scores.values():
                confidence = abs(score - 0.5) * 2  # Convert to confidence
                confidences.append(confidence)

        # Add voice detection confidence
        if result.voice_confidence > 0:
            confidences.append(result.voice_confidence)

        # Calculate weighted average
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.0

    def analyze_with_ensemble(self, audio_file: str, models: list[str] | None = None) -> MoodAnalysisResult | None:
        """
        Analyze using ensemble of models for improved accuracy.

        Args:
            audio_file: Path to the audio file
            models: List of model IDs to use (None for all)

        Returns:
            MoodAnalysisResult with ensemble voting
        """
        # Get base analysis
        result = self.analyze_mood(audio_file)
        if result is None:
            return None

        # For ensemble, we would run multiple models and vote
        # This is a simplified version
        if result.genres and len(result.genres) > 1:
            # Apply ensemble voting threshold
            filtered_genres = [g for g in result.genres if g["confidence"] >= self.ensemble_voting_threshold]
            if filtered_genres:
                result.genres = filtered_genres

        return result

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return list(self._loaded_models.keys())

    def clear_model_cache(self) -> None:
        """Clear the loaded models cache to free memory."""
        self._loaded_models.clear()
        logger.info("Cleared mood analyzer model cache")
