"""
Unit tests for the Mood and Genre Analyzer module.

Tests mood detection, genre classification, and feature extraction
using mocked TensorFlow models and Essentia functions.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from services.analysis_service.src.model_manager import ModelManager
from services.analysis_service.src.mood_analyzer import MoodAnalysisResult, MoodAnalyzer


class TestMoodAnalyzer:
    """Test cases for MoodAnalyzer class."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager."""
        manager = Mock(spec=ModelManager)
        manager.get_all_models.return_value = [
            "mood_happy",
            "mood_sad",
            "mood_aggressive",
            "mood_relaxed",
            "genre_discogs_effnet",
            "danceability",
        ]

        # Mock model loading
        def load_model(model_id):
            mock_model = Mock()
            # Make the model callable and return predictions
            mock_model.return_value = Mock(numpy=lambda: np.array([0.7]))
            return mock_model

        manager.load_model.side_effect = load_model
        return manager

    @pytest.fixture
    def mood_analyzer(self, mock_model_manager):
        """Create a MoodAnalyzer instance with mocked dependencies."""
        return MoodAnalyzer(
            model_manager=mock_model_manager,
            enable_mood_detection=True,
            enable_genre_detection=True,
            enable_danceability=True,
            confidence_threshold=0.6,
        )

    @pytest.fixture
    def mock_essentia(self):
        """Create mock Essentia module."""
        mock_es = MagicMock()
        mock_es.MonoLoader.return_value.return_value = np.zeros(16000)  # 1 second at 16kHz
        return mock_es

    def test_initialization(self):
        """Test MoodAnalyzer initialization with custom parameters."""
        analyzer = MoodAnalyzer(
            enable_mood_detection=False,
            enable_genre_detection=False,
            enable_danceability=False,
            ensemble_voting_threshold=0.7,
            confidence_threshold=0.8,
            mood_dimensions=["happy", "sad"],
        )

        assert analyzer.enable_mood_detection is False
        assert analyzer.enable_genre_detection is False
        assert analyzer.enable_danceability is False
        assert analyzer.ensemble_voting_threshold == 0.7
        assert analyzer.confidence_threshold == 0.8
        assert analyzer.mood_dimensions == ["happy", "sad"]

    def test_initialization_with_default_model_manager(self):
        """Test initialization creates default ModelManager."""
        analyzer = MoodAnalyzer()
        assert analyzer.model_manager is not None
        assert isinstance(analyzer.model_manager, ModelManager)

    @patch("services.analysis_service.src.mood_analyzer.logger")
    def test_analyze_mood_success(self, mock_logger, mood_analyzer, mock_essentia):
        """Test successful mood analysis."""
        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = mood_analyzer.analyze_mood("test.mp3")

        assert result is not None
        assert isinstance(result, MoodAnalysisResult)
        assert len(result.mood_scores) > 0
        assert result.overall_confidence > 0
        mock_logger.info.assert_called()

    def test_analyze_mood_essentia_not_installed(self, mood_analyzer):
        """Test handling when Essentia is not installed."""
        with (
            patch.dict("sys.modules", {"essentia.standard": None}),
            patch(
                "builtins.__import__",
                side_effect=ImportError("No module named essentia"),
            ),
        ):
            result = mood_analyzer.analyze_mood("test.mp3")

        assert result is None

    @patch("services.analysis_service.src.mood_analyzer.logger")
    def test_analyze_mood_file_error(self, mock_logger, mood_analyzer, mock_essentia):
        """Test handling when audio file cannot be loaded."""
        mock_essentia.MonoLoader.return_value.side_effect = Exception("File not found")

        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = mood_analyzer.analyze_mood("nonexistent.mp3")

        assert result is None
        mock_logger.error.assert_called()

    def test_analyze_mood_dimensions(self, mood_analyzer):
        """Test mood dimension analysis."""
        audio = np.zeros(16000)

        # Mock the _run_mood_model method
        mood_analyzer._run_mood_model = Mock(side_effect=[0.8, 0.3, 0.6, 0.7])

        scores = mood_analyzer._analyze_mood_dimensions(audio)

        assert "happy" in scores
        assert scores["happy"] == 0.8
        assert len(scores) == 4  # 4 mood dimensions mocked

    def test_run_mood_model_success(self):
        """Test running a mood model successfully."""
        # Create analyzer with fresh mock manager
        mock_manager = Mock(spec=ModelManager)
        analyzer = MoodAnalyzer(model_manager=mock_manager)

        audio = np.zeros(16000)
        model_id = "mood_happy"

        # Mock model
        mock_model = Mock()
        mock_predictions = Mock()
        mock_predictions.numpy.return_value = np.array([0.85])
        mock_model.return_value = mock_predictions

        mock_manager.load_model.return_value = mock_model

        score = analyzer._run_mood_model(model_id, audio)

        assert score == 0.85

    def test_run_mood_model_caching(self):
        """Test that models are cached after first load."""
        # Create analyzer with fresh mock manager
        mock_manager = Mock(spec=ModelManager)
        analyzer = MoodAnalyzer(model_manager=mock_manager)

        audio = np.zeros(16000)
        model_id = "mood_happy"

        # Mock model
        mock_model = Mock()
        mock_model.return_value = Mock(numpy=lambda: np.array([0.75]))
        mock_manager.load_model.return_value = mock_model

        # First call
        score1 = analyzer._run_mood_model(model_id, audio)
        # Second call (should use cached model)
        score2 = analyzer._run_mood_model(model_id, audio)

        assert score1 == score2 == 0.75
        # Model should only be loaded once
        mock_manager.load_model.assert_called_once_with(model_id)

    def test_run_mood_model_clamps_values(self):
        """Test that mood scores are clamped to [0, 1]."""
        # Create analyzer with fresh mock manager
        mock_manager = Mock(spec=ModelManager)
        analyzer = MoodAnalyzer(model_manager=mock_manager)

        audio = np.zeros(16000)

        # Test value > 1
        mock_model = Mock()
        mock_model.return_value = Mock(numpy=lambda: np.array([1.5]))
        mock_manager.load_model.return_value = mock_model

        score = analyzer._run_mood_model("mood_happy", audio)
        assert score == 1.0

        # Test value < 0
        mock_model.return_value = Mock(numpy=lambda: np.array([-0.5]))
        analyzer._loaded_models.clear()  # Clear cache

        score = analyzer._run_mood_model("mood_sad", audio)
        assert score == 0.0

    def test_run_mood_model_error_handling(self):
        """Test mood model error handling."""
        # Create analyzer with fresh mock manager
        mock_manager = Mock(spec=ModelManager)
        analyzer = MoodAnalyzer(model_manager=mock_manager)

        audio = np.zeros(16000)

        mock_manager.load_model.return_value = None

        score = analyzer._run_mood_model("mood_happy", audio)
        assert score is None

    def test_analyze_genre(self):
        """Test genre analysis."""
        # Create analyzer with fresh mock manager
        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_all_models.return_value = ["genre_discogs_effnet"]
        analyzer = MoodAnalyzer(model_manager=mock_manager)

        audio = np.zeros(16000)

        # Mock genre model predictions - needs to be 2D array (batch dimension)
        mock_model = Mock()
        probs = np.zeros(15)
        probs[4] = 0.6  # Electronic (index 4)
        probs[11] = 0.15  # Pop (index 11)
        predictions = np.array([probs])  # Add batch dimension
        mock_model.return_value = Mock(numpy=lambda: predictions)
        mock_manager.load_model.return_value = mock_model

        genres = analyzer._analyze_genre(audio)

        assert len(genres) == 2  # Two genres above 0.1 threshold
        assert genres[0]["genre"] == "Electronic"
        assert genres[0]["confidence"] == 0.6
        assert genres[1]["genre"] == "Pop"
        assert genres[1]["confidence"] == 0.15

    def test_run_genre_model(self):
        """Test running a genre classification model."""
        # Create analyzer with fresh mock manager
        mock_manager = Mock(spec=ModelManager)
        analyzer = MoodAnalyzer(model_manager=mock_manager)

        audio = np.zeros(16000)
        model_id = "genre_discogs_effnet"

        # Mock model with genre predictions
        mock_model = Mock()
        # Probabilities for each genre
        probs = np.zeros(15)
        probs[4] = 0.7  # Electronic
        probs[11] = 0.2  # Pop
        mock_model.return_value = Mock(numpy=lambda: np.array([probs]))

        mock_manager.load_model.return_value = mock_model

        predictions = analyzer._run_genre_model(model_id, audio)

        assert len(predictions) == 2  # Only significant predictions
        assert predictions[0]["genre"] == "Electronic"
        assert predictions[0]["confidence"] == 0.7
        assert predictions[1]["genre"] == "Pop"
        assert predictions[1]["confidence"] == 0.2

    def test_analyze_danceability(self, mood_analyzer):
        """Test danceability analysis."""
        audio = np.zeros(16000)

        # Mock danceability model
        mood_analyzer._run_mood_model = Mock(return_value=0.85)

        danceability = mood_analyzer._analyze_danceability(audio)

        assert danceability == 0.85

    def test_analyze_danceability_no_model(self, mood_analyzer):
        """Test danceability when model not available."""
        audio = np.zeros(16000)

        mood_analyzer.model_manager.get_all_models.return_value = []

        danceability = mood_analyzer._analyze_danceability(audio)

        assert danceability == 0.5  # Default value

    def test_calculate_valence(self, mood_analyzer):
        """Test valence calculation from mood scores."""
        # Positive valence
        mood_scores = {
            "happy": 0.8,
            "party": 0.7,
            "sad": 0.2,
        }
        valence = mood_analyzer._calculate_valence(mood_scores)
        assert valence > 0.7  # Should be positive

        # Negative valence
        mood_scores = {
            "happy": 0.2,
            "sad": 0.8,
            "aggressive": 0.7,
        }
        valence = mood_analyzer._calculate_valence(mood_scores)
        assert valence < 0.3  # Should be negative

        # Neutral (no scores)
        valence = mood_analyzer._calculate_valence({})
        assert valence == 0.5

    def test_calculate_arousal(self, mood_analyzer):
        """Test arousal calculation from mood scores."""
        # High arousal
        mood_scores = {
            "aggressive": 0.8,
            "party": 0.7,
            "relaxed": 0.2,
        }
        arousal = mood_analyzer._calculate_arousal(mood_scores)
        assert arousal > 0.7  # Should be high

        # Low arousal
        mood_scores = {
            "relaxed": 0.8,
            "sad": 0.6,
            "aggressive": 0.1,
        }
        arousal = mood_analyzer._calculate_arousal(mood_scores)
        assert arousal < 0.3  # Should be low

    def test_calculate_energy(self, mood_analyzer):
        """Test energy calculation from various features."""
        result = MoodAnalysisResult()
        result.danceability = 0.8
        result.arousal = 0.7
        result.mood_scores = {
            "aggressive": 0.6,
            "party": 0.8,
            "relaxed": 0.2,
        }

        energy = mood_analyzer._calculate_energy(result)

        assert 0.6 < energy < 0.9  # Should be relatively high

    def test_detect_voice_instrumental(self, mood_analyzer):
        """Test voice/instrumental detection."""
        # High acoustic score -> likely voice
        mood_scores = {"acoustic": 0.8}
        classification, _confidence = mood_analyzer._detect_voice_instrumental(mood_scores)
        assert classification == "voice"

        # Low acoustic score -> likely instrumental
        mood_scores = {"acoustic": 0.2}
        classification, _confidence = mood_analyzer._detect_voice_instrumental(mood_scores)
        assert classification == "instrumental"

        # Medium score -> unknown
        mood_scores = {"acoustic": 0.5}
        classification, _confidence = mood_analyzer._detect_voice_instrumental(mood_scores)
        assert classification == "unknown"

    def test_calculate_overall_confidence(self, mood_analyzer):
        """Test overall confidence calculation."""
        result = MoodAnalysisResult()
        result.genre_confidence = 0.8
        result.mood_scores = {
            "happy": 0.9,  # High confidence (far from 0.5)
            "sad": 0.1,  # High confidence (far from 0.5)
            "relaxed": 0.5,  # Low confidence (at 0.5)
        }
        result.voice_confidence = 0.7

        confidence = mood_analyzer._calculate_overall_confidence(result)

        assert 0.5 < confidence < 0.9

    def test_calculate_overall_confidence_no_data(self, mood_analyzer):
        """Test confidence calculation with no data."""
        result = MoodAnalysisResult()

        confidence = mood_analyzer._calculate_overall_confidence(result)

        assert confidence == 0.0

    def test_analyze_with_ensemble(self, mood_analyzer, mock_essentia):
        """Test ensemble analysis."""
        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            # Mock the base analyze_mood
            base_result = MoodAnalysisResult()
            base_result.genres = [
                {"genre": "Electronic", "confidence": 0.8, "model": "model1"},
                {
                    "genre": "Pop",
                    "confidence": 0.3,
                    "model": "model1",
                },  # Below threshold
            ]

            with patch.object(mood_analyzer, "analyze_mood", return_value=base_result):
                result = mood_analyzer.analyze_with_ensemble("test.mp3")

        assert result is not None
        # Low confidence genre should be filtered out
        assert len(result.genres) == 1
        assert result.genres[0]["genre"] == "Electronic"

    def test_needs_review_flag(self, mood_analyzer, mock_essentia):
        """Test that needs_review flag is set correctly."""
        with (
            patch.dict("sys.modules", {"essentia.standard": mock_essentia}),
            patch.object(mood_analyzer, "_calculate_overall_confidence", return_value=0.4),
        ):
            # Mock low confidence analysis
            result = mood_analyzer.analyze_mood("test.mp3")

        assert result is not None
        assert result.needs_review is True  # Below 0.6 threshold

    def test_get_loaded_models(self, mood_analyzer):
        """Test getting list of loaded models."""
        mood_analyzer._loaded_models = {
            "mood_happy": Mock(),
            "genre_discogs_effnet": Mock(),
        }

        models = mood_analyzer.get_loaded_models()

        assert len(models) == 2
        assert "mood_happy" in models
        assert "genre_discogs_effnet" in models

    def test_clear_model_cache(self, mood_analyzer):
        """Test clearing the model cache."""
        mood_analyzer._loaded_models = {
            "mood_happy": Mock(),
            "genre_discogs_effnet": Mock(),
        }

        mood_analyzer.clear_model_cache()

        assert len(mood_analyzer._loaded_models) == 0

    def test_complete_analysis_flow(self, mood_analyzer, mock_essentia):
        """Test complete analysis flow with all features enabled."""
        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            # Mock various analysis methods
            mood_analyzer._analyze_mood_dimensions = Mock(
                return_value={
                    "happy": 0.7,
                    "sad": 0.3,
                    "aggressive": 0.4,
                    "relaxed": 0.6,
                }
            )
            mood_analyzer._analyze_genre = Mock(
                return_value=[
                    {"genre": "Pop", "confidence": 0.75, "model": "test"},
                ]
            )
            mood_analyzer._analyze_danceability = Mock(return_value=0.65)

            result = mood_analyzer.analyze_mood("test.mp3")

        assert result is not None
        assert len(result.mood_scores) == 4
        assert result.primary_genre == "Pop"
        assert result.genre_confidence == 0.75
        assert result.danceability == 0.65
        assert result.valence > 0  # Calculated from mood scores
        assert result.arousal > 0  # Calculated from mood scores
        assert result.energy > 0  # Calculated from features
        assert result.overall_confidence > 0
