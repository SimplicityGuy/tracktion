"""
Unit tests for the Musical Key Detector module.

Tests key detection functionality using mocked Essentia functions
for deterministic and fast testing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.analysis_service.src.key_detector import KeyDetectionResult, KeyDetector


class TestKeyDetector:
    """Test cases for KeyDetector class."""

    @pytest.fixture
    def key_detector(self):
        """Create a KeyDetector instance with default settings."""
        return KeyDetector(
            confidence_threshold=0.7,
            agreement_boost=1.2,
            disagreement_penalty=0.8,
            needs_review_threshold=0.7,
        )

    @pytest.fixture
    def mock_essentia(self):
        """Create mock Essentia module."""
        mock_es = MagicMock()

        # Mock MonoLoader
        mock_es.MonoLoader.return_value.return_value = np.zeros(44100)  # 1 second of audio

        # Mock KeyExtractor (returns key, scale, strength)
        mock_es.KeyExtractor.return_value.return_value = ("C", "major", 0.85)

        # Mock Spectrum
        mock_es.Spectrum.return_value.return_value = np.zeros(2048)

        # Mock HPCP
        mock_es.HPCP.return_value.return_value = np.zeros(12)

        # Mock Key (returns key, scale, strength, firstToSecondRelativeStrength)
        mock_es.Key.return_value.return_value = ("C", "major", 0.85, 0.5)

        return mock_es

    def test_initialization(self):
        """Test KeyDetector initialization with custom parameters."""
        detector = KeyDetector(
            confidence_threshold=0.8,
            agreement_boost=1.3,
            disagreement_penalty=0.7,
            needs_review_threshold=0.6,
        )

        assert detector.confidence_threshold == 0.8
        assert detector.agreement_boost == 1.3
        assert detector.disagreement_penalty == 0.7
        assert detector.needs_review_threshold == 0.6

    @patch("services.analysis_service.src.key_detector.logger")
    def test_detect_key_success_with_agreement(self, mock_logger, key_detector, mock_essentia):
        """Test successful key detection when algorithms agree."""
        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = key_detector.detect_key("test.mp3")

        assert result is not None
        assert result.key == "C"
        assert result.scale == "major"
        assert result.confidence == pytest.approx(1.0)  # 0.85 * 1.2 capped at 1.0
        assert result.agreement is True
        assert result.needs_review is False
        assert result.alternative_key is None
        assert result.alternative_scale is None

        # Verify logging
        mock_logger.info.assert_called()

    @patch("services.analysis_service.src.key_detector.logger")
    def test_detect_key_with_disagreement(self, mock_logger, key_detector, mock_essentia):
        """Test key detection when algorithms disagree."""
        # Set up different results for the two algorithms
        mock_essentia.KeyExtractor.return_value.return_value = ("C", "major", 0.8)
        mock_essentia.Key.return_value.return_value = ("G", "minor", 0.75, 0.5)

        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = key_detector.detect_key("test.mp3")

        assert result is not None
        assert result.key == "C"  # Primary has higher confidence
        assert result.scale == "major"
        assert result.confidence == pytest.approx(0.64)  # 0.8 * 0.8
        assert result.agreement is False
        assert result.needs_review is True  # Below 0.7 threshold
        assert result.alternative_key == "G"
        assert result.alternative_scale == "minor"

    def test_detect_key_alternative_stronger(self, key_detector, mock_essentia):
        """Test when alternative algorithm has stronger confidence."""
        # Alternative has higher confidence
        mock_essentia.KeyExtractor.return_value.return_value = ("C", "major", 0.6)
        mock_essentia.Key.return_value.return_value = ("G", "minor", 0.9, 0.5)

        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = key_detector.detect_key("test.mp3")

        assert result is not None
        assert result.key == "G"  # Alternative has higher confidence
        assert result.scale == "minor"
        assert result.confidence == pytest.approx(0.72)  # 0.9 * 0.8
        assert result.agreement is False

    def test_detect_key_essentia_not_installed(self, key_detector):
        """Test handling when Essentia is not installed."""
        with (
            patch.dict("sys.modules", {"essentia.standard": None}),
            patch(
                "builtins.__import__",
                side_effect=ImportError("No module named essentia"),
            ),
        ):
            result = key_detector.detect_key("test.mp3")

        assert result is None

    @patch("services.analysis_service.src.key_detector.logger")
    def test_detect_key_file_error(self, mock_logger, key_detector, mock_essentia):
        """Test handling when audio file cannot be loaded."""
        mock_essentia.MonoLoader.return_value.side_effect = Exception("File not found")

        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = key_detector.detect_key("nonexistent.mp3")

        assert result is None
        mock_logger.error.assert_called()

    def test_detect_with_key_extractor(self, key_detector, mock_essentia):
        """Test KeyExtractor method directly."""
        audio = np.zeros(44100)
        mock_essentia.KeyExtractor.return_value.return_value = ("F#", "minor", 0.92)

        result = key_detector._detect_with_key_extractor(audio, mock_essentia)

        assert result[0] == "F#"
        assert result[1] == "minor"
        assert result[2] == pytest.approx(0.92)

    def test_detect_with_key_extractor_error(self, key_detector, mock_essentia):
        """Test KeyExtractor error handling."""
        audio = np.zeros(44100)
        mock_essentia.KeyExtractor.return_value.side_effect = Exception("KeyExtractor error")

        result = key_detector._detect_with_key_extractor(audio, mock_essentia)

        # Should return defaults on error
        assert result[0] == "C"
        assert result[1] == "major"
        assert result[2] == 0.0

    def test_detect_with_hpcp(self, key_detector, mock_essentia):
        """Test HPCP-based detection method directly."""
        audio = np.zeros(44100)
        mock_essentia.Key.return_value.return_value = ("Bb", "major", 0.88, 0.6)

        result = key_detector._detect_with_hpcp(audio, mock_essentia)

        assert result[0] == "Bb"
        assert result[1] == "major"
        assert result[2] == pytest.approx(0.88)

    def test_detect_with_hpcp_error(self, key_detector, mock_essentia):
        """Test HPCP detection error handling."""
        audio = np.zeros(44100)
        mock_essentia.Spectrum.return_value.side_effect = Exception("Spectrum error")

        result = key_detector._detect_with_hpcp(audio, mock_essentia)

        # Should return defaults on error
        assert result[0] == "C"
        assert result[1] == "major"
        assert result[2] == 0.0

    def test_combine_results_agreement(self, key_detector):
        """Test combining results when algorithms agree."""
        primary = ("D", "minor", 0.75)
        alternative = ("D", "minor", 0.73)

        result = key_detector._combine_results(primary, alternative)

        assert result.key == "D"
        assert result.scale == "minor"
        assert result.confidence == pytest.approx(0.9)  # 0.75 * 1.2
        assert result.agreement is True
        assert result.needs_review is False

    def test_combine_results_disagreement(self, key_detector):
        """Test combining results when algorithms disagree."""
        primary = ("E", "major", 0.82)
        alternative = ("A", "minor", 0.78)

        result = key_detector._combine_results(primary, alternative)

        assert result.key == "E"  # Primary has higher confidence
        assert result.scale == "major"
        assert result.confidence == pytest.approx(0.656)  # 0.82 * 0.8
        assert result.agreement is False
        assert result.needs_review is True
        assert result.alternative_key == "A"
        assert result.alternative_scale == "minor"

    def test_needs_review_threshold(self, key_detector):
        """Test needs_review flag based on confidence threshold."""
        # Low confidence should trigger review
        primary = ("C", "major", 0.5)
        alternative = ("C", "major", 0.5)

        result = key_detector._combine_results(primary, alternative)

        assert result.confidence == pytest.approx(0.6)  # 0.5 * 1.2
        assert result.needs_review is True  # Below 0.7 threshold

    def test_detect_key_with_segments(self, key_detector, mock_essentia):
        """Test segment-based key detection."""
        # Mock different results for different segments
        call_count = 0

        def key_extractor_side_effect():
            nonlocal call_count
            results = [
                ("C", "major", 0.85),
                ("C", "major", 0.80),
                ("G", "major", 0.75),  # One different result
            ]
            result = results[call_count % 3]
            call_count += 1
            return result

        mock_essentia.KeyExtractor.return_value.side_effect = key_extractor_side_effect
        mock_essentia.Key.return_value.return_value = ("C", "major", 0.83, 0.5)

        with patch.dict("sys.modules", {"essentia.standard": mock_essentia}):
            result = key_detector.detect_key_with_segments("test.mp3", num_segments=3)

        assert result is not None
        assert result.key == "C"  # Most common key
        assert result.scale == "major"
        assert result.agreement is True  # 2/3 segments agree

    def test_vote_on_segments_unanimous(self, key_detector):
        """Test voting when all segments agree."""
        results = [
            KeyDetectionResult("F", "minor", 0.8, agreement=True, needs_review=False),
            KeyDetectionResult("F", "minor", 0.85, agreement=True, needs_review=False),
            KeyDetectionResult("F", "minor", 0.82, agreement=True, needs_review=False),
        ]

        final_result = key_detector._vote_on_segments(results)

        assert final_result.key == "F"
        assert final_result.scale == "minor"
        # Average confidence with boost: ((0.8 + 0.85 + 0.82) / 3) * 1.2
        assert final_result.confidence == pytest.approx(0.988, rel=0.01)
        assert final_result.agreement is True

    def test_vote_on_segments_majority(self, key_detector):
        """Test voting with majority agreement."""
        results = [
            KeyDetectionResult("A", "major", 0.9, agreement=True, needs_review=False),
            KeyDetectionResult("A", "major", 0.85, agreement=True, needs_review=False),
            KeyDetectionResult("D", "minor", 0.7, agreement=False, needs_review=True),
        ]

        final_result = key_detector._vote_on_segments(results)

        assert final_result.key == "A"
        assert final_result.scale == "major"
        # 2/3 agree, so gets boost
        assert final_result.agreement is True

    def test_vote_on_segments_no_clear_winner(self, key_detector):
        """Test voting when there's no clear majority."""
        results = [
            KeyDetectionResult("C", "major", 0.8, agreement=True, needs_review=False),
            KeyDetectionResult("G", "major", 0.85, agreement=True, needs_review=False),
            KeyDetectionResult("F", "major", 0.82, agreement=True, needs_review=False),
        ]

        final_result = key_detector._vote_on_segments(results)

        # Should pick the one with highest confidence
        assert final_result.key == "G"
        assert final_result.scale == "major"
        assert final_result.agreement is False  # Less than 2/3 agreement

    def test_key_names_mapping(self):
        """Test that KEY_NAMES constant is properly defined."""
        detector = KeyDetector()
        assert len(detector.KEY_NAMES) == 12
        assert "C" in detector.KEY_NAMES
        assert "F#" in detector.KEY_NAMES
        assert "Bb" in detector.KEY_NAMES
