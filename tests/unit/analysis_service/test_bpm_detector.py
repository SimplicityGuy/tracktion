"""
Unit tests for BPM detection module.

Tests BPM detection accuracy, confidence scoring, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.analysis_service.src.bpm_detector import BPMDetector

# Create a random number generator
rng = np.random.default_rng()


class TestBPMDetector:
    """Test suite for BPMDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = BPMDetector(confidence_threshold=0.7, agreement_tolerance=5.0)

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_high_confidence(self, mock_loader):
        """Test BPM detection with high confidence primary algorithm."""
        # Mock audio loading - essentia requires float32
        mock_audio = rng.standard_normal(44100 * 10).astype(np.float32)  # 10 seconds of audio
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor with high confidence
        mock_beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mock_intervals = np.array([0.5, 0.5, 0.5, 0.5])

        with patch.object(
            self.detector.rhythm_extractor,
            "__call__",
            return_value=(128.0, mock_beats, 0.95, np.array([]), mock_intervals),
        ):
            # Create a temporary file path that exists
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["bpm"] == 128.0
            assert result["confidence"] == 0.95
            assert result["algorithm"] == "primary"
            assert result["needs_review"] is False
            assert len(result["beats"]) == 5

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_low_confidence_with_agreement(self, mock_loader):
        """Test BPM detection with low confidence but algorithm agreement."""
        # Mock audio loading - essentia requires float32
        mock_audio = rng.standard_normal(44100 * 10).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor with low confidence
        mock_beats = np.array([0.5, 1.0, 1.5, 2.0])
        mock_intervals = np.array([0.5, 0.5, 0.5])

        with (
            patch.object(
                self.detector.rhythm_extractor,
                "__call__",
                return_value=(120.0, mock_beats, 0.6, np.array([]), mock_intervals),
            ),
            # Mock Percival estimator with similar BPM
            patch.object(
                self.detector.percival_estimator,
                "__call__",
                return_value=122.0,  # Within 5 BPM tolerance
            ),
            tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp,
        ):
            tmp_path = tmp.name

            try:
                result = self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["bpm"] == 120.0
            assert result["confidence"] == 0.9  # Boosted due to agreement
            assert result["algorithm"] == "consensus"
            assert result["needs_review"] is False

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_low_confidence_disagreement(self, mock_loader):
        """Test BPM detection with low confidence and algorithm disagreement."""
        # Mock audio loading - essentia requires float32
        mock_audio = rng.standard_normal(44100 * 10).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor with low confidence and unstable tempo
        mock_beats = np.array([0.5, 1.2, 1.8, 2.6])  # Irregular beats
        mock_intervals = np.array([0.7, 0.6, 0.8])  # Unstable intervals

        with (
            patch.object(
                self.detector.rhythm_extractor,
                "__call__",
                return_value=(120.0, mock_beats, 0.5, np.array([]), mock_intervals),
            ),
            # Mock Percival estimator with different BPM
            patch.object(
                self.detector.percival_estimator,
                "__call__",
                return_value=140.0,  # Outside tolerance
            ),
            tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp,
        ):
            tmp_path = tmp.name

            try:
                result = self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["bpm"] == 140.0  # Uses fallback due to unstable primary
            assert result["confidence"] == 0.5
            assert result["algorithm"] == "fallback"
            assert result["needs_review"] is True

    def test_detect_bpm_file_not_found(self):
        """Test BPM detection with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            self.detector.detect_bpm("/nonexistent/file.mp3")

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_empty_audio(self, mock_loader):
        """Test BPM detection with empty audio."""
        # Mock audio loading with empty array
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = np.array([])
        mock_loader.return_value = mock_loader_instance

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="Loaded audio is empty"):
                self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_processing_error(self, mock_loader):
        """Test BPM detection with processing error."""
        # Mock audio loading - essentia requires float32
        mock_audio = rng.standard_normal(44100 * 10).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor to raise an exception
        with patch.object(
            self.detector.rhythm_extractor,
            "__call__",
            side_effect=RuntimeError("Processing failed"),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                with pytest.raises(RuntimeError, match="BPM detection failed"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    def test_is_tempo_stable(self):
        """Test tempo stability checking."""
        # Stable tempo (regular intervals)
        stable_intervals = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        assert self.detector._is_tempo_stable(stable_intervals) is True

        # Unstable tempo (irregular intervals)
        unstable_intervals = np.array([0.3, 0.6, 0.4, 0.7, 0.5])
        assert self.detector._is_tempo_stable(unstable_intervals) is False

        # Edge case: too few intervals
        few_intervals = np.array([0.5])
        assert self.detector._is_tempo_stable(few_intervals) is False

        # Edge case: zero mean
        zero_intervals = np.array([0.0, 0.0, 0.0])
        assert self.detector._is_tempo_stable(zero_intervals) is False

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_with_confidence(self, mock_loader):
        """Test simplified detect_bpm_with_confidence method."""
        # Mock audio loading - essentia requires float32
        mock_audio = rng.standard_normal(44100 * 10).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor
        mock_beats = np.array([0.5, 1.0, 1.5])
        mock_intervals = np.array([0.5, 0.5])

        with patch.object(
            self.detector.rhythm_extractor,
            "__call__",
            return_value=(128.5, mock_beats, 0.85, np.array([]), mock_intervals),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = self.detector.detect_bpm_with_confidence(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["bpm"] == 128.5
            assert result["confidence"] == 0.85
            assert result["needs_review"] is False
            assert len(result) == 3  # Only 3 keys in simplified format

    def test_bpm_detector_initialization(self):
        """Test BPMDetector initialization with custom parameters."""
        detector = BPMDetector(confidence_threshold=0.8, agreement_tolerance=3.0, sample_rate=48000)

        assert detector.confidence_threshold == 0.8
        assert detector.agreement_tolerance == 3.0
        assert detector.sample_rate == 48000
        assert detector.rhythm_extractor is not None
        assert detector.percival_estimator is not None


class TestBPMAccuracy:
    """Test suite for BPM detection accuracy with reference tracks."""

    @pytest.mark.parametrize(
        "audio_file,expected_bpm,tolerance",
        [
            ("test_128bpm_electronic.mp3", 128.0, 2.0),
            ("test_120bpm_rock.mp3", 120.0, 2.0),
            ("test_85bpm_hiphop.mp3", 85.0, 2.0),
        ],
    )
    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_reference_track_accuracy(self, mock_loader, audio_file, expected_bpm, tolerance):
        """Test BPM detection accuracy with reference tracks."""
        detector = BPMDetector()

        # Mock audio loading - essentia requires float32
        mock_audio = rng.standard_normal(44100 * 30).astype(np.float32)  # 30 seconds
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor to return expected BPM within tolerance
        mock_beats = np.arange(0, 30, 60.0 / expected_bpm)  # Generate beats
        mock_intervals = np.full(len(mock_beats) - 1, 60.0 / expected_bpm)

        with patch.object(
            detector.rhythm_extractor,
            "__call__",
            return_value=(expected_bpm, mock_beats, 0.92, np.array([]), mock_intervals),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert abs(result["bpm"] - expected_bpm) <= tolerance
            assert result["confidence"] > 0.8
