"""
Unit tests for BPM detection module using complete mocking.

Tests BPM detection accuracy, confidence scoring, and error handling
using mocked Essentia functions for reliable test results.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.analysis_service.src.bpm_detector import BPMDetector


class TestBPMDetectorMocked:
    """Test suite for BPMDetector class with complete mocking."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock Essentia extractors
        with (
            patch("services.analysis_service.src.bpm_detector.es.RhythmExtractor2013") as mock_rhythm,
            patch("services.analysis_service.src.bpm_detector.es.PercivalBpmEstimator") as mock_percival,
        ):
            self.mock_rhythm_instance = Mock()
            self.mock_percival_instance = Mock()
            mock_rhythm.return_value = self.mock_rhythm_instance
            mock_percival.return_value = self.mock_percival_instance

            self.detector = BPMDetector(confidence_threshold=0.7, agreement_tolerance=5.0)

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_high_confidence(self, mock_loader):
        """Test BPM detection with high confidence primary algorithm."""
        # Setup mock audio loader
        mock_audio = np.ones(44100, dtype=np.float32)  # 1 second of dummy audio
        mock_loader_instance = Mock(return_value=mock_audio)
        mock_loader.return_value = mock_loader_instance

        # Setup rhythm extractor response
        mock_beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mock_intervals = np.array([0.5, 0.5, 0.5, 0.5])
        self.mock_rhythm_instance.return_value = (
            128.0,  # BPM
            mock_beats,  # Beat positions
            0.95,  # Confidence
            np.array([]),  # Estimates (unused)
            mock_intervals,  # Beat intervals
        )

        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

        # Verify results
        assert result["bpm"] == 128.0
        assert result["confidence"] == 0.95
        assert result["algorithm"] == "primary"
        assert result["needs_review"] is False
        assert len(result["beats"]) == 5

        # Verify mocks were called correctly
        mock_loader.assert_called_once_with(filename=tmp_path, sampleRate=44100)
        self.mock_rhythm_instance.assert_called_once()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_low_confidence_with_agreement(self, mock_loader):
        """Test BPM detection with low confidence but algorithm agreement."""
        # Setup mock audio loader
        mock_audio = np.ones(44100, dtype=np.float32)
        mock_loader.return_value = Mock(return_value=mock_audio)

        # Setup rhythm extractor with low confidence
        mock_beats = np.array([0.5, 1.0, 1.5, 2.0])
        mock_intervals = np.array([0.5, 0.5, 0.5])
        self.mock_rhythm_instance.return_value = (
            120.0,
            mock_beats,
            0.6,
            np.array([]),
            mock_intervals,
        )

        # Setup Percival with similar BPM (within tolerance)
        self.mock_percival_instance.return_value = 122.0

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

        assert result["bpm"] == 120.0
        assert abs(result["confidence"] - 0.9) < 0.01  # Boosted due to agreement (with float tolerance)
        assert result["algorithm"] == "consensus"
        assert result["needs_review"] is False

        # Both algorithms should be called
        self.mock_rhythm_instance.assert_called_once()
        self.mock_percival_instance.assert_called_once()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_low_confidence_disagreement(self, mock_loader):
        """Test BPM detection with low confidence and algorithm disagreement."""
        # Setup mock audio loader
        mock_audio = np.ones(44100, dtype=np.float32)
        mock_loader.return_value = Mock(return_value=mock_audio)

        # Setup rhythm extractor with low confidence and stable tempo
        mock_beats = np.array([0.5, 1.0, 1.5, 2.0])
        mock_intervals = np.array([0.5, 0.5, 0.5])  # Stable intervals
        self.mock_rhythm_instance.return_value = (
            120.0,
            mock_beats,
            0.6,
            np.array([]),
            mock_intervals,
        )

        # Setup Percival with very different BPM
        self.mock_percival_instance.return_value = 140.0  # Outside tolerance

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

        # Should keep primary due to stable tempo
        assert result["bpm"] == 120.0
        assert result["confidence"] == 0.6
        assert result["algorithm"] == "primary"
        assert result["needs_review"] is True

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_unstable_tempo_uses_fallback(self, mock_loader):
        """Test that unstable tempo triggers fallback algorithm."""
        # Setup mock audio loader
        mock_audio = np.ones(44100, dtype=np.float32)
        mock_loader.return_value = Mock(return_value=mock_audio)

        # Setup rhythm extractor with low confidence and unstable tempo
        mock_beats = np.array([0.5, 1.1, 1.5, 2.3])  # Irregular beats
        mock_intervals = np.array([0.6, 0.4, 0.8])  # Highly variable intervals
        self.mock_rhythm_instance.return_value = (
            120.0,
            mock_beats,
            0.6,
            np.array([]),
            mock_intervals,
        )

        # Setup Percival with different BPM
        self.mock_percival_instance.return_value = 130.0

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

        # Should use fallback due to unstable primary
        assert result["bpm"] == 130.0
        assert result["algorithm"] == "fallback"
        assert result["needs_review"] is True

    def test_detect_bpm_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            self.detector.detect_bpm("/nonexistent/file.mp3")

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_empty_audio(self, mock_loader):
        """Test error handling for empty audio."""
        # Setup mock to return empty array
        mock_loader.return_value = Mock(return_value=np.array([], dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="Loaded audio is empty"):
                self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_processing_error(self, mock_loader):
        """Test error handling during processing."""
        # Setup mock to raise error
        mock_loader.return_value = Mock(side_effect=RuntimeError("Audio loading failed"))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="BPM detection failed"):
                self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_is_tempo_stable(self):
        """Test tempo stability calculation."""
        # Stable tempo (low variation)
        stable_intervals = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        assert self.detector._is_tempo_stable(stable_intervals) is True

        # Unstable tempo (high variation)
        unstable_intervals = np.array([0.3, 0.7, 0.4, 0.8, 0.2])
        assert self.detector._is_tempo_stable(unstable_intervals) is False

        # Edge case: too few intervals
        few_intervals = np.array([0.5])
        assert self.detector._is_tempo_stable(few_intervals) is False

        # Edge case: zero mean
        zero_mean = np.array([0.0, 0.0, 0.0])
        assert self.detector._is_tempo_stable(zero_mean) is False

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_with_confidence(self, mock_loader):
        """Test simplified interface matching Story 2.2 pattern."""
        # Setup mock
        mock_audio = np.ones(44100, dtype=np.float32)
        mock_loader.return_value = Mock(return_value=mock_audio)

        mock_beats = np.array([0.5, 1.0, 1.5])
        mock_intervals = np.array([0.5, 0.5])
        self.mock_rhythm_instance.return_value = (
            128.0,
            mock_beats,
            0.85,
            np.array([]),
            mock_intervals,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.detector.detect_bpm_with_confidence(tmp_path)
        finally:
            Path(tmp_path).unlink()

        # Should return simplified format
        assert "bpm" in result
        assert "confidence" in result
        assert "needs_review" in result
        assert result["bpm"] == 128.0
        assert result["confidence"] == 0.85
        assert result["needs_review"] is False

    def test_bpm_detector_initialization(self):
        """Test BPMDetector initialization with different configurations."""
        # Test with custom config
        detector = BPMDetector(confidence_threshold=0.8, agreement_tolerance=3.0, sample_rate=48000)
        assert detector.confidence_threshold == 0.8
        assert detector.agreement_tolerance == 3.0
        assert detector.sample_rate == 48000

        # Test with config object (mock)
        mock_config = Mock()
        mock_config.confidence_threshold = 0.9
        mock_config.agreement_tolerance = 2.0
        detector = BPMDetector(config=mock_config)
        assert detector.confidence_threshold == 0.9
        assert detector.agreement_tolerance == 2.0

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_confidence_normalization(self, mock_loader):
        """Test that confidence values are properly normalized."""
        # Setup mock
        mock_audio = np.ones(44100, dtype=np.float32)
        mock_loader.return_value = Mock(return_value=mock_audio)

        # Test with confidence > 1.0 (needs normalization)
        mock_beats = np.array([0.5, 1.0])
        mock_intervals = np.array([0.5])
        self.mock_rhythm_instance.return_value = (
            120.0,
            mock_beats,
            3.5,
            np.array([]),
            mock_intervals,  # High confidence value
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.detector.detect_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

        # Confidence should be normalized to 0-1 range
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] == 0.7  # 3.5 / 5.0 = 0.7


class TestBPMReferenceFiles:
    """Test BPM detection with reference audio files (if available)."""

    @pytest.fixture
    def detector(self):
        """Create BPM detector instance."""
        return BPMDetector(confidence_threshold=0.7, agreement_tolerance=5.0)

    @pytest.fixture
    def test_files_dir(self):
        """Get path to test files directory."""
        return Path(__file__).parent.parent / "fixtures"

    def test_reference_files_if_available(self, detector, test_files_dir):
        """Test with actual reference files if they exist."""
        reference_files = {
            "test_120bpm_rock.wav": (120.0, 5.0),
            "test_128bpm_electronic.wav": (128.0, 5.0),
            "test_140bpm_dnb.wav": (140.0, 10.0),  # Higher tolerance for fast BPM
        }

        tested_count = 0
        for filename, (expected_bpm, tolerance) in reference_files.items():
            file_path = test_files_dir / filename
            if file_path.exists():
                result = detector.detect_bpm(str(file_path))

                # Check BPM is within tolerance
                assert abs(result["bpm"] - expected_bpm) <= tolerance, (
                    f"BPM detection failed for {filename}: expected {expected_bpm}±{tolerance}, got {result['bpm']}"
                )

                # Basic structure checks
                assert "confidence" in result
                assert "algorithm" in result
                assert 0.0 <= result["confidence"] <= 1.0

                tested_count += 1

        if tested_count == 0:
            pytest.skip("No reference test files found")
