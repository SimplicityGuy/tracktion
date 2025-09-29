"""
Comprehensive edge case tests for BPM detection algorithm.

Tests handle various failure scenarios, boundary conditions, and error cases
with complete mocking to avoid file dependencies and ensure reliability.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.analysis_service.src.bpm_detector import BPMDetector


class TestBPMDetectorEdgeCases:
    """Comprehensive edge case test suite for BPMDetector class."""

    def setup_method(self):
        """Set up test fixtures with mocked Essentia components."""
        with (
            patch("services.analysis_service.src.bpm_detector.es.RhythmExtractor2013") as mock_rhythm,
            patch("services.analysis_service.src.bpm_detector.es.PercivalBpmEstimator") as mock_percival,
        ):
            self.mock_rhythm_instance = Mock()
            self.mock_percival_instance = Mock()
            mock_rhythm.return_value = self.mock_rhythm_instance
            mock_percival.return_value = self.mock_percival_instance

            self.detector = BPMDetector(confidence_threshold=0.7, agreement_tolerance=5.0)

    def test_corrupted_audio_file_handling(self):
        """Test handling of corrupted audio files that can't be loaded."""
        with (
            patch("services.analysis_service.src.bpm_detector.es.MonoLoader") as mock_loader,
            tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp,
        ):
            tmp_path = tmp.name

            # Mock corrupted file behavior - loader raises exception
            mock_loader.side_effect = RuntimeError("Failed to decode audio file: corrupted header")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*corrupted header"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_silent_audio_handling(self, mock_loader):
        """Test handling of completely silent audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Mock silent audio - all zeros
            silent_audio = np.zeros(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=silent_audio)

            # Mock rhythm extractor to return low confidence for silent audio
            self.mock_rhythm_instance.return_value = (
                0.0,  # BPM (silence often returns 0 or very low BPM)
                np.array([]),  # No beats detected
                0.0,  # Zero confidence
                np.array([]),  # No estimates
                np.array([]),  # No intervals
            )

            # Mock fallback also struggling with silence
            self.mock_percival_instance.return_value = 60.0  # Default fallback value

            try:
                result = self.detector.detect_bpm(tmp_path)

                # Silent audio should have very low confidence and need review
                assert result["confidence"] <= 0.3  # After boosting, still low
                assert result["needs_review"] is True
                assert result["algorithm"] in ("fallback", "consensus")

            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_extremely_quiet_audio(self, mock_loader):
        """Test handling of extremely quiet but not silent audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Mock very quiet audio - very small amplitude
            quiet_audio = np.ones(44100, dtype=np.float32) * 0.001  # Very quiet
            mock_loader.return_value = Mock(return_value=quiet_audio)

            # Mock rhythm extractor with poor performance on quiet audio
            self.mock_rhythm_instance.return_value = (
                120.0,  # Detected BPM
                np.array([0.5, 1.0, 1.5]),  # Some beats detected
                0.2,  # Very low confidence
                np.array([]),
                np.array([0.5, 0.5]),  # Beat intervals
            )

            self.mock_percival_instance.return_value = 118.0  # Close agreement

            try:
                result = self.detector.detect_bpm(tmp_path)

                # Should boost confidence due to agreement but still flag for review
                assert result["confidence"] > 0.2  # Boosted
                assert result["needs_review"] is True
                assert result["algorithm"] == "consensus"

            finally:
                Path(tmp_path).unlink()

    def test_invalid_sample_rates(self):
        """Test handling of invalid sample rate values."""
        # BPMDetector doesn't validate sample rates in constructor
        # These will be accepted but may cause issues during audio loading

        # Test negative sample rate (accepted in constructor)
        detector = BPMDetector(sample_rate=-44100)
        assert detector.sample_rate == -44100

        # Test zero sample rate (accepted in constructor)
        detector = BPMDetector(sample_rate=0)
        assert detector.sample_rate == 0

        # Test extremely high sample rate (should work but might be impractical)
        detector = BPMDetector(sample_rate=1000000)
        assert detector.sample_rate == 1000000

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_invalid_sample_rate_during_loading(self, mock_loader):
        """Test that invalid sample rates cause errors during audio loading."""
        detector = BPMDetector(sample_rate=-44100)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Mock MonoLoader to raise error with negative sample rate
            mock_loader.side_effect = ValueError("Sample rate must be positive")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Sample rate must be positive"):
                    detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_memory_pressure_scenarios(self, mock_loader):
        """Test handling of memory-related errors during processing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Mock memory error during audio loading
            mock_loader.side_effect = MemoryError("Cannot allocate memory for audio buffer")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Cannot allocate memory"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_memory_pressure_during_processing(self, mock_loader):
        """Test memory errors during rhythm extraction."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Audio loads fine
            mock_audio = np.ones(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=mock_audio)

            # Memory error during rhythm extraction
            self.mock_rhythm_instance.side_effect = MemoryError("Out of memory during FFT processing")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Out of memory during FFT"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_algorithm_disagreement_extreme(self, mock_loader):
        """Test handling of extreme disagreement between algorithms."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            mock_audio = np.ones(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=mock_audio)

            # Primary algorithm detects slow BPM with low confidence and unstable intervals
            self.mock_rhythm_instance.return_value = (
                60.0,  # Slow BPM
                np.array([1.0, 2.0, 3.0]),
                0.5,  # Low confidence
                np.array([]),
                np.array([1.0, 1.5]),  # Unstable intervals (high variation)
            )

            # Fallback detects very fast BPM (extreme disagreement)
            self.mock_percival_instance.return_value = 180.0  # Way outside tolerance

            try:
                result = self.detector.detect_bpm(tmp_path)

                # Should use fallback due to instability
                assert result["bpm"] == 180.0
                assert result["algorithm"] == "fallback"
                assert result["needs_review"] is True

            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_timeout_scenarios(self, mock_loader):
        """Test handling of processing timeouts."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            mock_audio = np.ones(44100 * 60, dtype=np.float32)  # 1 minute of audio
            mock_loader.return_value = Mock(return_value=mock_audio)

            # Mock a slow processing scenario
            def slow_rhythm_extraction(*args):
                time.sleep(0.1)  # Simulate slow processing
                raise TimeoutError("Processing timeout after 30 seconds")

            self.mock_rhythm_instance.side_effect = slow_rhythm_extraction

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*timeout"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_timeout_during_fallback(self, mock_loader):
        """Test timeout during fallback algorithm execution."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            mock_audio = np.ones(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=mock_audio)

            # Primary works but low confidence
            self.mock_rhythm_instance.return_value = (
                120.0,
                np.array([0.5, 1.0]),
                0.4,  # Triggers fallback
                np.array([]),
                np.array([0.5]),
            )

            # Fallback times out
            self.mock_percival_instance.side_effect = TimeoutError("Percival algorithm timeout")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*timeout"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    def test_invalid_audio_formats(self):
        """Test handling of invalid or unsupported audio formats."""
        # Test with non-audio file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"This is not an audio file")
            tmp_path = tmp.name

            try:
                with pytest.raises(RuntimeError, match="BPM detection failed"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_empty_audio_data(self, mock_loader):
        """Test handling of files that load but contain no audio data."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Mock loader returning empty array
            mock_loader.return_value = Mock(return_value=np.array([], dtype=np.float32))

            try:
                with pytest.raises(RuntimeError, match="Loaded audio is empty"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_audio_with_nan_values(self, mock_loader):
        """Test handling of audio containing NaN or infinite values."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Audio with NaN values
            corrupt_audio = np.ones(44100, dtype=np.float32)
            corrupt_audio[1000:2000] = np.nan
            mock_loader.return_value = Mock(return_value=corrupt_audio)

            # Essentia might handle NaN gracefully or raise error
            self.mock_rhythm_instance.side_effect = RuntimeError("NaN values in audio signal")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*NaN values"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_audio_with_infinite_values(self, mock_loader):
        """Test handling of audio containing infinite values."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Audio with infinite values
            corrupt_audio = np.ones(44100, dtype=np.float32)
            corrupt_audio[500:1500] = np.inf
            mock_loader.return_value = Mock(return_value=corrupt_audio)

            self.mock_rhythm_instance.side_effect = RuntimeError("Infinite values in audio signal")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Infinite values"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_extremely_short_audio(self, mock_loader):
        """Test handling of extremely short audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Very short audio (0.1 seconds)
            short_audio = np.ones(4410, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=short_audio)

            # Short audio typically results in poor BPM detection
            self.mock_rhythm_instance.return_value = (
                0.0,  # No BPM detected
                np.array([]),  # No beats
                0.0,  # Zero confidence
                np.array([]),
                np.array([]),
            )

            self.mock_percival_instance.return_value = 120.0  # Fallback guess

            try:
                result = self.detector.detect_bpm(tmp_path)

                # Should use fallback and need review
                assert result["algorithm"] == "fallback"
                assert result["needs_review"] is True
                assert result["bpm"] == 120.0

            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_extremely_long_audio(self, mock_loader):
        """Test handling of very long audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Very long audio (10 minutes)
            long_audio = np.ones(44100 * 600, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=long_audio)

            # Long audio might have variable BPM
            self.mock_rhythm_instance.return_value = (
                125.0,
                np.array([i * 0.48 for i in range(1250)]),  # Many beats
                0.6,  # Medium confidence
                np.array([]),
                np.array([0.48] * 1249),  # Consistent but not perfect
            )

            self.mock_percival_instance.return_value = 123.0  # Close agreement

            try:
                result = self.detector.detect_bpm(tmp_path)

                # Should achieve consensus
                assert result["algorithm"] == "consensus"
                assert result["confidence"] > 0.6  # Boosted
                assert abs(result["bpm"] - 125.0) < 0.1

            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_audio_with_extreme_dynamics(self, mock_loader):
        """Test handling of audio with extreme dynamic range."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Audio with extreme dynamics (very loud and very quiet sections)
            dynamic_audio = np.ones(44100, dtype=np.float32)
            dynamic_audio[:22050] *= 0.001  # Very quiet first half
            dynamic_audio[22050:] *= 0.999  # Very loud second half
            mock_loader.return_value = Mock(return_value=dynamic_audio)

            # Extreme dynamics might confuse rhythm detection
            self.mock_rhythm_instance.return_value = (
                130.0,
                np.array([0.25, 0.75, 1.25]),  # Irregular due to dynamics
                0.45,  # Low confidence
                np.array([]),
                np.array([0.5, 0.5]),  # Intervals might be affected
            )

            self.mock_percival_instance.return_value = 133.0  # Within tolerance (3 BPM difference)

            try:
                result = self.detector.detect_bpm(tmp_path)

                # Should use consensus but need review due to low original confidence
                assert result["algorithm"] == "consensus"
                assert result["needs_review"] is True

            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_concurrent_algorithm_failures(self, mock_loader):
        """Test scenario where both algorithms fail."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            mock_audio = np.ones(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=mock_audio)

            # Primary algorithm fails
            self.mock_rhythm_instance.side_effect = RuntimeError("Rhythm extraction failed")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Rhythm extraction failed"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_fallback_algorithm_failure(self, mock_loader):
        """Test scenario where primary works but fallback fails when needed."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            mock_audio = np.ones(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=mock_audio)

            # Primary has low confidence
            self.mock_rhythm_instance.return_value = (
                120.0,
                np.array([0.5, 1.0]),
                0.3,  # Low confidence triggers fallback
                np.array([]),
                np.array([0.5]),
            )

            # Fallback fails
            self.mock_percival_instance.side_effect = RuntimeError("Percival estimation failed")

            try:
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Percival estimation failed"):
                    self.detector.detect_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

    def test_edge_case_confidence_values(self):
        """Test handling of edge case confidence values."""
        # Test with very small confidence threshold
        detector = BPMDetector(confidence_threshold=0.01)
        assert detector.confidence_threshold == 0.01

        # Test with confidence threshold of 1.0 (maximum)
        detector = BPMDetector(confidence_threshold=1.0)
        assert detector.confidence_threshold == 1.0

        # Test with confidence threshold above 1.0 (should be allowed)
        detector = BPMDetector(confidence_threshold=1.5)
        assert detector.confidence_threshold == 1.5

    def test_edge_case_agreement_tolerance(self):
        """Test handling of edge case agreement tolerance values."""
        # Very strict tolerance
        detector = BPMDetector(agreement_tolerance=0.1)
        assert detector.agreement_tolerance == 0.1

        # Very loose tolerance
        detector = BPMDetector(agreement_tolerance=50.0)
        assert detector.agreement_tolerance == 50.0

        # Zero tolerance (algorithms must match exactly)
        detector = BPMDetector(agreement_tolerance=0.0)
        assert detector.agreement_tolerance == 0.0

    def test_beat_intervals_edge_cases(self):
        """Test tempo stability with various edge case beat intervals."""
        # Test with single interval
        single_interval = np.array([0.5])
        assert self.detector._is_tempo_stable(single_interval) is False

        # Test with empty intervals
        empty_intervals = np.array([])
        assert self.detector._is_tempo_stable(empty_intervals) is False

        # Test with all identical intervals (perfectly stable)
        perfect_intervals = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        assert self.detector._is_tempo_stable(perfect_intervals) is True

        # Test with one outlier
        outlier_intervals = np.array([0.5, 0.5, 2.0, 0.5, 0.5])
        assert self.detector._is_tempo_stable(outlier_intervals) is False

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_confidence_normalization_edge_cases(self, mock_loader):
        """Test confidence normalization with extreme values."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            mock_audio = np.ones(44100, dtype=np.float32)
            mock_loader.return_value = Mock(return_value=mock_audio)

            test_cases = [
                (-1.0, 0.3),  # Negative confidence -> 0.0, but boosted to 0.3 due to consensus
                (0.0, 0.3),  # Zero confidence -> 0.0, but boosted to 0.3 due to consensus
                (0.5, 0.8),  # Low confidence -> boosted to 0.8 due to consensus
                (1.0, 1.0),  # High confidence -> unchanged (no fallback)
                (5.0, 1.0),  # Very high confidence -> normalized to 1.0 (no fallback)
                (10.0, 1.0),  # Extremely high confidence -> clamped to 1.0 (no fallback)
            ]

            for input_confidence, expected_confidence in test_cases:
                self.mock_rhythm_instance.return_value = (
                    120.0,
                    np.array([0.5, 1.0]),
                    input_confidence,
                    np.array([]),
                    np.array([0.5]),
                )

                # Setup fallback for low confidence cases
                self.mock_percival_instance.return_value = 122.0  # Close agreement

                result = self.detector.detect_bpm(tmp_path)
                assert abs(result["confidence"] - expected_confidence) < 0.01, (
                    f"Confidence normalization failed: input={input_confidence}, "
                    f"expected={expected_confidence}, got={result['confidence']}"
                )

            Path(tmp_path).unlink()

    def test_file_permissions_error(self):
        """Test handling of file permission errors."""
        # This test might be platform-specific, so we'll mock the behavior
        with (
            patch("services.analysis_service.src.bpm_detector.Path") as mock_path,
            patch("services.analysis_service.src.bpm_detector.es.MonoLoader") as mock_loader,
        ):
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_path.return_value = mock_file

            # Mock permission error
            mock_loader.side_effect = PermissionError("Permission denied accessing audio file")

            with pytest.raises(RuntimeError, match=r"BPM detection failed.*Permission denied"):
                self.detector.detect_bpm("/restricted/file.mp3")

    def test_network_file_timeout(self):
        """Test handling of network file access timeouts."""
        with (
            patch("services.analysis_service.src.bpm_detector.Path") as mock_path,
            patch("services.analysis_service.src.bpm_detector.es.MonoLoader") as mock_loader,
        ):
            # Mock file exists check to return True for network URL
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_path.return_value = mock_file

            # Mock network timeout during file loading
            mock_loader.side_effect = TimeoutError("Network timeout while loading remote audio file")

            with pytest.raises(RuntimeError, match=r"BPM detection failed.*Network timeout"):
                self.detector.detect_bpm("https://example.com/audio.mp3")

    def test_detector_with_invalid_config(self):
        """Test detector initialization with invalid configuration parameters."""
        # Mock config with invalid values
        mock_config = Mock()
        mock_config.confidence_threshold = -0.5  # Invalid negative threshold
        mock_config.agreement_tolerance = -10.0  # Invalid negative tolerance

        # Constructor should accept these (validation might be elsewhere)
        detector = BPMDetector(config=mock_config)
        assert detector.confidence_threshold == -0.5
        assert detector.agreement_tolerance == -10.0

    @patch("services.analysis_service.src.bpm_detector.es.MonoLoader")
    def test_detect_bpm_with_confidence_error_handling(self, mock_loader):
        """Test that detect_bpm_with_confidence properly handles errors from detect_bpm."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

            # Mock an error during detection
            mock_loader.side_effect = RuntimeError("Audio processing failed")

            try:
                # Should propagate the error from detect_bpm
                with pytest.raises(RuntimeError, match=r"BPM detection failed.*Audio processing failed"):
                    self.detector.detect_bpm_with_confidence(tmp_path)
            finally:
                Path(tmp_path).unlink()
