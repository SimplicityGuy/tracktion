"""
Unit tests for temporal BPM analysis module.

Tests windowed BPM analysis, tempo stability detection, and tempo changes.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.analysis_service.src.temporal_analyzer import TemporalAnalyzer


class TestTemporalAnalyzer:
    """Test suite for TemporalAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TemporalAnalyzer(window_size=10.0, hop_size=5.0, start_duration=30.0, end_duration=30.0)

    @patch("services.analysis_service.src.temporal_analyzer.es.MonoLoader")
    def test_analyze_constant_tempo(self, mock_loader):
        """Test analysis of track with constant tempo."""
        # Mock audio loading - 60 seconds of audio
        mock_audio = np.random.default_rng().standard_normal(44100 * 60).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor to return constant BPM
        with patch.object(
            self.analyzer.rhythm_extractor,
            "__call__",
            return_value=(128.0, np.array([]), 0.9, np.array([]), np.array([])),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = self.analyzer.analyze_temporal_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["average_bpm"] == 128.0
            assert result["start_bpm"] == 128.0
            assert result["end_bpm"] == 128.0
            assert result["stability_score"] == 1.0
            assert result["is_variable_tempo"] is False
            assert len(result["tempo_changes"]) == 0

    @patch("services.analysis_service.src.temporal_analyzer.es.MonoLoader")
    def test_analyze_variable_tempo(self, mock_loader):
        """Test analysis of track with variable tempo."""
        # Mock audio loading
        mock_audio = np.random.default_rng().standard_normal(44100 * 60).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor to return variable BPM
        bpm_sequence = [120.0, 120.0, 125.0, 130.0, 135.0, 140.0]
        call_count = 0

        def mock_rhythm_extract(audio):
            nonlocal call_count
            bpm = bpm_sequence[min(call_count, len(bpm_sequence) - 1)]
            call_count += 1
            return (bpm, np.array([]), 0.8, np.array([]), np.array([]))

        with patch.object(self.analyzer.rhythm_extractor, "__call__", side_effect=mock_rhythm_extract):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = self.analyzer.analyze_temporal_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["average_bpm"] > 120.0  # Should be higher than start
            assert result["stability_score"] < 1.0  # Not perfectly stable
            assert result["is_variable_tempo"] is True
            assert len(result["tempo_changes"]) > 0

    @patch("services.analysis_service.src.temporal_analyzer.es.MonoLoader")
    def test_analyze_dj_mix(self, mock_loader):
        """Test analysis of DJ mix with tempo transitions."""
        # Mock audio loading - 5 minutes
        mock_audio = np.random.default_rng().standard_normal(44100 * 300).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor for DJ mix pattern
        def mock_rhythm_extract(audio):
            # Simulate transition from 128 to 140 BPM
            audio_len = len(audio) / 44100
            if audio_len < 20:  # Start or short window
                return (128.0, np.array([]), 0.9, np.array([]), np.array([]))
            # End or later window
            return (140.0, np.array([]), 0.9, np.array([]), np.array([]))

        with patch.object(self.analyzer.rhythm_extractor, "__call__", side_effect=mock_rhythm_extract):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = self.analyzer.analyze_temporal_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            assert result["start_bpm"] == 128.0
            assert result["end_bpm"] == 140.0
            assert result["start_bpm"] != result["end_bpm"]  # Different tempos

    @patch("services.analysis_service.src.temporal_analyzer.es.MonoLoader")
    def test_analyze_beatless_track(self, mock_loader):
        """Test analysis of beatless/ambient track."""
        # Mock audio loading
        mock_audio = np.random.default_rng().standard_normal(44100 * 60).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        # Mock rhythm extractor with low confidence
        with patch.object(
            self.analyzer.rhythm_extractor,
            "__call__",
            return_value=(60.0, np.array([]), 0.3, np.array([]), np.array([])),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = self.analyzer.analyze_temporal_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            # Low confidence results should be filtered out
            assert result["average_bpm"] is None
            assert result["start_bpm"] is None
            assert result["end_bpm"] is None
            assert result["stability_score"] == 0.0

    def test_analyze_file_not_found(self):
        """Test analysis with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            self.analyzer.analyze_temporal_bpm("/nonexistent/file.mp3")

    @patch("services.analysis_service.src.temporal_analyzer.es.MonoLoader")
    def test_analyze_empty_audio(self, mock_loader):
        """Test analysis with empty audio."""
        # Mock audio loading with empty array
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = np.array([])
        mock_loader.return_value = mock_loader_instance

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="Loaded audio is empty"):
                self.analyzer.analyze_temporal_bpm(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_calculate_stability(self):
        """Test tempo stability calculation."""
        # Perfect stability
        constant_bpm = [128.0] * 10
        assert self.analyzer._calculate_stability(constant_bpm) == 1.0

        # Low stability (high variation)
        variable_bpm = [120.0, 140.0, 110.0, 150.0, 100.0]
        stability = self.analyzer._calculate_stability(variable_bpm)
        assert 0.0 <= stability < 0.5

        # Moderate stability
        slight_variation = [128.0, 127.0, 129.0, 128.5, 127.5]
        stability = self.analyzer._calculate_stability(slight_variation)
        assert 0.5 < stability < 1.0

    def test_detect_tempo_changes(self):
        """Test tempo change detection."""
        temporal_data = [
            {"start_time": 0.0, "bpm": 128.0, "confidence": 0.9},
            {"start_time": 10.0, "bpm": 128.5, "confidence": 0.9},
            {"start_time": 20.0, "bpm": 135.0, "confidence": 0.9},  # Change
            {"start_time": 30.0, "bpm": 140.0, "confidence": 0.9},  # Change
            {"start_time": 40.0, "bpm": 140.5, "confidence": 0.9},
        ]

        changes = self.analyzer._detect_tempo_changes(temporal_data, threshold=5.0)

        assert len(changes) == 2
        assert changes[0]["from_bpm"] == 128.5
        assert changes[0]["to_bpm"] == 135.0
        assert changes[1]["from_bpm"] == 135.0
        assert changes[1]["to_bpm"] == 140.0

    def test_initialization_custom_parameters(self):
        """Test TemporalAnalyzer initialization with custom parameters."""
        analyzer = TemporalAnalyzer(
            window_size=20.0,
            hop_size=10.0,
            sample_rate=48000,
            start_duration=20.0,
            end_duration=20.0,
        )

        assert analyzer.window_size == 20.0
        assert analyzer.hop_size == 10.0
        assert analyzer.sample_rate == 48000
        assert analyzer.start_duration == 20.0
        assert analyzer.end_duration == 20.0

    def test_initialization_default_hop_size(self):
        """Test that default hop size is half of window size."""
        analyzer = TemporalAnalyzer(window_size=20.0)
        assert analyzer.hop_size == 10.0


class TestTemporalAnalysisEdgeCases:
    """Test suite for edge cases in temporal analysis."""

    @patch("services.analysis_service.src.temporal_analyzer.es.MonoLoader")
    def test_short_audio_file(self, mock_loader):
        """Test analysis of very short audio file."""
        analyzer = TemporalAnalyzer(window_size=10.0)

        # Mock 3 seconds of audio (too short for most windows)
        mock_audio = np.random.default_rng().standard_normal(44100 * 3).astype(np.float32)
        mock_loader_instance = MagicMock()
        mock_loader_instance.return_value = mock_audio
        mock_loader.return_value = mock_loader_instance

        with patch.object(
            analyzer.rhythm_extractor,
            "__call__",
            return_value=(120.0, np.array([]), 0.8, np.array([]), np.array([])),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = analyzer.analyze_temporal_bpm(tmp_path)
            finally:
                Path(tmp_path).unlink()

            # Should handle short files gracefully
            assert result is not None
            assert "average_bpm" in result
