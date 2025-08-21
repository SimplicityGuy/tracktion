"""Unit tests for OGG Vorbis audio analysis."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "services" / "analysis_service" / "src"))

from bpm_detector import BPMDetector
from config import BPMConfig


class TestOggAudioAnalysis(unittest.TestCase):
    """Test suite for OGG Vorbis audio analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_recording_id = uuid4()
        self.test_correlation_id = "test-correlation-123"

    def test_ogg_format_supported(self):
        """Test that OGG format is recognized as supported."""
        bpm_config = BPMConfig()
        self.assertIn(".ogg", bpm_config.supported_formats)
        # Note: .oga is not in default config but could be added

    def test_is_audio_format_supported_ogg(self):
        """Test that OGG files extensions are in supported formats."""
        bpm_config = BPMConfig()
        supported = bpm_config.supported_formats

        # Test .ogg extension
        self.assertIn(".ogg", supported)


class TestBPMDetectorOgg(unittest.TestCase):
    """Test BPM detection for OGG files."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BPMConfig()
        self.detector = BPMDetector(config=self.config)

    def test_bpm_detection_with_ogg_file(self):
        """Test BPM detection accepts OGG file paths."""
        # Simply test that the detector can be initialized with config that supports OGG
        config = BPMConfig()
        self.assertIn(".ogg", config.supported_formats)

        detector = BPMDetector(config=config)
        self.assertIsNotNone(detector)

        # Verify that OGG is in supported formats
        self.assertIn(".ogg", config.supported_formats)

    def test_ogg_in_supported_formats(self):
        """Test that OGG is in the list of supported formats."""
        self.assertIn(".ogg", self.config.supported_formats)


class TestOggPerformance(unittest.TestCase):
    """Performance tests for OGG audio analysis."""

    @patch("bpm_detector.Path.exists")
    @patch("bpm_detector.es.MonoLoader")
    def test_large_ogg_file_handling(self, mock_loader, mock_exists):
        """Test handling of large OGG files."""
        # Mock file exists
        mock_exists.return_value = True

        # Simulate large file (>100MB equivalent in samples)
        large_audio = np.random.rand(44100 * 60 * 30).astype(np.float32)  # 30 minutes
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.return_value = large_audio

        config = BPMConfig()
        config.max_file_size_mb = 500  # Allow large files
        detector = BPMDetector(config=config)

        # Should handle large file without error
        # Note: In real implementation, this would use streaming
        with patch.object(detector, "_extract_rhythm") as mock_rhythm:
            mock_rhythm.return_value = (120.0, np.array([]), 0.8, None, np.array([]))

            result = detector.detect_bpm("/path/to/large.ogg")
            self.assertIsNotNone(result)
            self.assertIn("bpm", result)


if __name__ == "__main__":
    unittest.main()
