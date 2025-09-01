"""Integration tests for the complete analysis pipeline."""

import concurrent.futures
import contextlib
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from services.analysis_service.src.audio_cache import AudioCache
from services.analysis_service.src.bpm_detector import BPMDetector
from services.analysis_service.src.key_detector import KeyDetector
from services.analysis_service.src.message_consumer import MessageConsumer
from services.analysis_service.src.model_manager import ModelManager
from services.analysis_service.src.mood_analyzer import MoodAnalyzer
from services.analysis_service.src.storage_handler import StorageHandler


class TestFullPipeline:
    """Integration tests for the complete analysis pipeline."""

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a simple sine wave audio file
            sample_rate = 44100
            duration = 5  # seconds
            frequency = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t)

            # Mock writing WAV file (simplified)
            f.write(audio.astype(np.float32).tobytes())
            temp_path = f.name

        yield temp_path

        # Cleanup
        with contextlib.suppress(Exception):
            Path(temp_path).unlink()

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage handler."""
        storage = Mock(spec=StorageHandler)
        storage.store_bpm_data.return_value = True
        storage.store_key_data.return_value = True
        storage.store_mood_data.return_value = True
        return storage

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache."""
        cache = Mock(spec=AudioCache)
        cache.get_bpm_results.return_value = None
        cache.get_key_results.return_value = None
        cache.get_mood_results.return_value = None
        cache.get_temporal_results.return_value = None
        cache.set_bpm_results.return_value = True
        cache.set_key_results.return_value = True
        cache.set_mood_results.return_value = True
        cache.set_temporal_results.return_value = True
        return cache

    @pytest.fixture
    def message_consumer(self, mock_storage, mock_cache):
        """Create a message consumer with mocked dependencies."""
        with patch("services.analysis_service.src.message_consumer.AudioCache") as mock_cache_class:
            mock_cache_class.return_value = mock_cache

            consumer = MessageConsumer(
                rabbitmq_url="amqp://localhost",
                enable_cache=True,
                enable_temporal_analysis=True,
                enable_key_detection=True,
                enable_mood_analysis=True,
            )
            consumer.storage = mock_storage
            consumer.cache = mock_cache

            # Mock the analyzers to avoid loading real models
            consumer.bpm_detector = Mock(spec=BPMDetector)
            consumer.key_detector = Mock(spec=KeyDetector)
            consumer.mood_analyzer = Mock(spec=MoodAnalyzer)

            return consumer

    def test_process_audio_file_complete_pipeline(self, message_consumer, temp_audio_file):
        """Test processing an audio file through the complete pipeline."""
        recording_id = str(uuid4())

        # Mock analyzer responses
        message_consumer.bpm_detector.detect_bpm.return_value = {
            "bpm": 120.0,
            "confidence": 0.95,
            "algorithm": "multifeature",
        }

        key_result = Mock()
        key_result.key = "C"
        key_result.scale = "major"
        key_result.confidence = 0.85
        key_result.agreement = True
        key_result.needs_review = False
        key_result.alternative_key = None
        key_result.alternative_scale = None
        message_consumer.key_detector.detect_key.return_value = key_result

        mood_result = Mock()
        mood_result.mood_scores = {"happy": 0.8, "energetic": 0.7}
        mood_result.primary_genre = "Pop"
        mood_result.genre_confidence = 0.9
        mood_result.genres = [{"genre": "Pop", "confidence": 0.9}]
        mood_result.danceability = 0.75
        mood_result.energy = 0.8
        mood_result.valence = 0.7
        mood_result.arousal = 0.6
        mood_result.voice_instrumental = "voice"
        mood_result.overall_confidence = 0.85
        mood_result.needs_review = False
        message_consumer.mood_analyzer.analyze_mood.return_value = mood_result

        # Process the audio file
        results = message_consumer.process_audio_file(temp_audio_file, recording_id)

        # Verify all analysis was performed
        assert "bpm_data" in results
        assert results["bpm_data"]["bpm"] == 120.0

        assert "key_data" in results
        assert results["key_data"]["key"] == "C"
        assert results["key_data"]["scale"] == "major"

        assert "mood_data" in results
        assert results["mood_data"]["primary_genre"] == "Pop"
        assert results["mood_data"]["danceability"] == 0.75

        # Verify storage was called
        assert message_consumer.storage.store_bpm_data.called
        assert message_consumer.storage.store_key_data.called
        assert message_consumer.storage.store_mood_data.called

    def test_process_with_cache_hit(self, message_consumer, temp_audio_file):
        """Test processing when results are cached."""
        recording_id = str(uuid4())

        # Set up cache to return cached results
        message_consumer.cache.get_bpm_results.return_value = {
            "bpm": 128.0,
            "confidence": 0.92,
            "cached_at": "2024-01-01T00:00:00Z",
        }
        message_consumer.cache.get_key_results.return_value = {
            "key": "G",
            "scale": "minor",
            "confidence": 0.88,
        }
        message_consumer.cache.get_mood_results.return_value = {
            "primary_genre": "Electronic",
            "danceability": 0.85,
        }

        # Process the audio file
        results = message_consumer.process_audio_file(temp_audio_file, recording_id)

        # Verify cached results were used
        assert results["from_cache"] is True
        assert results["bpm_data"]["bpm"] == 128.0
        assert results["key_data"]["key"] == "G"
        assert results["mood_data"]["primary_genre"] == "Electronic"

        # Verify analyzers were not called
        assert not message_consumer.bpm_detector.detect_bpm.called
        assert not message_consumer.key_detector.detect_key.called
        assert not message_consumer.mood_analyzer.analyze_mood.called

    def test_process_with_partial_cache(self, message_consumer, temp_audio_file):
        """Test processing when only some results are cached."""
        recording_id = str(uuid4())

        # Set up cache to return only BPM results
        message_consumer.cache.get_bpm_results.return_value = {
            "bpm": 130.0,
            "confidence": 0.9,
        }
        message_consumer.cache.get_key_results.return_value = None
        message_consumer.cache.get_mood_results.return_value = None

        # Mock analyzer responses for non-cached results
        key_result = Mock()
        key_result.key = "D"
        key_result.scale = "major"
        key_result.confidence = 0.82
        key_result.agreement = True
        key_result.needs_review = False
        key_result.alternative_key = None
        key_result.alternative_scale = None
        message_consumer.key_detector.detect_key.return_value = key_result

        mood_result = Mock()
        mood_result.mood_scores = {"relaxed": 0.8}
        mood_result.primary_genre = "Ambient"
        mood_result.genre_confidence = 0.75
        mood_result.genres = []
        mood_result.danceability = 0.3
        mood_result.energy = 0.4
        mood_result.valence = 0.6
        mood_result.arousal = 0.3
        mood_result.voice_instrumental = "instrumental"
        mood_result.overall_confidence = 0.7
        mood_result.needs_review = False
        message_consumer.mood_analyzer.analyze_mood.return_value = mood_result

        # Process the audio file
        results = message_consumer.process_audio_file(temp_audio_file, recording_id)

        # Verify mixed results
        assert results["from_cache"] is True  # BPM was cached
        assert results["bpm_data"]["bpm"] == 130.0
        assert results["key_data"]["key"] == "D"  # From analyzer
        assert results["mood_data"]["primary_genre"] == "Ambient"  # From analyzer

        # Verify only non-cached analyzers were called
        assert not message_consumer.bpm_detector.detect_bpm.called
        assert message_consumer.key_detector.detect_key.called
        assert message_consumer.mood_analyzer.analyze_mood.called

    def test_process_with_analyzer_failure(self, message_consumer, temp_audio_file):
        """Test handling of analyzer failures."""
        recording_id = str(uuid4())

        # Mock analyzer failures
        message_consumer.bpm_detector.detect_bpm.side_effect = Exception("BPM detection failed")
        message_consumer.key_detector.detect_key.side_effect = Exception("Key detection failed")
        message_consumer.mood_analyzer.analyze_mood.return_value = None  # Mood analysis returns None on failure

        # Process the audio file
        results = message_consumer.process_audio_file(temp_audio_file, recording_id)

        # Verify errors are captured
        assert "error" in results["bpm_data"]
        assert "BPM detection failed" in results["bpm_data"]["error"]

        assert "error" in results["key_data"]
        assert "Key detection failed" in results["key_data"]["error"]

        assert "error" in results["mood_data"]
        assert "Mood analysis failed" in results["mood_data"]["error"]

        # Verify storage is not called for failed analyses
        assert not message_consumer.storage.store_bpm_data.called
        assert not message_consumer.storage.store_key_data.called
        assert not message_consumer.storage.store_mood_data.called

    def test_performance_measurement(self, message_consumer, temp_audio_file):
        """Test performance of the analysis pipeline."""
        recording_id = str(uuid4())

        # Mock fast responses
        message_consumer.bpm_detector.detect_bpm.return_value = {
            "bpm": 120,
            "confidence": 0.9,
        }

        key_result = Mock()
        key_result.key = "A"
        key_result.scale = "minor"
        key_result.confidence = 0.8
        key_result.agreement = True
        key_result.needs_review = False
        key_result.alternative_key = None
        key_result.alternative_scale = None
        message_consumer.key_detector.detect_key.return_value = key_result

        mood_result = Mock()
        mood_result.mood_scores = {}
        mood_result.primary_genre = "Rock"
        mood_result.genre_confidence = 0.85
        mood_result.genres = []
        mood_result.danceability = 0.6
        mood_result.energy = 0.7
        mood_result.valence = 0.5
        mood_result.arousal = 0.6
        mood_result.voice_instrumental = "voice"
        mood_result.overall_confidence = 0.8
        mood_result.needs_review = False
        message_consumer.mood_analyzer.analyze_mood.return_value = mood_result

        # Measure processing time
        start_time = time.time()
        results = message_consumer.process_audio_file(temp_audio_file, recording_id)
        processing_time = time.time() - start_time

        # Verify processing completed
        assert results["recording_id"] == recording_id
        assert "bpm_data" in results
        assert "key_data" in results
        assert "mood_data" in results

        # Performance should be fast with mocked analyzers
        assert processing_time < 1.0  # Should complete in under 1 second with mocks


class TestPerformanceOptimization:
    """Tests for performance optimization."""

    def test_model_loading_performance(self):
        """Test model loading performance."""
        with patch("services.analysis_service.src.model_manager.urlopen") as mock_urlopen:
            # Mock successful model download
            mock_response = Mock()
            mock_response.read = Mock(return_value=b"fake_model_data")
            mock_response.info = Mock(return_value={"Content-Length": "1000"})
            mock_urlopen.return_value = mock_response

            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a mock model file to avoid actual download
                model_path = Path(temp_dir) / "test_model.pb"
                model_path.write_bytes(b"fake_model_data")

                manager = ModelManager(models_dir=temp_dir, lazy_load=True)

                # Measure loading time (for lazy loading this should be fast)
                start_time = time.time()

                # Check if model exists using get_model_path
                _ = manager.get_model_path("test_model")

                loading_time = time.time() - start_time

                # Verify model checking was fast
                assert loading_time < 0.5

                # The path checking should be very fast for lazy loading
                # We're testing the performance, not the actual functionality

    def test_cache_performance(self):
        """Test cache operation performance."""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis_class.return_value = mock_redis

            cache = AudioCache(enabled=True)

            # Test cache write performance
            start_time = time.time()
            for _ in range(100):
                cache.set_bpm_results("/fake/path.mp3", {"bpm": 120, "confidence": 0.9})
            write_time = time.time() - start_time

            # Test cache read performance
            mock_redis.get.return_value = '{"bpm": 120, "confidence": 0.9}'
            start_time = time.time()
            for _ in range(100):
                _ = cache.get_bpm_results("/fake/path.mp3")
            read_time = time.time() - start_time

            # Cache operations should be fast
            assert write_time < 1.0  # 100 writes in under 1 second
            assert read_time < 1.0  # 100 reads in under 1 second

    def test_parallel_analysis_simulation(self):
        """Test simulated parallel analysis of multiple features."""
        # This tests the concept of parallel analysis

        def simulate_bpm_detection():
            time.sleep(0.1)  # Simulate processing
            return {"bpm": 120, "confidence": 0.9}

        def simulate_key_detection():
            time.sleep(0.1)  # Simulate processing
            return {"key": "C", "scale": "major", "confidence": 0.85}

        def simulate_mood_analysis():
            time.sleep(0.1)  # Simulate processing
            return {"genre": "Pop", "danceability": 0.75}

        # Sequential execution
        start_time = time.time()
        _ = simulate_bpm_detection()
        _ = simulate_key_detection()
        _ = simulate_mood_analysis()
        sequential_time = time.time() - start_time

        # Parallel execution
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_bpm = executor.submit(simulate_bpm_detection)
            future_key = executor.submit(simulate_key_detection)
            future_mood = executor.submit(simulate_mood_analysis)

            bpm_parallel = future_bpm.result()
            key_parallel = future_key.result()
            mood_parallel = future_mood.result()
        parallel_time = time.time() - start_time

        # Verify results are the same
        assert bpm_parallel is not None
        assert key_parallel is not None
        assert mood_parallel is not None

        # Parallel should be faster than sequential
        assert parallel_time < sequential_time
        # With 0.1s sleep each, sequential should be ~0.3s, parallel ~0.1s
        assert sequential_time > 0.25  # Allow some margin
        assert parallel_time < 0.2  # Should complete in parallel
