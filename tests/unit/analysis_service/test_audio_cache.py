"""
Unit tests for Redis caching layer.

Tests cache operations, TTL management, and error handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import redis

from services.analysis_service.src.audio_cache import AudioCache


class TestAudioCache:
    """Test suite for AudioCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create cache with mocked Redis
        with patch("services.analysis_service.src.audio_cache.redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            self.cache = AudioCache(redis_host="localhost", redis_port=6379, algorithm_version="1.0")
            self.cache.redis_client = mock_client

    def test_initialization_success(self):
        """Test successful cache initialization."""
        with patch("services.analysis_service.src.audio_cache.redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            cache = AudioCache(redis_host="localhost", redis_port=6379, redis_password="secret")

            assert cache.redis_client is not None
            mock_client.ping.assert_called_once()

    def test_initialization_connection_error(self):
        """Test cache initialization with connection error."""
        with patch("services.analysis_service.src.audio_cache.redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.side_effect = redis.ConnectionError("Connection failed")
            mock_redis.return_value = mock_client

            cache = AudioCache(redis_host="localhost")

            assert cache.redis_client is None

    @patch("builtins.open", create=True)
    @patch("services.analysis_service.src.audio_cache.hashlib.sha256")
    def test_generate_file_hash_sha256(self, mock_sha256, mock_open):
        """Test file hashing with SHA256."""
        # Setup mock file reading
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.side_effect = [b"audio_data", b""]
        mock_open.return_value = mock_file

        # Setup mock hasher
        mock_hasher = MagicMock()
        mock_hasher.hexdigest.return_value = "abc123"
        mock_sha256.return_value = mock_hasher

        self.cache.use_xxh128 = False
        result = self.cache._generate_file_hash("/path/to/audio.mp3")

        assert result == "abc123"
        mock_hasher.update.assert_called_with(b"audio_data")

    def test_build_cache_key(self):
        """Test cache key generation."""
        key = self.cache._build_cache_key("bpm", "file_hash_123")
        assert key == "bpm:file_hash_123:1.0"

    @patch.object(AudioCache, "_generate_file_hash")
    def test_get_bpm_results_cache_hit(self, mock_hash):
        """Test retrieving cached BPM results."""
        mock_hash.return_value = "file_hash_123"

        cached_data = {"bpm": 128.0, "confidence": 0.95, "algorithm_version": "1.0"}
        self.cache.redis_client.get.return_value = json.dumps(cached_data)

        result = self.cache.get_bpm_results("/path/to/audio.mp3")

        assert result == cached_data
        self.cache.redis_client.get.assert_called_once_with("bpm:file_hash_123:1.0")

    @patch.object(AudioCache, "_generate_file_hash")
    def test_get_bpm_results_cache_miss(self, mock_hash):
        """Test cache miss for BPM results."""
        mock_hash.return_value = "file_hash_123"
        self.cache.redis_client.get.return_value = None

        result = self.cache.get_bpm_results("/path/to/audio.mp3")

        assert result is None

    @patch.object(AudioCache, "_generate_file_hash")
    def test_set_bpm_results_success(self, mock_hash):
        """Test caching BPM results."""
        mock_hash.return_value = "file_hash_123"
        self.cache.redis_client.setex.return_value = True

        results = {"bpm": 128.0, "confidence": 0.95}

        success = self.cache.set_bpm_results("/path/to/audio.mp3", results, confidence=0.95)

        assert success is True

        # Verify setex was called with correct TTL (30 days)
        self.cache.redis_client.setex.assert_called_once()
        call_args = self.cache.redis_client.setex.call_args
        assert call_args[0][0] == "bpm:file_hash_123:1.0"
        assert call_args[0][1] == 30 * 24 * 60 * 60  # DEFAULT_TTL

    @patch.object(AudioCache, "_generate_file_hash")
    def test_set_bpm_results_low_confidence(self, mock_hash):
        """Test caching with low confidence (shorter TTL)."""
        mock_hash.return_value = "file_hash_123"
        self.cache.redis_client.setex.return_value = True

        results = {"bpm": 60.0, "confidence": 0.3}

        success = self.cache.set_bpm_results("/path/to/audio.mp3", results, confidence=0.3)

        assert success is True

        # Verify shorter TTL for low confidence (7 days)
        call_args = self.cache.redis_client.setex.call_args
        assert call_args[0][1] == 7 * 24 * 60 * 60  # LOW_CONFIDENCE_TTL

    @patch.object(AudioCache, "_generate_file_hash")
    def test_set_bpm_results_failed(self, mock_hash):
        """Test caching failed analysis (short TTL)."""
        mock_hash.return_value = "file_hash_123"
        self.cache.redis_client.setex.return_value = True

        results = {"error": "Analysis failed"}

        success = self.cache.set_bpm_results("/path/to/audio.mp3", results, failed=True)

        assert success is True

        # Verify short TTL for failed analysis (1 hour)
        call_args = self.cache.redis_client.setex.call_args
        assert call_args[0][1] == 60 * 60  # FAILED_TTL

    @patch.object(AudioCache, "_generate_file_hash")
    def test_get_temporal_results(self, mock_hash):
        """Test retrieving cached temporal results."""
        mock_hash.return_value = "file_hash_123"

        cached_data = {
            "average_bpm": 128.0,
            "stability_score": 0.95,
            "temporal_bpm": [],
        }
        self.cache.redis_client.get.return_value = json.dumps(cached_data)

        result = self.cache.get_temporal_results("/path/to/audio.mp3")

        assert result == cached_data
        self.cache.redis_client.get.assert_called_once_with("temporal:file_hash_123:1.0")

    @patch.object(AudioCache, "_generate_file_hash")
    def test_set_temporal_results(self, mock_hash):
        """Test caching temporal analysis results."""
        mock_hash.return_value = "file_hash_123"
        self.cache.redis_client.setex.return_value = True

        results = {"average_bpm": 128.0, "stability_score": 0.95, "temporal_bpm": []}

        success = self.cache.set_temporal_results("/path/to/audio.mp3", results, stability_score=0.95)

        assert success is True
        self.cache.redis_client.setex.assert_called_once()

    @patch.object(AudioCache, "_generate_file_hash")
    def test_invalidate_cache(self, mock_hash):
        """Test cache invalidation for a file."""
        mock_hash.return_value = "file_hash_123"
        self.cache.redis_client.delete.return_value = 1

        success = self.cache.invalidate_cache("/path/to/audio.mp3")

        assert success is True
        # Should delete both BPM and temporal cache entries
        assert self.cache.redis_client.delete.call_count == 2

    def test_flush_version_cache(self):
        """Test flushing cache for specific version."""
        self.cache.redis_client.keys.return_value = [
            "bpm:hash1:1.0",
            "bpm:hash2:1.0",
            "temporal:hash3:1.0",
        ]
        self.cache.redis_client.delete.return_value = 3

        deleted = self.cache.flush_version_cache("1.0")

        assert deleted == 3
        self.cache.redis_client.keys.assert_called_once_with("*:1.0")

    @patch.object(AudioCache, "get_bpm_results")
    @patch.object(AudioCache, "set_bpm_results")
    def test_warm_cache(self, mock_set, mock_get):
        """Test cache warming functionality."""
        # Setup: first file not cached, second already cached
        mock_get.side_effect = [None, {"bpm": 120.0}]
        mock_set.return_value = True

        # Mock analyzer callback
        analyzer = MagicMock()
        analyzer.return_value = {"bpm": 128.0, "confidence": 0.9}

        file_paths = ["/audio1.mp3", "/audio2.mp3"]
        cached = self.cache.warm_cache(file_paths, analyzer)

        assert cached == 1  # Only first file was cached
        analyzer.assert_called_once_with("/audio1.mp3")

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        self.cache.redis_client.info.side_effect = [
            {"used_memory_human": "10M"},  # memory info
            {"db0": {"keys": 100}},  # keyspace info
        ]
        self.cache.redis_client.keys.side_effect = [
            ["bpm:1", "bpm:2"],  # BPM keys
            ["temporal:1"],  # Temporal keys
        ]

        stats = self.cache.get_cache_stats()

        assert stats["connected"] is True
        assert stats["memory_used"] == "10M"
        assert stats["total_keys"] == 100
        assert stats["bpm_cached"] == 2
        assert stats["temporal_cached"] == 1
        assert stats["algorithm_version"] == "1.0"

    def test_no_redis_connection(self):
        """Test operations without Redis connection."""
        self.cache.redis_client = None

        assert self.cache.get_bpm_results("/audio.mp3") is None
        assert self.cache.set_bpm_results("/audio.mp3", {}) is False
        assert self.cache.invalidate_cache("/audio.mp3") is False
        assert self.cache.flush_version_cache() == 0

        stats = self.cache.get_cache_stats()
        assert stats["connected"] is False


class TestAudioCacheIntegration:
    """Integration tests for AudioCache with real Redis operations."""

    @pytest.mark.skip(reason="Requires Redis server")
    def test_real_redis_operations(self):
        """Test with actual Redis connection (requires Redis server)."""
        cache = AudioCache(redis_host="localhost", redis_port=6379)

        if cache.redis_client:
            # Test real operations
            results = {"bpm": 128.0, "confidence": 0.95}

            # Set and get
            cache.set_bpm_results("/test.mp3", results, confidence=0.95)
            cached = cache.get_bpm_results("/test.mp3")

            assert cached is not None
            assert cached["bpm"] == 128.0

            # Cleanup
            cache.invalidate_cache("/test.mp3")
