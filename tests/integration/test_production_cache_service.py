"""
Integration tests for ProductionCacheService Redis operations.

Tests Redis caching operations, error handling, fallback behavior,
and performance characteristics with real Redis instance.
"""

import concurrent.futures
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta

import pytest
import redis

from services.shared.production_cache_service import ProductionCacheService, generate_cache_key, hash_key

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_TEST_DB", "1"))  # Use different DB for tests


@pytest.fixture(scope="module")
def redis_client():
    """Create Redis client for direct testing."""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # Test connection
    try:
        client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT} DB:{REDIS_DB}")
    except redis.ConnectionError:
        pytest.skip("Redis server not available")

    yield client

    # Cleanup test data
    client.flushdb()
    client.close()


@pytest.fixture
def cache_service(redis_client):
    """Create ProductionCacheService instance."""
    service = ProductionCacheService(
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
        service_prefix="test_cache",
        enabled=True,
        default_ttl=3600,
    )

    yield service

    # Cleanup
    service.flush_service_cache()
    service.close()


@pytest.fixture
def disabled_cache_service():
    """Create disabled cache service for fallback testing."""
    return ProductionCacheService(enabled=False)


class TestProductionCacheServiceBasicOperations:
    """Test basic cache operations."""

    def test_cache_service_initialization(self, cache_service: ProductionCacheService):
        """Test cache service initialization."""
        assert cache_service.enabled is True
        assert cache_service.service_prefix == "test_cache"
        assert cache_service.default_ttl == 3600
        assert cache_service.redis_client is not None

        # Test stats initialization
        stats = cache_service.get_stats()
        assert stats["enabled"] is True
        assert stats["connected"] is True
        assert stats["service_prefix"] == "test_cache"
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["deletes"] == 0
        assert stats["errors"] == 0

    def test_disabled_cache_service(self, disabled_cache_service: ProductionCacheService):
        """Test disabled cache service behavior."""
        assert disabled_cache_service.enabled is False
        assert disabled_cache_service.redis_client is None

        # All operations should return None/False but not error
        assert disabled_cache_service.get("test_key") is None
        assert disabled_cache_service.set("test_key", "test_value") is False
        assert disabled_cache_service.delete("test_key") is False
        assert disabled_cache_service.exists("test_key") is False

    def test_set_and_get_string_value(self, cache_service: ProductionCacheService):
        """Test basic string value caching."""
        key = "test_string"
        value = "Hello, Redis!"

        # Set value
        success = cache_service.set(key, value)
        assert success is True

        # Get value
        retrieved = cache_service.get(key)
        assert retrieved == value

        # Verify stats
        stats = cache_service.get_stats()
        assert stats["sets"] >= 1
        assert stats["hits"] >= 1

    def test_set_and_get_json_value(self, cache_service: ProductionCacheService):
        """Test JSON object caching."""
        key = "test_json"
        value = {
            "id": 123,
            "name": "Test Object",
            "metadata": {"created_at": "2024-01-01T00:00:00Z", "tags": ["test", "integration", "cache"]},
            "metrics": {"bpm": 128.5, "key": "C major", "energy": 0.85},
        }

        # Set value
        success = cache_service.set(key, value)
        assert success is True

        # Get value
        retrieved = cache_service.get(key)
        assert retrieved == value
        assert isinstance(retrieved, dict)
        assert retrieved["id"] == 123
        assert retrieved["metadata"]["tags"] == ["test", "integration", "cache"]
        assert retrieved["metrics"]["bpm"] == 128.5

    def test_set_and_get_list_value(self, cache_service: ProductionCacheService):
        """Test list caching."""
        key = "test_list"
        value = [
            {"track": 1, "title": "Track One"},
            {"track": 2, "title": "Track Two"},
            {"track": 3, "title": "Track Three"},
        ]

        success = cache_service.set(key, value)
        assert success is True

        retrieved = cache_service.get(key)
        assert retrieved == value
        assert len(retrieved) == 3
        assert retrieved[0]["title"] == "Track One"

    def test_cache_miss(self, cache_service: ProductionCacheService):
        """Test cache miss behavior."""
        initial_stats = cache_service.get_stats()
        initial_misses = initial_stats["misses"]

        # Get non-existent key
        result = cache_service.get("non_existent_key")
        assert result is None

        # Verify miss was recorded
        stats = cache_service.get_stats()
        assert stats["misses"] > initial_misses

    def test_cache_overwrite(self, cache_service: ProductionCacheService):
        """Test overwriting existing cache values."""
        key = "overwrite_test"
        original_value = {"version": 1, "data": "original"}
        updated_value = {"version": 2, "data": "updated"}

        # Set original value
        cache_service.set(key, original_value)
        assert cache_service.get(key) == original_value

        # Overwrite with new value
        cache_service.set(key, updated_value)
        retrieved = cache_service.get(key)
        assert retrieved == updated_value
        assert retrieved["version"] == 2
        assert retrieved["data"] == "updated"


class TestProductionCacheServiceTTLOperations:
    """Test TTL (Time To Live) operations."""

    def test_set_with_custom_ttl(self, cache_service: ProductionCacheService):
        """Test setting cache value with custom TTL."""
        key = "ttl_test"
        value = "expires_soon"
        ttl = 2  # 2 seconds

        success = cache_service.set(key, value, ttl=ttl)
        assert success is True

        # Value should be available immediately
        assert cache_service.get(key) == value
        assert cache_service.exists(key) is True

        # Wait for expiration
        time.sleep(ttl + 0.5)

        # Value should be expired
        assert cache_service.get(key) is None
        assert cache_service.exists(key) is False

    def test_expire_existing_key(self, cache_service: ProductionCacheService):
        """Test setting expiration on existing key."""
        key = "expire_test"
        value = "will_expire"

        # Set value without TTL
        cache_service.set(key, value)
        assert cache_service.get(key) == value

        # Set expiration
        success = cache_service.expire(key, 1)  # 1 second
        assert success is True

        # Value should still be available
        assert cache_service.get(key) == value

        # Wait for expiration
        time.sleep(1.5)

        # Value should be expired
        assert cache_service.get(key) is None

    def test_set_with_nx_flag(self, cache_service: ProductionCacheService):
        """Test setting value only if key doesn't exist."""
        key = "nx_test"
        original_value = "original"
        new_value = "new"

        # Set original value
        success = cache_service.set(key, original_value)
        assert success is True

        # Try to set with nx=True (should fail)
        success = cache_service.set(key, new_value, nx=True)
        assert success is False

        # Original value should remain
        assert cache_service.get(key) == original_value

        # Delete key and try again
        cache_service.delete(key)

        # Now nx=True should succeed
        success = cache_service.set(key, new_value, nx=True)
        assert success is True
        assert cache_service.get(key) == new_value


class TestProductionCacheServiceAdvancedOperations:
    """Test advanced cache operations."""

    def test_delete_operation(self, cache_service: ProductionCacheService):
        """Test cache key deletion."""
        key = "delete_test"
        value = "to_be_deleted"

        # Set and verify
        cache_service.set(key, value)
        assert cache_service.get(key) == value

        # Delete
        success = cache_service.delete(key)
        assert success is True

        # Verify deletion
        assert cache_service.get(key) is None
        assert cache_service.exists(key) is False

        # Delete non-existent key
        success = cache_service.delete("non_existent")
        assert success is False

    def test_exists_operation(self, cache_service: ProductionCacheService):
        """Test key existence checking."""
        key = "exists_test"
        value = "exists"

        # Initially should not exist
        assert cache_service.exists(key) is False

        # Set value
        cache_service.set(key, value)
        assert cache_service.exists(key) is True

        # Delete and check again
        cache_service.delete(key)
        assert cache_service.exists(key) is False

    def test_increment_operation(self, cache_service: ProductionCacheService):
        """Test numeric value increment."""
        key = "counter"

        # Increment non-existent key (should create with value 1)
        result = cache_service.increment(key)
        assert result == 1

        # Increment existing key
        result = cache_service.increment(key, amount=5)
        assert result == 6

        # Increment with negative amount (decrement)
        result = cache_service.increment(key, amount=-2)
        assert result == 4

        # Test with TTL
        counter_key = "counter_with_ttl"
        result = cache_service.increment(counter_key, amount=10, ttl=2)
        assert result == 10
        assert cache_service.exists(counter_key) is True

        # Wait for expiration
        time.sleep(2.5)
        assert cache_service.exists(counter_key) is False

    def test_hash_operations(self, cache_service: ProductionCacheService):
        """Test Redis hash operations."""
        hash_key = "user_profile"

        # Set hash fields
        assert cache_service.hash_set(hash_key, "name", "John Doe") is True
        assert cache_service.hash_set(hash_key, "age", 30) is True
        assert cache_service.hash_set(hash_key, "email", "john@example.com") is True

        # Get individual fields
        assert cache_service.hash_get(hash_key, "name") == "John Doe"
        assert cache_service.hash_get(hash_key, "age") == 30
        assert cache_service.hash_get(hash_key, "email") == "john@example.com"

        # Get non-existent field
        assert cache_service.hash_get(hash_key, "nonexistent") is None

        # Get all fields
        all_fields = cache_service.hash_get_all(hash_key)
        expected = {"name": "John Doe", "age": 30, "email": "john@example.com"}
        assert all_fields == expected

        # Test hash with complex data
        profile_data = {"preferences": {"theme": "dark", "notifications": True}, "last_login": "2024-01-01T12:00:00Z"}
        cache_service.hash_set(hash_key, "profile_data", profile_data)
        retrieved_profile = cache_service.hash_get(hash_key, "profile_data")
        assert retrieved_profile == profile_data

    def test_clear_pattern(self, cache_service: ProductionCacheService):
        """Test pattern-based cache clearing."""
        # Set up test data with different patterns
        cache_service.set("user:123:profile", {"name": "User 123"})
        cache_service.set("user:124:profile", {"name": "User 124"})
        cache_service.set("user:125:settings", {"theme": "dark"})
        cache_service.set("session:abc123", {"user_id": 123})
        cache_service.set("session:def456", {"user_id": 124})

        # Clear all user profiles
        deleted_count = cache_service.clear_pattern("user:*:profile")
        assert deleted_count == 2

        # Verify profiles are gone but other keys remain
        assert cache_service.get("user:123:profile") is None
        assert cache_service.get("user:124:profile") is None
        assert cache_service.get("user:125:settings") is not None
        assert cache_service.get("session:abc123") is not None

        # Clear all session data
        deleted_count = cache_service.clear_pattern("session:*")
        assert deleted_count == 2

        # Verify sessions are gone
        assert cache_service.get("session:abc123") is None
        assert cache_service.get("session:def456") is None
        assert cache_service.get("user:125:settings") is not None

    def test_flush_service_cache(self, cache_service: ProductionCacheService):
        """Test flushing entire service cache."""
        # Set multiple values
        test_data = {"key1": "value1", "key2": {"nested": "object"}, "key3": [1, 2, 3]}

        for key, value in test_data.items():
            cache_service.set(key, value)

        # Verify all values exist
        for key, value in test_data.items():
            assert cache_service.get(key) == value

        # Flush all cache
        success = cache_service.flush_service_cache()
        assert success is True

        # Verify all values are gone
        for key in test_data:
            assert cache_service.get(key) is None


class TestProductionCacheServiceErrorHandling:
    """Test error handling and edge cases."""

    def test_connection_failure_handling(self):
        """Test handling of Redis connection failures."""
        # Create service with invalid Redis config
        invalid_service = ProductionCacheService(
            redis_host="invalid_host", redis_port=9999, service_prefix="test_invalid", enabled=True
        )

        # Should be disabled due to connection failure
        assert invalid_service.enabled is False
        assert invalid_service.redis_client is None

        # All operations should return None/False gracefully
        assert invalid_service.get("test") is None
        assert invalid_service.set("test", "value") is False
        assert invalid_service.delete("test") is False
        assert invalid_service.exists("test") is False

        invalid_service.close()

    def test_serialization_error_handling(self, cache_service: ProductionCacheService):
        """Test handling of serialization errors."""

        # Test with non-serializable object
        class NonSerializable:
            def __init__(self):
                self.circular_ref = self

        # This should fail gracefully
        success = cache_service.set("bad_object", NonSerializable())
        assert success is False

        # Stats should show error
        stats = cache_service.get_stats()
        assert stats["errors"] > 0

    def test_large_data_handling(self, cache_service: ProductionCacheService):
        """Test handling of large data objects."""
        # Create large data (1MB of data)
        large_data = {"data": "x" * (1024 * 1024), "metadata": {"size": "1MB", "type": "test"}}

        # Should handle large data gracefully
        success = cache_service.set("large_data", large_data, ttl=60)
        assert success is True

        # Should retrieve correctly
        retrieved = cache_service.get("large_data")
        assert retrieved is not None
        assert len(retrieved["data"]) == 1024 * 1024
        assert retrieved["metadata"]["size"] == "1MB"

    def test_unicode_handling(self, cache_service: ProductionCacheService):
        """Test handling of Unicode data."""
        unicode_data = {
            "chinese": "你好世界",
            "japanese": "こんにちは世界",
            "arabic": "مرحبا بالعالم",
            "emoji": "🚀🌟💫🎵🎶",
            "mixed": "Hello 世界 🌍",
        }

        success = cache_service.set("unicode_test", unicode_data)
        assert success is True

        retrieved = cache_service.get("unicode_test")
        assert retrieved == unicode_data
        assert retrieved["chinese"] == "你好世界"
        assert retrieved["emoji"] == "🚀🌟💫🎵🎶"


class TestProductionCacheServicePerformance:
    """Test performance characteristics."""

    def test_bulk_operations_performance(self, cache_service: ProductionCacheService):
        """Test performance of bulk cache operations."""
        num_operations = 1000
        test_data = {
            f"perf_key_{i}": {
                "id": i,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": f"test_data_{i}",
                "metrics": {"value": i * 1.5},
            }
            for i in range(num_operations)
        }

        # Time bulk set operations
        start_time = time.time()
        for key, value in test_data.items():
            cache_service.set(key, value)
        set_duration = time.time() - start_time

        # Time bulk get operations
        start_time = time.time()
        retrieved_count = 0
        for key in test_data:
            if cache_service.get(key) is not None:
                retrieved_count += 1
        get_duration = time.time() - start_time

        # Performance assertions
        assert retrieved_count == num_operations
        assert set_duration < 10.0  # Should complete in under 10 seconds
        assert get_duration < 5.0  # Retrieval should be faster

        # Calculate operations per second
        set_ops_per_sec = num_operations / set_duration
        get_ops_per_sec = num_operations / get_duration

        logger.info(f"Set operations: {set_ops_per_sec:.1f} ops/sec")
        logger.info(f"Get operations: {get_ops_per_sec:.1f} ops/sec")

        # Should achieve reasonable throughput
        assert set_ops_per_sec > 100  # At least 100 sets per second
        assert get_ops_per_sec > 200  # At least 200 gets per second

    def test_concurrent_operations(self, cache_service: ProductionCacheService):
        """Test concurrent cache operations."""

        num_threads = 10
        operations_per_thread = 100

        def worker_function(worker_id: int):
            """Worker function for concurrent operations."""
            success_count = 0
            for i in range(operations_per_thread):
                key = f"concurrent_{worker_id}_{i}"
                value = {"worker": worker_id, "operation": i}

                # Set value
                if cache_service.set(key, value):
                    success_count += 1

                # Get value back
                retrieved = cache_service.get(key)
                if retrieved and retrieved["worker"] == worker_id:
                    success_count += 1

            return success_count

        # Run concurrent operations
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, worker_id) for worker_id in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        duration = time.time() - start_time

        # Verify results
        total_operations = sum(results)
        expected_operations = num_threads * operations_per_thread * 2  # set + get

        assert total_operations == expected_operations

        # Calculate throughput
        total_ops = num_threads * operations_per_thread * 2
        ops_per_sec = total_ops / duration

        logger.info(f"Concurrent operations: {ops_per_sec:.1f} ops/sec with {num_threads} threads")

        # Should handle concurrent access well
        assert ops_per_sec > 500  # At least 500 concurrent ops per second


class TestProductionCacheServiceHealthAndStats:
    """Test health checking and statistics."""

    def test_health_check(self, cache_service: ProductionCacheService):
        """Test cache service health checking."""
        health = cache_service.health_check()

        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert "redis_response_time_ms" in health
        assert health["message"] == "Cache service is healthy"
        assert isinstance(health["redis_response_time_ms"], int | float)
        assert health["redis_response_time_ms"] < 1000  # Should respond quickly

    def test_disabled_service_health_check(self, disabled_cache_service: ProductionCacheService):
        """Test health check for disabled service."""
        health = disabled_cache_service.health_check()

        assert health["status"] == "disabled"
        assert health["healthy"] is True
        assert health["message"] == "Cache service is disabled"

    def test_detailed_stats(self, cache_service: ProductionCacheService):
        """Test detailed cache statistics."""
        # Perform various operations to generate stats
        cache_service.set("stats_test_1", "value1")
        cache_service.set("stats_test_2", "value2")
        cache_service.get("stats_test_1")  # hit
        cache_service.get("nonexistent")  # miss
        cache_service.delete("stats_test_2")

        stats = cache_service.get_stats()

        # Basic stats
        assert stats["enabled"] is True
        assert stats["connected"] is True
        assert stats["service_prefix"] == "test_cache"

        # Operation counters
        assert stats["sets"] >= 2
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["deletes"] >= 1

        # Hit rate calculation
        assert "hit_rate" in stats
        assert isinstance(stats["hit_rate"], int | float)
        assert 0 <= stats["hit_rate"] <= 100

        # Redis info
        assert "redis_memory" in stats
        assert "redis_clients" in stats
        assert "redis_commands" in stats


class TestUtilityFunctions:
    """Test utility functions."""

    def test_generate_cache_key(self):
        """Test cache key generation utility."""
        # Test with string components
        key = generate_cache_key("user", "123", "profile")
        assert key == "user:123:profile"

        # Test with mixed types
        key = generate_cache_key("recording", 456, "analysis", "bpm")
        assert key == "recording:456:analysis:bpm"

        # Test with None values
        key = generate_cache_key("session", None, "data")
        assert key == "session:null:data"

        # Test with single component
        key = generate_cache_key("simple")
        assert key == "simple"

        # Test with no components
        key = generate_cache_key()
        assert key == ""

    def test_hash_key(self):
        """Test hash key generation utility."""
        # Test with string
        hash1 = hash_key("test string")
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # First 16 characters of SHA256

        # Test with bytes
        hash2 = hash_key(b"test bytes")
        assert isinstance(hash2, str)
        assert len(hash2) == 16

        # Test consistency
        hash3 = hash_key("test string")
        assert hash1 == hash3

        # Test different inputs produce different hashes
        hash4 = hash_key("different string")
        assert hash1 != hash4

        # Test with complex data (converted to string)
        complex_data = json.dumps({"key": "value", "list": [1, 2, 3]}, sort_keys=True)
        hash5 = hash_key(complex_data)
        assert isinstance(hash5, str)
        assert len(hash5) == 16


@pytest.mark.integration
@pytest.mark.requires_docker
class TestProductionCacheServiceIntegration:
    """Integration tests requiring Redis server."""

    def test_full_caching_workflow(self, cache_service: ProductionCacheService):
        """Test complete caching workflow."""
        # Simulate analysis result caching workflow
        recording_id = "12345678-1234-5678-9012-123456789abc"
        analysis_type = "bpm"

        # Generate cache key
        cache_key = generate_cache_key("analysis", recording_id, analysis_type)
        assert cache_key == f"analysis:{recording_id}:{analysis_type}"

        # Check if cached result exists (should be miss)
        cached_result = cache_service.get(cache_key)
        assert cached_result is None

        # Simulate analysis computation and caching
        analysis_result = {
            "bpm": 128.5,
            "confidence": 0.95,
            "analysis_time_ms": 2500,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {"algorithm": "beat_tracker_v2", "version": "1.2.0"},
        }

        # Cache the result with appropriate TTL
        ttl = ProductionCacheService.MEDIUM_TTL  # 7 days
        success = cache_service.set(cache_key, analysis_result, ttl=ttl)
        assert success is True

        # Retrieve cached result (should be hit)
        retrieved_result = cache_service.get(cache_key)
        assert retrieved_result == analysis_result
        assert retrieved_result["bpm"] == 128.5
        assert retrieved_result["confidence"] == 0.95

        # Verify cache statistics
        stats = cache_service.get_stats()
        assert stats["hits"] >= 1
        assert stats["sets"] >= 1

        # Test cache invalidation
        invalidated_count = cache_service.clear_pattern(f"analysis:{recording_id}:*")
        assert invalidated_count >= 1

        # Verify result is gone
        cached_result = cache_service.get(cache_key)
        assert cached_result is None

    def test_multi_format_cue_caching(self, cache_service: ProductionCacheService):
        """Test caching CUE files in multiple formats."""
        tracklist_id = "87654321-4321-8765-2109-876543210def"

        formats = ["standard", "cdj", "traktor", "serato", "rekordbox"]
        cue_data = {}

        # Cache CUE files for different formats
        for format_type in formats:
            cache_key = generate_cache_key("cue", tracklist_id, format_type)
            cue_content = {
                "format": format_type,
                "content": f"CUE file content for {format_type}",
                "generated_at": datetime.now(UTC).isoformat(),
                "tracklist_id": tracklist_id,
            }

            success = cache_service.set(cache_key, cue_content, ttl=ProductionCacheService.LONG_TTL)
            assert success is True
            cue_data[format_type] = cue_content

        # Retrieve all formats
        for format_type in formats:
            cache_key = generate_cache_key("cue", tracklist_id, format_type)
            retrieved = cache_service.get(cache_key)

            assert retrieved is not None
            assert retrieved["format"] == format_type
            assert retrieved["tracklist_id"] == tracklist_id
            assert retrieved == cue_data[format_type]

        # Clear all CUE files for this tracklist
        pattern = f"cue:{tracklist_id}:*"
        deleted_count = cache_service.clear_pattern(pattern)
        assert deleted_count == len(formats)

        # Verify all are gone
        for format_type in formats:
            cache_key = generate_cache_key("cue", tracklist_id, format_type)
            assert cache_service.get(cache_key) is None

    def test_session_cache_management(self, cache_service: ProductionCacheService):
        """Test session-based cache management."""
        session_id = "session_" + hash_key("test_session_123")
        user_id = "user_456"

        # Cache session data
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
            "permissions": ["read", "write", "delete"],
        }

        session_key = generate_cache_key("session", session_id)
        cache_service.set(session_key, session_data, ttl=24 * 3600)  # 24 hours

        # Cache user preferences
        user_prefs = {"theme": "dark", "language": "en", "notifications": {"email": True, "push": False, "sms": True}}

        prefs_key = generate_cache_key("user_prefs", user_id)
        cache_service.set(prefs_key, user_prefs, ttl=7 * 24 * 3600)  # 7 days

        # Retrieve and verify session
        retrieved_session = cache_service.get(session_key)
        assert retrieved_session == session_data
        assert retrieved_session["user_id"] == user_id

        # Retrieve and verify preferences
        retrieved_prefs = cache_service.get(prefs_key)
        assert retrieved_prefs == user_prefs
        assert retrieved_prefs["theme"] == "dark"

        # Simulate session cleanup
        cache_service.delete(session_key)
        assert cache_service.get(session_key) is None

        # User preferences should remain
        assert cache_service.get(prefs_key) is not None
