"""Unit tests for hash utility functions."""

from services.file_watcher.src.hash_utils import HashStrategy, HashType


class TestHashType:
    """Test HashType enumeration."""

    def test_hash_type_values(self):
        """Test that HashType has expected values."""
        assert HashType.SHA256.value == "sha256"
        assert HashType.XXH128.value == "xxh128"


class TestHashStrategy:
    """Test HashStrategy selection logic."""

    def test_for_duplicate_detection(self):
        """Test hash selection for duplicate detection."""
        assert HashStrategy.for_duplicate_detection() == HashType.XXH128

    def test_for_integrity_verification(self):
        """Test hash selection for integrity verification."""
        assert HashStrategy.for_integrity_verification() == HashType.SHA256

    def test_for_database_lookup(self):
        """Test hash selection for database lookups."""
        assert HashStrategy.for_database_lookup() == HashType.XXH128

    def test_for_api_response(self):
        """Test hash selection for API responses."""
        hash_types = HashStrategy.for_api_response()
        assert len(hash_types) == 2
        assert HashType.SHA256 in hash_types
        assert HashType.XXH128 in hash_types

    def test_select_hash_sha256(self):
        """Test selecting SHA256 hash."""
        sha256 = "abc123def456"
        xxh128 = "xyz789ghi012"

        result = HashStrategy.select_hash(sha256, xxh128, HashType.SHA256)
        assert result == sha256

    def test_select_hash_xxh128(self):
        """Test selecting XXH128 hash."""
        sha256 = "abc123def456"
        xxh128 = "xyz789ghi012"

        result = HashStrategy.select_hash(sha256, xxh128, HashType.XXH128)
        assert result == xxh128

    def test_select_hash_with_none_values(self):
        """Test selecting hash when some values are None."""
        result = HashStrategy.select_hash(None, "xyz789", HashType.SHA256)
        assert result is None

        result = HashStrategy.select_hash("abc123", None, HashType.XXH128)
        assert result is None

        result = HashStrategy.select_hash(None, None, HashType.SHA256)
        assert result is None
