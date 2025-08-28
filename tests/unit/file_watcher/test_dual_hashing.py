"""Unit tests for dual hashing functionality."""

import hashlib
import tempfile
from pathlib import Path

import pytest
import xxhash

from services.file_watcher.src.file_scanner import FileScanner


class TestDualHashing:
    """Test dual hashing implementation."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            # Write known test data
            test_data = b"Test audio file content for hash testing"
            f.write(test_data)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        temp_path.unlink()

    @pytest.fixture
    def scanner(self):
        """Create a FileScanner instance."""
        return FileScanner()

    def test_calculate_dual_hashes(self, scanner, temp_file):
        """Test that dual hash calculation returns both hashes."""
        sha256_hash, xxh128_hash = scanner._calculate_dual_hashes(temp_file)

        # Verify both hashes are returned
        assert sha256_hash is not None
        assert xxh128_hash is not None

        # Verify hash formats (hex strings)
        assert len(sha256_hash) == 64  # SHA256 is 256 bits = 64 hex chars
        assert len(xxh128_hash) == 32  # XXH128 is 128 bits = 32 hex chars

        # Verify they're different
        assert sha256_hash != xxh128_hash

    def test_dual_hashes_known_values(self, scanner):
        """Test dual hashing with known input and expected outputs."""
        # Create temp file with known content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            test_data = b"Hello, World!"
            f.write(test_data)
            temp_path = Path(f.name)

        try:
            sha256_hash, xxh128_hash = scanner._calculate_dual_hashes(temp_path)

            # Calculate expected hashes
            expected_sha256 = hashlib.sha256(test_data).hexdigest()
            expected_xxh128 = xxhash.xxh128(test_data).hexdigest()

            assert sha256_hash == expected_sha256
            assert xxh128_hash == expected_xxh128
        finally:
            temp_path.unlink()

    def test_dual_hashes_large_file(self, scanner):
        """Test dual hashing with a larger file to verify chunked reading."""
        # Create a 10MB test file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            # Write 10MB of data in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            test_chunk = b"x" * chunk_size

            sha256_expected = hashlib.sha256()
            xxh128_expected = xxhash.xxh128()

            for _ in range(10):
                f.write(test_chunk)
                sha256_expected.update(test_chunk)
                xxh128_expected.update(test_chunk)

            temp_path = Path(f.name)

        try:
            sha256_hash, xxh128_hash = scanner._calculate_dual_hashes(temp_path)

            assert sha256_hash == sha256_expected.hexdigest()
            assert xxh128_hash == xxh128_expected.hexdigest()
        finally:
            temp_path.unlink()

    def test_calculate_sha256_hash(self, scanner, temp_file):
        """Test SHA256 hash calculation helper."""
        sha256_hash = scanner._calculate_sha256_hash(temp_file)

        assert sha256_hash is not None
        assert len(sha256_hash) == 64

    def test_calculate_xxh128_hash(self, scanner, temp_file):
        """Test XXH128 hash calculation helper."""
        xxh128_hash = scanner._calculate_xxh128_hash(temp_file)

        assert xxh128_hash is not None
        assert len(xxh128_hash) == 32

    def test_get_file_info_includes_both_hashes(self, scanner, temp_file):
        """Test that file info includes both hash types."""
        file_info = scanner._get_file_info(temp_file)

        # Verify both hashes are present
        assert "sha256_hash" in file_info
        assert "xxh128_hash" in file_info

        # Verify hash formats
        assert len(file_info["sha256_hash"]) == 64
        assert len(file_info["xxh128_hash"]) == 32

        # Verify other expected fields
        assert "path" in file_info
        assert "name" in file_info
        assert "extension" in file_info
        assert "size_bytes" in file_info
        assert "modified_time" in file_info

    def test_dual_hashes_error_handling(self, scanner):
        """Test dual hash calculation with non-existent file."""
        non_existent = Path("/non/existent/file.mp3")

        sha256_hash, xxh128_hash = scanner._calculate_dual_hashes(non_existent)

        # Should return fallback hashes based on path and size
        assert sha256_hash is not None
        assert xxh128_hash is not None
        assert len(sha256_hash) == 64
        assert len(xxh128_hash) == 32

    def test_empty_file_hashing(self, scanner):
        """Test hashing an empty file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            # Empty file
            temp_path = Path(f.name)

        try:
            sha256_hash, xxh128_hash = scanner._calculate_dual_hashes(temp_path)

            # Expected hashes for empty input
            expected_sha256 = hashlib.sha256(b"").hexdigest()
            expected_xxh128 = xxhash.xxh128(b"").hexdigest()

            assert sha256_hash == expected_sha256
            assert xxh128_hash == expected_xxh128
        finally:
            temp_path.unlink()
