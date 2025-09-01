#!/usr/bin/env python3
"""Benchmark script for dual-hash performance testing."""

import hashlib
import os
import tempfile
import time
from pathlib import Path

import xxhash


def create_test_file(size_mb: int) -> Path:
    """Create a temporary test file of specified size.

    Args:
        size_mb: Size of file to create in megabytes

    Returns:
        Path to the created file

    """
    data = os.urandom(size_mb * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".test") as f:
        f.write(data)
        return Path(f.name)


def benchmark_single_hash_sha256(file_path: Path, chunk_size: int = 8192) -> tuple[str, float]:
    """Benchmark SHA256 hashing alone.

    Args:
        file_path: Path to file to hash
        chunk_size: Size of chunks to read

    Returns:
        Tuple of (hash, time_taken)

    """
    start_time = time.perf_counter()
    sha256_hasher = hashlib.sha256()

    with Path(file_path).open("rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hasher.update(chunk)

    hash_result = sha256_hasher.hexdigest()
    elapsed_time = time.perf_counter() - start_time

    return hash_result, elapsed_time


def benchmark_single_hash_xxh128(file_path: Path, chunk_size: int = 8192) -> tuple[str, float]:
    """Benchmark XXH128 hashing alone.

    Args:
        file_path: Path to file to hash
        chunk_size: Size of chunks to read

    Returns:
        Tuple of (hash, time_taken)

    """
    start_time = time.perf_counter()
    xxh128_hasher = xxhash.xxh128()

    with Path(file_path).open("rb") as f:
        while chunk := f.read(chunk_size):
            xxh128_hasher.update(chunk)

    hash_result = xxh128_hasher.hexdigest()
    elapsed_time = time.perf_counter() - start_time

    return hash_result, elapsed_time


def benchmark_dual_hashes(file_path: Path, chunk_size: int = 8192) -> tuple[str, str, float]:
    """Benchmark dual hashing (SHA256 + XXH128).

    Args:
        file_path: Path to file to hash
        chunk_size: Size of chunks to read

    Returns:
        Tuple of (sha256_hash, xxh128_hash, time_taken)

    """
    start_time = time.perf_counter()
    sha256_hasher = hashlib.sha256()
    xxh128_hasher = xxhash.xxh128()

    with Path(file_path).open("rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hasher.update(chunk)
            xxh128_hasher.update(chunk)

    sha256_result = sha256_hasher.hexdigest()
    xxh128_result = xxh128_hasher.hexdigest()
    elapsed_time = time.perf_counter() - start_time

    return sha256_result, xxh128_result, elapsed_time


def run_benchmarks() -> None:
    """Run benchmarks for various file sizes."""
    test_sizes = [1, 100, 1000]  # 1MB, 100MB, 1GB

    print("Hash Performance Benchmark")
    print("=" * 60)
    print(f"{'Size':<10} {'SHA256 Only':<15} {'XXH128 Only':<15} {'Dual Hash':<15} {'Overhead':<10}")
    print("-" * 60)

    for size_mb in test_sizes:
        # Create test file
        test_file = create_test_file(size_mb)

        try:
            # Run benchmarks
            sha256_hash, sha256_time = benchmark_single_hash_sha256(test_file)
            xxh128_hash, xxh128_time = benchmark_single_hash_xxh128(test_file)
            dual_sha256, dual_xxh128, dual_time = benchmark_dual_hashes(test_file)

            # Calculate overhead
            overhead_pct = ((dual_time - sha256_time) / sha256_time) * 100 if sha256_time > 0 else 0

            # Verify dual hashing produces same results
            assert sha256_hash == dual_sha256, "SHA256 hash mismatch!"
            assert xxh128_hash == dual_xxh128, "XXH128 hash mismatch!"

            # Print results
            print(
                f"{size_mb}MB".ljust(10)
                + f"{sha256_time:.3f}s".ljust(15)
                + f"{xxh128_time:.3f}s".ljust(15)
                + f"{dual_time:.3f}s".ljust(15)
                + f"+{overhead_pct:.1f}%".ljust(10),
            )

        finally:
            # Cleanup test file
            test_file.unlink()

    print("=" * 60)
    print("\nPerformance Characteristics:")
    print("- XXH128 alone is typically 10-50x faster than SHA256")
    print("- Dual hashing overhead is minimal (< 5% over SHA256 alone)")
    print("- Single file read pass for both hashes is efficient")
    print("- Benefits: Fast lookups with XXH128, integrity with SHA256")


if __name__ == "__main__":
    run_benchmarks()
