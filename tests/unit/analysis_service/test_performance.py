"""
Unit tests for performance optimization utilities.

Tests streaming, parallel processing, and memory management.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from services.analysis_service.src.config import PerformanceConfig
from services.analysis_service.src.performance import (
    AudioStreamer,
    MemoryManager,
    ParallelProcessor,
    PerformanceMonitor,
    PerformanceOptimizer,
)


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""

    def test_measure_operation(self):
        """Test measuring operation duration."""
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.1)  # Simulate work

        metrics = monitor.get_metrics()
        assert "test_operation" in metrics
        assert "duration_seconds" in metrics["test_operation"]
        assert metrics["test_operation"]["duration_seconds"] >= 0.1

    def test_add_custom_metric(self):
        """Test adding custom metrics."""
        monitor = PerformanceMonitor()
        monitor.add_metric("files_processed", 10)
        monitor.add_metric("cache_hits", 5)

        metrics = monitor.get_metrics()
        assert metrics["files_processed"] == 10
        assert metrics["cache_hits"] == 5

    def test_system_metrics(self):
        """Test that system metrics are included."""
        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()

        assert "memory_usage_mb" in metrics
        assert "cpu_percent" in metrics
        assert metrics["memory_usage_mb"] > 0

    def test_nested_measurements(self):
        """Test nested operation measurements."""
        monitor = PerformanceMonitor()

        with monitor.measure("outer"):
            time.sleep(0.05)
            with monitor.measure("inner"):
                time.sleep(0.05)

        metrics = monitor.get_metrics()
        assert "outer" in metrics
        assert "inner" in metrics
        assert metrics["outer"]["duration_seconds"] >= 0.1
        assert metrics["inner"]["duration_seconds"] >= 0.05


class TestAudioStreamer:
    """Test suite for AudioStreamer."""

    def test_should_stream_enabled(self):
        """Test streaming decision when enabled."""
        config = PerformanceConfig(enable_streaming=True, streaming_threshold_mb=10)
        streamer = AudioStreamer(config)

        # Mock file size
        with patch("os.path.getsize") as mock_size:
            # Large file - should stream
            mock_size.return_value = 20 * 1024 * 1024  # 20 MB
            assert streamer.should_stream("large.mp3") is True

            # Small file - should not stream
            mock_size.return_value = 5 * 1024 * 1024  # 5 MB
            assert streamer.should_stream("small.mp3") is False

    def test_should_stream_disabled(self):
        """Test streaming decision when disabled."""
        config = PerformanceConfig(enable_streaming=False)
        streamer = AudioStreamer(config)

        with patch("os.path.getsize") as mock_size:
            mock_size.return_value = 200 * 1024 * 1024  # 200 MB
            assert streamer.should_stream("huge.mp3") is False

    def test_should_stream_file_not_found(self):
        """Test streaming decision for non-existent file."""
        streamer = AudioStreamer()
        assert streamer.should_stream("/nonexistent/file.mp3") is False

    def test_stream_audio_chunks(self):
        """Test audio streaming in chunks."""
        streamer = AudioStreamer()
        streamer.chunk_size = 1024  # Small chunks for testing

        # Use a non-existent file which will trigger the fallback
        chunks = list(streamer.stream_audio_chunks("nonexistent.mp3"))

        # Should have multiple chunks
        assert len(chunks) > 1
        # All chunks should be numpy arrays
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        # Should have generated some audio
        total_samples = sum(len(chunk) for chunk in chunks)
        assert total_samples > 0

    def test_process_chunked_audio(self):
        """Test processing audio in chunks."""
        streamer = AudioStreamer()

        # Mock chunking
        mock_chunks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        with patch.object(streamer, "stream_audio_chunks", return_value=mock_chunks):
            # Simple processor that returns mean
            def processor(chunk):
                return {"bpm": float(np.mean(chunk))}

            result = streamer.process_chunked_audio("test.mp3", processor)

            assert result["num_chunks"] == 2
            assert "average_bpm" in result
            assert result["average_bpm"] == 3.5  # (2 + 5) / 2

    def test_aggregate_chunk_results(self):
        """Test chunk result aggregation."""
        streamer = AudioStreamer()

        results = [
            {"bpm": 120.0, "confidence": 0.9},
            {"bpm": 122.0, "confidence": 0.85},
            {"bpm": 121.0, "confidence": 0.88},
        ]

        aggregated = streamer._aggregate_chunk_results(results)

        assert aggregated["num_chunks"] == 3
        assert aggregated["average_bpm"] == 121.0
        assert "bpm_std" in aggregated


class TestParallelProcessor:
    """Test suite for ParallelProcessor."""

    def test_sequential_processing(self):
        """Test sequential processing when workers = 1."""
        config = PerformanceConfig(parallel_workers=1)
        processor = ParallelProcessor(config)

        files = ["file1.mp3", "file2.mp3", "file3.mp3"]

        def mock_processor(file_path):
            return {"file": file_path, "processed": True}

        results = processor.process_files(files, mock_processor)

        assert len(results) == 3
        assert all(r["processed"] for r in results)

    def test_parallel_processing(self):
        """Test parallel processing with multiple workers."""
        config = PerformanceConfig(parallel_workers=3)
        processor = ParallelProcessor(config)

        files = ["file1.mp3", "file2.mp3", "file3.mp3"]

        def mock_processor(file_path):
            time.sleep(0.05)  # Simulate work
            return {"file": file_path, "processed": True}

        start = time.time()
        results = processor.process_files(files, mock_processor)
        duration = time.time() - start

        assert len(results) == 3
        assert all(r["processed"] for r in results)
        # Should be faster than sequential (0.15s)
        assert duration < 0.15

    def test_parallel_processing_with_error(self):
        """Test parallel processing with errors."""
        config = PerformanceConfig(parallel_workers=2)
        processor = ParallelProcessor(config)

        files = ["file1.mp3", "file2.mp3", "file3.mp3"]

        def mock_processor(file_path):
            if file_path == "file2.mp3":
                raise ValueError("Processing error")
            return {"file": file_path, "processed": True}

        results = processor.process_files(files, mock_processor)

        # Should still get 3 results
        assert len(results) == 3
        # One should have an error
        error_results = [r for r in results if "error" in r]
        assert len(error_results) == 1

    def test_process_in_batches(self):
        """Test batch processing."""
        processor = ParallelProcessor()

        files = [f"file{i}.mp3" for i in range(10)]

        def mock_processor(file_path):
            return {"file": file_path}

        batches = list(processor.process_in_batches(files, mock_processor, batch_size=3))

        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1

    def test_parallel_processing_with_monitor(self):
        """Test parallel processing with performance monitoring."""
        processor = ParallelProcessor()
        monitor = PerformanceMonitor()

        files = ["file1.mp3", "file2.mp3"]

        def mock_processor(file_path):
            return {"file": file_path}

        results = processor.process_files(files, mock_processor, monitor)

        assert len(results) == 2
        metrics = monitor.get_metrics()
        # Check for either the process or processed metrics
        has_metrics = any(
            key in metrics
            for key in [
                "processed_file1.mp3",
                "processed_file2.mp3",
                "process_file1.mp3",
                "process_file2.mp3",
            ]
        )
        assert has_metrics


class TestMemoryManager:
    """Test suite for MemoryManager."""

    def test_check_memory(self):
        """Test memory checking."""
        config = PerformanceConfig(memory_limit_mb=10000)  # High limit
        manager = MemoryManager(config)

        memory_mb, within_limit = manager.check_memory()

        assert memory_mb > 0
        assert within_limit is True

    def test_check_memory_exceeded(self):
        """Test memory checking when limit exceeded."""
        config = PerformanceConfig(memory_limit_mb=1)  # Very low limit
        manager = MemoryManager(config)

        memory_mb, within_limit = manager.check_memory()

        assert memory_mb > 1
        assert within_limit is False

    def test_get_memory_info(self):
        """Test getting detailed memory information."""
        manager = MemoryManager()
        info = manager.get_memory_info()

        assert "process_memory_mb" in info
        assert "process_memory_percent" in info
        assert "system_memory_mb" in info
        assert "system_memory_percent" in info
        assert "available_memory_mb" in info

        assert all(v >= 0 for v in info.values())

    def test_memory_guard_success(self):
        """Test memory guard with successful operation."""
        config = PerformanceConfig(memory_limit_mb=10000)  # High limit
        manager = MemoryManager(config)

        with manager.memory_guard("test_operation"):
            # Simulate work
            _ = [0] * 1000

        # Should complete without error
        assert True

    def test_memory_guard_exceeded(self):
        """Test memory guard when limit exceeded."""
        config = PerformanceConfig(memory_limit_mb=1)  # Very low limit
        manager = MemoryManager(config)

        with pytest.raises(MemoryError, match="Memory limit exceeded"), manager.memory_guard("test_operation"):
            # Current process already uses more than 1MB
            pass


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer."""

    def test_optimize_processing_standard(self):
        """Test standard file processing optimization."""
        config = PerformanceConfig(enable_streaming=True, streaming_threshold_mb=100)
        optimizer = PerformanceOptimizer(config)

        def mock_processor(file_path):
            return {"file": file_path, "bpm": 120.0}

        with patch("os.path.getsize", return_value=50 * 1024 * 1024):  # 50MB
            result = optimizer.optimize_processing("test.mp3", mock_processor)

        assert result["file"] == "test.mp3"
        assert result["bpm"] == 120.0
        assert "performance_metrics" in result
        assert "memory_info" in result

    def test_optimize_processing_streaming(self):
        """Test streaming file processing optimization."""
        config = PerformanceConfig(enable_streaming=True, streaming_threshold_mb=50)
        optimizer = PerformanceOptimizer(config)

        def mock_processor(file_path):
            return {"file": file_path, "bpm": 128.0}

        with patch("os.path.getsize", return_value=100 * 1024 * 1024):  # 100MB
            result = optimizer.optimize_processing("large.mp3", mock_processor)

        assert result["file"] == "large.mp3"
        assert "performance_metrics" in result

    def test_optimize_batch_processing_sequential(self):
        """Test batch processing with sequential strategy."""
        config = PerformanceConfig(parallel_workers=1)
        optimizer = PerformanceOptimizer(config)

        files = ["file1.mp3", "file2.mp3"]

        def mock_processor(file_path):
            return {"file": file_path}

        with patch("os.path.getsize", return_value=10 * 1024 * 1024):  # 10MB each
            results = optimizer.optimize_batch_processing(files, mock_processor)

        assert len(results) == 2
        assert all("performance_metrics" in r for r in results)

    def test_optimize_batch_processing_parallel(self):
        """Test batch processing with parallel strategy."""
        config = PerformanceConfig(parallel_workers=4)
        optimizer = PerformanceOptimizer(config)

        files = [f"file{i}.mp3" for i in range(15)]  # Many files

        def mock_processor(file_path):
            return {"file": file_path}

        results = optimizer.optimize_batch_processing(files, mock_processor)

        assert len(results) == 15

    def test_optimize_with_memory_guard(self):
        """Test optimization with memory guard."""
        config = PerformanceConfig(memory_limit_mb=10000)
        optimizer = PerformanceOptimizer(config)

        def mock_processor(file_path):
            # Allocate some memory
            data = np.zeros(1000)
            return {"file": file_path, "data_size": len(data)}

        result = optimizer.optimize_processing("test.mp3", mock_processor)

        assert result["data_size"] == 1000
        assert "memory_info" in result
