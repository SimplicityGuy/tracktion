"""
Performance optimization utilities for audio analysis.

Provides streaming, chunking, and parallel processing capabilities
for efficient handling of large audio files.
"""

import gc
import logging
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import essentia.standard as es
import librosa
import numpy as np
import psutil

from .config import PerformanceConfig, get_config

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.metrics: dict[str, Any] = {}
        self.start_times: dict[str, float] = {}

    @contextmanager
    def measure(self, operation: str) -> Generator[None]:
        """Context manager to measure operation duration.

        Args:
            operation: Name of the operation being measured

        Yields:
            None

        Example:
            with monitor.measure("bpm_detection"):
                detect_bpm(file)
        """
        start_time = time.time()
        self.start_times[operation] = start_time

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics[operation] = {
                "duration_seconds": duration,
                "timestamp": start_time,
            }
            logger.debug(f"Operation '{operation}' took {duration:.3f} seconds")

    def add_metric(self, name: str, value: Any) -> None:
        """Add a custom metric.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary of metrics
        """
        # Add system metrics
        process = psutil.Process()
        self.metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        self.metrics["cpu_percent"] = process.cpu_percent()

        return self.metrics


class AudioStreamer:
    """Stream large audio files in chunks for memory-efficient processing."""

    def __init__(self, config: PerformanceConfig | None = None):
        """Initialize audio streamer.

        Args:
            config: Performance configuration
        """
        self.config = config or get_config().performance
        self.chunk_size = self.config.chunk_size_bytes

    def should_stream(self, file_path: str) -> bool:
        """Determine if file should be streamed based on size.

        Args:
            file_path: Path to the audio file

        Returns:
            True if file should be streamed
        """
        if not self.config.enable_streaming:
            return False

        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return file_size_mb > self.config.streaming_threshold_mb
        except OSError:
            return False

    def stream_audio_chunks(self, file_path: str, sample_rate: int = 44100, mono: bool = True) -> Generator[np.ndarray]:
        """Stream audio file in chunks.

        Args:
            file_path: Path to the audio file
            sample_rate: Target sample rate
            mono: Convert to mono if True

        Yields:
            Audio chunks as numpy arrays

        Note:
            This is a simplified version. In production, you'd use
            essentia.streaming or librosa streaming capabilities.
        """
        try:
            # First try to load with Essentia for better performance with streaming
            loader = es.MonoLoader(filename=file_path, sampleRate=sample_rate)
            audio = loader()
        except (ImportError, RuntimeError) as e:
            # Fallback to librosa if Essentia fails
            logger.info(f"Essentia failed, using librosa: {e}")
            try:
                audio, _loaded_sample_rate = librosa.load(file_path, sr=sample_rate, mono=mono)
                # Convert to float32 for consistency with Essentia
                audio = audio.astype(np.float32)
            except Exception as audio_error:
                raise FileNotFoundError(f"Failed to load audio file '{file_path}': {audio_error}") from audio_error
        chunk_samples = self.chunk_size // 4  # 4 bytes per float32 sample

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            if len(chunk) > 0:
                yield chunk

    def process_chunked_audio(
        self, file_path: str, processor: Callable[[np.ndarray], dict[str, Any]]
    ) -> dict[str, Any]:
        """Process audio file in chunks with a given processor.

        Args:
            file_path: Path to the audio file
            processor: Function to process each chunk

        Returns:
            Aggregated results from all chunks
        """
        results = []

        for chunk in self.stream_audio_chunks(file_path):
            chunk_result = processor(chunk)
            results.append(chunk_result)

        # Aggregate results (this is application-specific)
        return self._aggregate_chunk_results(results)

    def _aggregate_chunk_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results from multiple chunks.

        Args:
            results: List of chunk processing results

        Returns:
            Aggregated results
        """
        if not results:
            return {}

        # Example aggregation - customize based on needs
        aggregated = {
            "num_chunks": len(results),
            "chunk_results": results,
        }

        # If BPM values are present, calculate average
        if all("bpm" in r for r in results):
            bpms = [r["bpm"] for r in results]
            aggregated["average_bpm"] = sum(bpms) / len(bpms)
            aggregated["bpm_std"] = np.std(bpms)

        return aggregated


class ParallelProcessor:
    """Process multiple audio files in parallel."""

    def __init__(self, config: PerformanceConfig | None = None):
        """Initialize parallel processor.

        Args:
            config: Performance configuration
        """
        self.config = config or get_config().performance
        self.max_workers = self.config.parallel_workers

    def process_files(
        self,
        file_paths: list[str],
        processor: Callable[[str], dict[str, Any]],
        monitor: PerformanceMonitor | None = None,
    ) -> list[dict[str, Any]]:
        """Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            processor: Function to process each file
            monitor: Optional performance monitor

        Returns:
            List of results for each file
        """
        if self.max_workers <= 1:
            # Sequential processing
            results = []
            for file_path in file_paths:
                with monitor.measure(f"process_{Path(file_path).name}") if monitor else nullcontext():
                    result = processor(file_path)
                    results.append(result)
            return results

        # Parallel processing
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(processor, fp): fp for fp in file_paths}

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=self.config.processing_timeout_seconds)
                    results.append(result)

                    if monitor:
                        monitor.add_metric(f"processed_{Path(file_path).name}", True)

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results.append({"file": file_path, "error": str(e)})

                    if monitor:
                        monitor.add_metric(f"failed_{Path(file_path).name}", str(e))

        return results

    def process_in_batches(
        self,
        file_paths: list[str],
        processor: Callable[[str], dict[str, Any]],
        batch_size: int | None = None,
    ) -> Generator[list[dict[str, Any]]]:
        """Process files in batches for memory efficiency.

        Args:
            file_paths: List of file paths to process
            processor: Function to process each file
            batch_size: Size of each batch (default from config)

        Yields:
            Batch results
        """
        batch_size = batch_size or get_config().storage.batch_size

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            batch_results = self.process_files(batch, processor)
            yield batch_results


class MemoryManager:
    """Manage memory usage during processing."""

    def __init__(self, config: PerformanceConfig | None = None):
        """Initialize memory manager.

        Args:
            config: Performance configuration
        """
        self.config = config or get_config().performance
        self.memory_limit_mb = self.config.memory_limit_mb

    def check_memory(self) -> tuple[float, bool]:
        """Check current memory usage.

        Returns:
            Tuple of (memory_usage_mb, is_within_limit)
        """
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        within_limit = memory_mb < self.memory_limit_mb

        if not within_limit:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")

        return memory_mb, within_limit

    def get_memory_info(self) -> dict[str, float]:
        """Get detailed memory information.

        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process()
        virtual_mem = psutil.virtual_memory()

        return {
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "process_memory_percent": process.memory_percent(),
            "system_memory_mb": virtual_mem.used / 1024 / 1024,
            "system_memory_percent": virtual_mem.percent,
            "available_memory_mb": virtual_mem.available / 1024 / 1024,
        }

    @contextmanager
    def memory_guard(self, operation: str) -> Generator[None]:
        """Guard against excessive memory usage.

        Args:
            operation: Name of the operation

        Raises:
            MemoryError: If memory limit is exceeded

        Example:
            with memory_manager.memory_guard("large_file_processing"):
                process_large_file()
        """
        initial_memory, _ = self.check_memory()
        logger.debug(f"Starting {operation} with {initial_memory:.1f}MB memory usage")

        try:
            yield
        finally:
            final_memory, within_limit = self.check_memory()
            memory_increase = final_memory - initial_memory

            logger.debug(f"Completed {operation}: memory increased by {memory_increase:.1f}MB")

            if not within_limit:
                # In production, you might want to trigger garbage collection
                gc.collect()
                # Re-check after garbage collection
                final_memory, within_limit = self.check_memory()
                if not within_limit:
                    raise MemoryError(
                        f"Memory limit exceeded after {operation}: {final_memory:.1f}MB > {self.memory_limit_mb}MB"
                    )


class PerformanceOptimizer:
    """Main class for performance optimization coordination."""

    def __init__(self, config: PerformanceConfig | None = None):
        """Initialize performance optimizer.

        Args:
            config: Performance configuration
        """
        self.config = config or get_config().performance
        self.monitor = PerformanceMonitor()
        self.streamer = AudioStreamer(config)
        self.parallel_processor = ParallelProcessor(config)
        self.memory_manager = MemoryManager(config)

    def optimize_processing(self, file_path: str, processor: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        """Optimize processing based on file characteristics.

        Args:
            file_path: Path to the audio file
            processor: Function to process the file

        Returns:
            Processing results with performance metrics
        """
        with self.monitor.measure("total_processing"):
            # Check if streaming is needed
            if self.streamer.should_stream(file_path):
                logger.info(f"Using streaming for large file: {file_path}")
                with self.memory_manager.memory_guard("streaming_processing"):
                    # For streaming, we'd need a streaming-capable processor
                    # This is a simplified example
                    result = processor(file_path)
            else:
                with self.memory_manager.memory_guard("standard_processing"):
                    result = processor(file_path)

            # Add performance metrics to result
            result["performance_metrics"] = self.monitor.get_metrics()
            result["memory_info"] = self.memory_manager.get_memory_info()

            return result

    def optimize_batch_processing(
        self, file_paths: list[str], processor: Callable[[str], dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Optimize processing of multiple files.

        Args:
            file_paths: List of file paths to process
            processor: Function to process each file

        Returns:
            List of processing results
        """
        with self.monitor.measure("batch_processing"):
            # Determine optimal strategy based on file count and system resources
            if len(file_paths) > 10 and self.config.parallel_workers > 1:
                logger.info(f"Processing {len(file_paths)} files in parallel")
                results = self.parallel_processor.process_files(file_paths, processor, self.monitor)
            else:
                logger.info(f"Processing {len(file_paths)} files sequentially")
                results = []
                for file_path in file_paths:
                    result = self.optimize_processing(file_path, processor)
                    results.append(result)

            return results


@contextmanager
def nullcontext() -> Generator[None]:
    """Null context manager for conditional context usage."""
    yield
