"""
Integration tests for async audio processing components.

Tests the CPU optimizer, error handler, and message queue integration.
"""

import asyncio
import contextlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.analysis_service.src.async_audio_analysis import AudioAnalysisResult
from services.analysis_service.src.async_cpu_optimizer import (
    AsyncCPUOptimizer,
    CPUProfile,
    OptimizationConfig,
    ParallelFFTOptimizer,
)
from services.analysis_service.src.async_error_handler import (
    AsyncErrorHandler,
    AudioFallbackHandler,
    ErrorType,
    ResourceCleanupManager,
    RetryPolicy,
)
from services.analysis_service.src.async_message_integration import (
    AnalysisRequest,
    AsyncMessageQueueIntegration,
    TaskPriority,
)

# Create a random number generator
rng = np.random.default_rng()


class TestAsyncCPUOptimizer:
    """Test AsyncCPUOptimizer functionality."""

    @pytest.fixture
    def optimizer_config(self):
        """Create test configuration."""
        return OptimizationConfig(
            enable_profiling=True,
            enable_dynamic_sizing=True,
            profile_interval_seconds=0.1,
            target_cpu_utilization=0.7,
        )

    @pytest.fixture
    def optimizer(self, optimizer_config):
        """Create optimizer instance."""
        return AsyncCPUOptimizer(optimizer_config, cpu_count=4)

    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer.cpu_count == 4
        assert optimizer.current_thread_count == 8  # 2x CPU count
        assert optimizer.optimal_thread_count == 8
        assert len(optimizer.profiles) == 0

    @pytest.mark.asyncio
    async def test_start_stop_profiling(self, optimizer):
        """Test starting and stopping profiling."""
        await optimizer.start_profiling()
        assert optimizer.profiling_task is not None
        assert not optimizer.profiling_task.done()

        await optimizer.stop_profiling()
        assert optimizer.profiling_task is None

    @pytest.mark.asyncio
    async def test_collect_profile(self, optimizer):
        """Test CPU profile collection."""
        with patch("psutil.cpu_percent") as mock_cpu:
            mock_cpu.side_effect = [75.0, [80.0, 70.0, 75.0, 65.0]]

            profile = await optimizer._collect_profile()

            assert isinstance(profile, CPUProfile)
            assert profile.overall_percent == 75.0
            assert len(profile.per_core_percent) == 4
            assert profile.efficiency_score >= 0.0
            assert profile.efficiency_score <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_efficiency(self, optimizer):
        """Test efficiency calculation."""
        # Perfect efficiency scenario
        efficiency = optimizer._calculate_efficiency(
            overall_cpu=70.0,  # Target is 70%
            process_cpu=60.0,
            task_count=8,
            thread_count=8,
        )
        assert efficiency > 0.8

        # Poor efficiency scenario
        efficiency = optimizer._calculate_efficiency(
            overall_cpu=20.0,  # Under-utilized
            process_cpu=10.0,
            task_count=1,
            thread_count=8,
        )
        assert efficiency < 0.5

    @pytest.mark.asyncio
    async def test_dynamic_thread_adjustment(self, optimizer):
        """Test dynamic thread pool adjustment."""
        # Create under-utilized profile
        profile = CPUProfile(
            timestamp=time.time(),
            overall_percent=30.0,  # Under target
            per_core_percent=[30.0] * 4,
            process_percent=25.0,
            thread_count=8,
            task_count=4,
            avg_task_time=0.1,
            efficiency_score=0.5,
        )

        # Simulate warmup period passed
        optimizer.start_time = time.time() - 60

        await optimizer._analyze_and_optimize(profile)

        # Should increase threads
        assert optimizer.current_thread_count > 8

    @pytest.mark.asyncio
    async def test_task_time_tracking(self, optimizer):
        """Test task time tracking."""
        optimizer.record_task_start("task1")
        assert "task1" in optimizer.task_start_times

        await asyncio.sleep(0.1)
        optimizer.record_task_completion("task1")

        assert "task1" not in optimizer.task_start_times
        assert len(optimizer.task_times) > 0
        assert optimizer.task_times[-1] >= 0.1

    @pytest.mark.asyncio
    async def test_get_stats(self, optimizer):
        """Test statistics retrieval."""
        stats = optimizer.get_optimization_stats()

        assert "current_threads" in stats
        assert "optimal_threads" in stats
        assert "cpu_count" in stats
        assert "avg_cpu_percent" in stats
        assert "avg_efficiency" in stats


class TestParallelFFTOptimizer:
    """Test ParallelFFTOptimizer functionality."""

    @pytest.fixture
    def fft_optimizer(self):
        """Create FFT optimizer instance."""
        return ParallelFFTOptimizer(cpu_count=4)

    @pytest.mark.asyncio
    async def test_parallel_stft(self, fft_optimizer):
        """Test parallel STFT computation."""
        # Create test audio
        audio_data = rng.standard_normal(44100)  # 1 second

        result = await fft_optimizer.compute_parallel_stft(audio_data, fft_size=2048, n_parallel=2)

        assert result is not None
        assert len(result.shape) == 2

    @pytest.mark.asyncio
    async def test_optimize_fft_size(self, fft_optimizer):
        """Test FFT size optimization."""
        # Target 10 Hz resolution at 44.1kHz
        fft_size = fft_optimizer.optimize_fft_size(44100, 10.0)

        assert fft_size >= 4096
        assert fft_size <= 8192
        # Should be power of 2
        assert (fft_size & (fft_size - 1)) == 0

    @pytest.mark.asyncio
    async def test_benchmark_performance(self, fft_optimizer):
        """Test FFT performance benchmarking."""
        results = await fft_optimizer.benchmark_fft_performance(0.1)

        assert len(results) > 0
        for duration in results.values():
            assert duration > 0
            assert duration < 10  # Should complete in reasonable time


class TestAsyncErrorHandler:
    """Test AsyncErrorHandler functionality."""

    @pytest.fixture
    def retry_policy(self):
        """Create test retry policy."""
        return RetryPolicy(
            max_retries=2,
            initial_delay_seconds=0.1,
            max_delay_seconds=1.0,
            retry_on_timeout=True,
        )

    @pytest.fixture
    def error_handler(self, retry_policy):
        """Create error handler instance."""
        return AsyncErrorHandler(retry_policy)

    @pytest.mark.asyncio
    async def test_successful_execution(self, error_handler):
        """Test successful function execution."""

        async def success_func():
            return "success"

        result = await error_handler.handle_with_retry(success_func, task_id="test1", audio_file="test.mp3")

        assert result == "success"
        assert error_handler.circuit_failures == 0

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, error_handler):
        """Test retry logic on timeout."""
        call_count = 0

        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return "success"

        result = await error_handler.handle_with_retry(timeout_func, task_id="test2", audio_file="test.mp3")

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, error_handler):
        """Test behavior when max retries exceeded."""

        async def always_fails():
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError):
            await error_handler.handle_with_retry(always_fails, task_id="test3", audio_file="test.mp3")

        assert len(error_handler.error_history) > 0
        assert error_handler.error_counts[ErrorType.UNKNOWN] > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, error_handler):
        """Test circuit breaker functionality."""
        error_handler.circuit_threshold = 2

        async def always_fails():
            raise RuntimeError("Fails")

        # Trip the circuit
        for i in range(2):
            with contextlib.suppress(Exception):
                await error_handler.handle_with_retry(always_fails, task_id=f"test{i}", audio_file="test.mp3")

        assert error_handler.circuit_open

        # Should fail immediately when circuit is open
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            await error_handler.handle_with_retry(lambda: "success", task_id="test_open", audio_file="test.mp3")

    @pytest.mark.asyncio
    async def test_error_classification(self, error_handler):
        """Test error type classification."""
        assert error_handler._classify_error(TimeoutError("timeout")) == ErrorType.TIMEOUT
        assert error_handler._classify_error(MemoryError("memory")) == ErrorType.MEMORY
        assert error_handler._classify_error(RuntimeError("corrupted file")) == ErrorType.CORRUPTED_FILE
        assert error_handler._classify_error(ConnectionError("network error")) == ErrorType.NETWORK


class TestAudioFallbackHandler:
    """Test AudioFallbackHandler functionality."""

    @pytest.fixture
    def fallback_handler(self):
        """Create fallback handler instance."""
        return AudioFallbackHandler()

    @pytest.mark.asyncio
    async def test_primary_strategy_success(self, fallback_handler):
        """Test successful primary strategy."""

        async def primary(audio_file):
            return {"result": "primary", "file": audio_file}

        async def fallback1(audio_file):
            return {"result": "fallback1"}

        result = await fallback_handler.process_with_fallback("test.mp3", primary, [fallback1])

        assert result["result"] == "primary"
        assert fallback_handler.successful_fallbacks == 1

    @pytest.mark.asyncio
    async def test_fallback_strategy_used(self, fallback_handler):
        """Test fallback strategy when primary fails."""

        async def primary(audio_file):
            raise RuntimeError("Primary failed")

        async def fallback1(audio_file):
            return {"result": "fallback1", "file": audio_file}

        result = await fallback_handler.process_with_fallback("test.mp3", primary, [fallback1])

        assert result["result"] == "fallback1"

    @pytest.mark.asyncio
    async def test_all_strategies_fail(self, fallback_handler):
        """Test when all strategies fail."""

        async def always_fails(audio_file):
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError, match="All processing strategies failed"):
            await fallback_handler.process_with_fallback("test.mp3", always_fails, [always_fails, always_fails])


class TestResourceCleanupManager:
    """Test ResourceCleanupManager functionality."""

    @pytest.fixture
    def cleanup_manager(self):
        """Create cleanup manager instance."""
        return ResourceCleanupManager()

    @pytest.mark.asyncio
    async def test_register_and_cleanup(self, cleanup_manager):
        """Test resource registration and cleanup."""
        cleaned = []

        async def cleanup_func(resource_id):
            cleaned.append(resource_id)

        await cleanup_manager.register_resource("resource1", cleanup_func, "resource1")
        await cleanup_manager.cleanup_resource("resource1")

        assert "resource1" in cleaned
        assert "resource1" not in cleanup_manager.resources_to_cleanup

    @pytest.mark.asyncio
    async def test_cleanup_all(self, cleanup_manager):
        """Test cleaning up all resources."""
        cleaned = []

        async def cleanup_func(resource_id):
            cleaned.append(resource_id)

        for i in range(3):
            await cleanup_manager.register_resource(f"resource{i}", cleanup_func, f"resource{i}")

        await cleanup_manager.cleanup_all()

        assert len(cleaned) == 3
        assert len(cleanup_manager.resources_to_cleanup) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test cleanup manager as context manager."""
        cleaned = []

        async def cleanup_func(resource_id):
            cleaned.append(resource_id)

        async with ResourceCleanupManager() as manager:
            await manager.register_resource("resource1", cleanup_func, "resource1")

        assert "resource1" in cleaned


class TestAsyncMessageQueueIntegration:
    """Test AsyncMessageQueueIntegration functionality."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        return {
            "processor": MagicMock(),
            "analyzer": AsyncMock(),
            "tracker": AsyncMock(),
            "resource_manager": AsyncMock(),
            "error_handler": AsyncMock(),
        }

    @pytest.fixture
    def integration(self, mock_components):
        """Create integration instance."""
        return AsyncMessageQueueIntegration(
            rabbitmq_url="amqp://localhost",
            **mock_components,
            enable_batch_processing=True,
            batch_size=3,
        )

    @pytest.mark.asyncio
    async def test_parse_message(self, integration):
        """Test message parsing."""
        message_data = {
            "recording_id": "rec123",
            "file_path": "/audio/test.mp3",
            "analysis_types": ["bpm", "key"],
            "metadata": {"source": "test"},
        }

        mock_message = MagicMock()
        mock_message.body = json.dumps(message_data).encode()
        mock_message.priority = 7
        mock_message.correlation_id = "corr123"

        request = await integration._parse_message(mock_message)

        assert isinstance(request, AnalysisRequest)
        assert request.recording_id == "rec123"
        assert request.file_path == "/audio/test.mp3"
        assert "bpm" in request.analysis_types

        assert request.priority == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_batch_processing(self, integration):
        """Test batch message processing."""
        # Create test requests
        requests = []
        messages = []

        for i in range(3):
            request = AnalysisRequest(
                recording_id=f"rec{i}",
                file_path=f"/audio/test{i}.mp3",
                analysis_types=["bpm"],
                priority=TaskPriority.NORMAL,
                metadata={},
            )
            requests.append(request)

            mock_message = AsyncMock()
            messages.append(mock_message)

        # Add to batch
        for req, msg in zip(requests, messages, strict=False):
            integration.batch_buffer.append(req)
            integration.batch_messages.append(msg)

        # Mock analyzer

        integration.analyzer.analyze_audio_complete.return_value = AudioAnalysisResult(
            file_path="/audio/test.mp3",
            bpm={"bpm": 120},
        )

        # Process batch
        await integration._process_batch()

        # Verify messages were acknowledged
        for msg in messages:
            msg.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_batch(self, integration):
        """Test error handling in batch processing."""

        # Create failing request
        request = AnalysisRequest(
            recording_id="rec_fail",
            file_path="/audio/fail.mp3",
            analysis_types=["bpm"],
            priority=TaskPriority.NORMAL,
            metadata={},
        )

        mock_message = AsyncMock()

        integration.batch_buffer.append(request)
        integration.batch_messages.append(mock_message)

        # Make analyzer fail
        integration.analyzer.analyze_audio_complete.side_effect = RuntimeError("Analysis failed")

        # Process batch
        await integration._process_batch()

        # Message should be nacked for requeue
        mock_message.nack.assert_called_once_with(requeue=True)
