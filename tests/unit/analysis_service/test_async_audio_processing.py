"""
Unit tests for async audio processing infrastructure.

Tests the async audio processor, analysis wrappers, progress tracking,
and resource management components.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.analysis_service.src.async_audio_analysis import (
    AsyncAudioAnalyzer,
    AsyncFFTProcessor,
    AudioAnalysisResult,
)
from services.analysis_service.src.async_audio_processor import (
    AsyncAudioProcessor,
    AudioTaskConfig,
    TaskPriority,
)
from services.analysis_service.src.async_progress_tracker import (
    AsyncProgressTracker,
    BatchProgressAggregator,
    ProgressEventType,
)
from services.analysis_service.src.async_resource_manager import (
    AsyncResourceManager,
    ResourceLimits,
)
from services.analysis_service.src.key_detector import KeyDetectionResult
from services.analysis_service.src.mood_analyzer import MoodAnalysisResult

# Create a random number generator
rng = np.random.default_rng()


class TestAsyncAudioProcessor:
    """Test AsyncAudioProcessor functionality."""

    @pytest.fixture
    def processor_config(self):
        """Create test configuration."""
        return AudioTaskConfig(
            min_threads=2,
            max_threads_multiplier=2.0,
            max_concurrent_analyses=4,
            max_memory_per_file_mb=50,
            task_timeout_seconds=10,
        )

    @pytest.fixture
    def processor(self, processor_config):
        """Create processor instance."""
        return AsyncAudioProcessor(processor_config)

    @pytest.mark.asyncio
    async def test_processor_initialization(self, processor):
        """Test processor initializes correctly."""
        assert processor.cpu_count > 0
        assert processor.optimal_thread_count >= processor.config.min_threads
        assert processor.optimal_thread_count <= processor.config.max_threads_absolute
        assert processor.completed_count == 0
        assert processor.failed_count == 0

    @pytest.mark.asyncio
    async def test_process_audio_async(self, processor):
        """Test async audio processing."""

        # Mock processing function
        def mock_process(audio_file):
            time.sleep(0.1)  # Simulate work
            return {"result": "processed", "file": audio_file}

        # Process audio
        result = await processor.process_audio_async("test.mp3", mock_process, priority=TaskPriority.NORMAL)

        assert result["result"] == "processed"
        assert result["file"] == "test.mp3"
        assert processor.completed_count == 1

    @pytest.mark.asyncio
    async def test_process_audio_timeout(self, processor):
        """Test processing timeout."""

        # Mock slow processing function
        def slow_process(audio_file):
            time.sleep(20)  # Exceed timeout
            return {"result": "processed"}

        # Should raise timeout error
        with pytest.raises(asyncio.TimeoutError):
            await processor.process_audio_async("test.mp3", slow_process, priority=TaskPriority.NORMAL)

        assert processor.failed_count == 1

    @pytest.mark.asyncio
    async def test_batch_process_audio(self, processor):
        """Test batch audio processing."""

        # Mock processing function
        def mock_process(audio_file):
            return {"bpm": 120, "file": audio_file}

        # Process batch
        audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
        results = await processor.batch_process_audio(audio_files, mock_process, max_batch_size=2)

        assert len(results) == 3
        for file in audio_files:
            assert file in results
            assert results[file]["success"]
            assert results[file]["result"]["file"] == file

    @pytest.mark.asyncio
    async def test_memory_check(self, processor):
        """Test memory availability checking."""
        with patch("psutil.virtual_memory") as mock_vm:
            # Simulate low memory
            mock_vm.return_value = MagicMock(available=50 * 1024 * 1024)  # 50MB
            available = await processor._check_memory_available()
            assert not available

            # Simulate sufficient memory
            mock_vm.return_value = MagicMock(available=500 * 1024 * 1024)  # 500MB
            available = await processor._check_memory_available()
            assert available

    @pytest.mark.asyncio
    async def test_get_stats(self, processor):
        """Test statistics retrieval."""
        stats = processor.get_stats()

        assert "active_tasks" in stats
        assert "completed_tasks" in stats
        assert "failed_tasks" in stats
        assert "thread_count" in stats
        assert "cpu_count" in stats
        assert "memory_mb" in stats
        assert "cpu_percent" in stats

    @pytest.mark.asyncio
    async def test_shutdown(self, processor):
        """Test processor shutdown."""

        # Create some active tasks
        async def dummy_task():
            await asyncio.sleep(10)

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())
        processor.active_tasks["task1"] = task1
        processor.active_tasks["task2"] = task2

        # Shutdown processor
        await processor.shutdown()

        # Tasks should be cancelled
        assert task1.cancelled()
        assert task2.cancelled()


class TestAsyncAudioAnalyzer:
    """Test AsyncAudioAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with mocked components."""
        processor = AsyncAudioProcessor()
        bpm_detector = MagicMock()
        key_detector = MagicMock()
        mood_analyzer = MagicMock()

        return AsyncAudioAnalyzer(
            processor=processor,
            bpm_detector=bpm_detector,
            key_detector=key_detector,
            mood_analyzer=mood_analyzer,
        )

    @pytest.mark.asyncio
    async def test_analyze_bpm_async(self, analyzer):
        """Test async BPM analysis."""
        # Mock BPM detection
        analyzer.bpm_detector.detect_bpm.return_value = {
            "bpm": 128.5,
            "confidence": 0.95,
            "needs_review": False,
        }

        result = await analyzer.analyze_bpm_async("test.mp3")

        assert result["bpm"] == 128.5
        assert result["confidence"] == 0.95
        analyzer.bpm_detector.detect_bpm.assert_called_once_with("test.mp3")

    @pytest.mark.asyncio
    async def test_analyze_key_async(self, analyzer):
        """Test async key analysis."""
        # Mock key detection

        analyzer.key_detector.detect_key.return_value = KeyDetectionResult(key="C", scale="major", confidence=0.85)

        result = await analyzer.analyze_key_async("test.mp3")

        assert result["key"] == "C"
        assert result["scale"] == "major"
        assert result["confidence"] == 0.85
        analyzer.key_detector.detect_key.assert_called_once_with("test.mp3")

    @pytest.mark.asyncio
    async def test_analyze_mood_async(self, analyzer):
        """Test async mood analysis."""
        # Mock mood analysis

        analyzer.mood_analyzer.analyze_mood.return_value = MoodAnalysisResult(
            primary_genre="Electronic",
            genre_confidence=0.8,
            danceability=0.75,
            energy=0.85,
        )

        result = await analyzer.analyze_mood_async("test.mp3")

        assert result["primary_genre"] == "Electronic"
        assert result["genre_confidence"] == 0.8
        assert result["danceability"] == 0.75
        analyzer.mood_analyzer.analyze_mood.assert_called_once_with("test.mp3")

    @pytest.mark.asyncio
    async def test_analyze_audio_complete(self, analyzer):
        """Test complete audio analysis."""
        # Mock all analyzers
        analyzer.bpm_detector.detect_bpm.return_value = {"bpm": 120}

        analyzer.key_detector.detect_key.return_value = KeyDetectionResult(key="G", scale="minor", confidence=0.9)

        analyzer.mood_analyzer.analyze_mood.return_value = MoodAnalysisResult(primary_genre="Rock", danceability=0.6)

        # Analyze
        result = await analyzer.analyze_audio_complete("test.mp3")

        assert isinstance(result, AudioAnalysisResult)
        assert result.file_path == "test.mp3"
        assert result.bpm == {"bpm": 120}
        assert result.key["key"] == "G"
        assert result.mood["primary_genre"] == "Rock"
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        # Mock analyzers
        analyzer.bpm_detector.detect_bpm.return_value = {"bpm": 125}

        analyzer.key_detector.detect_key.return_value = KeyDetectionResult(key="A", scale="major", confidence=0.8)

        # Analyze batch
        files = ["file1.mp3", "file2.mp3"]
        results = await analyzer.analyze_batch(files, enable_mood=False)

        assert len(results) == 2
        for file in files:
            assert file in results
            assert results[file].file_path == file
            assert results[file].bpm == {"bpm": 125}


class TestAsyncProgressTracker:
    """Test AsyncProgressTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return AsyncProgressTracker(enable_websocket=False)

    @pytest.mark.asyncio
    async def test_start_task(self, tracker):
        """Test starting task tracking."""
        await tracker.start_task("task1", total_stages=3, metadata={"type": "audio"})

        assert "task1" in tracker.active_tasks
        assert tracker.active_tasks["task1"].total_stages == 3
        assert tracker.active_tasks["task1"].current_stage == 0

    @pytest.mark.asyncio
    async def test_update_progress(self, tracker):
        """Test progress updates."""
        await tracker.start_task("task1", total_stages=3)

        # Update stage
        await tracker.update_progress("task1", stage="BPM Detection", stage_progress=50.0)

        progress = tracker.active_tasks["task1"]
        assert progress.stage_name == "BPM Detection"
        assert progress.stage_progress == 50.0
        assert progress.current_stage == 1

    @pytest.mark.asyncio
    async def test_complete_task(self, tracker):
        """Test task completion."""
        await tracker.start_task("task1", total_stages=2)
        await tracker.complete_task("task1", message="Success")

        assert "task1" not in tracker.active_tasks
        assert "task1" in tracker.completed_tasks

    @pytest.mark.asyncio
    async def test_fail_task(self, tracker):
        """Test task failure."""
        await tracker.start_task("task1", total_stages=2)
        await tracker.fail_task("task1", error="Processing error")

        assert "task1" not in tracker.active_tasks
        assert "task1" in tracker.failed_tasks

    @pytest.mark.asyncio
    async def test_event_listener(self, tracker):
        """Test event listener functionality."""
        events_received = []

        def listener(event):
            events_received.append(event)

        tracker.add_listener(listener)

        # Generate events
        await tracker.start_task("task1", total_stages=1)
        await tracker.complete_task("task1")

        assert len(events_received) == 2
        assert events_received[0].event_type == ProgressEventType.STARTED
        assert events_received[1].event_type == ProgressEventType.COMPLETED

    @pytest.mark.asyncio
    async def test_batch_aggregator(self, tracker):
        """Test batch progress aggregation."""
        aggregator = BatchProgressAggregator(tracker)

        # Start batch
        await aggregator.start_batch("batch1", ["task1", "task2", "task3"], metadata={"batch_type": "analysis"})

        # Update individual tasks
        await aggregator.update_batch_task("batch1", "task1", 100.0)
        await aggregator.update_batch_task("batch1", "task2", 50.0)
        await aggregator.update_batch_task("batch1", "task3", 0.0)

        # Check batch progress
        assert "batch1" in tracker.active_tasks
        progress = tracker.active_tasks["batch1"]
        # Average progress should be (100 + 50 + 0) / 3 = 50
        assert progress.overall_progress > 0

        # Complete batch
        await aggregator.complete_batch("batch1")
        assert "batch1" not in tracker.active_tasks


class TestAsyncResourceManager:
    """Test AsyncResourceManager functionality."""

    @pytest.fixture
    def resource_limits(self):
        """Create test resource limits."""
        return ResourceLimits(
            max_concurrent_analyses=2,
            max_memory_mb=1024,
            max_memory_per_task_mb=100,
            max_queue_size=10,
        )

    @pytest.fixture
    def manager(self, resource_limits):
        """Create resource manager instance."""
        return AsyncResourceManager(resource_limits)

    @pytest.mark.asyncio
    async def test_acquire_release_resources(self, manager):
        """Test resource acquisition and release."""
        # Mock system memory and CPU to be within acceptable limits
        with patch("psutil.virtual_memory") as mock_vm:
            # Set system memory usage to 50% (below the threshold)
            mock_vm.return_value = MagicMock(percent=50.0)

            # Mock the process CPU percent to be within limits
            with patch.object(manager.process, "cpu_percent", return_value=30.0):
                # Acquire resources
                acquired = await manager.acquire_resources(
                    "task1", estimated_memory_mb=50, priority=TaskPriority.NORMAL
                )
                assert acquired
                assert "task1" in manager.task_start_times
                assert manager.total_tasks_processed == 1

                # Release resources
                await manager.release_resources("task1")
                assert "task1" not in manager.task_start_times

    @pytest.mark.asyncio
    async def test_resource_limits(self, manager):
        """Test concurrent resource limits."""
        # Mock system memory and CPU to be within acceptable limits
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)

            # Mock the process CPU percent to be within limits
            with patch.object(manager.process, "cpu_percent", return_value=30.0):
                # Acquire max concurrent tasks
                acquired1 = await manager.acquire_resources("task1")
                acquired2 = await manager.acquire_resources("task2")
                assert acquired1 and acquired2

                # Third task should fail with timeout
                acquired3 = await manager.acquire_resources("task3", timeout=0.1)
                assert not acquired3

                # Release one and try again
                await manager.release_resources("task1")
                acquired3 = await manager.acquire_resources("task3")
                assert acquired3

    @pytest.mark.asyncio
    async def test_task_queue(self, manager):
        """Test task queuing."""
        executed = []

        async def mock_task(task_id):
            executed.append(task_id)
            return f"result_{task_id}"

        # Mock system resources to be within acceptable limits
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)

            with patch.object(manager.process, "cpu_percent", return_value=30.0):
                # Queue tasks
                await manager.queue_task("task1", mock_task, args=("task1",))
                await manager.queue_task("task2", mock_task, args=("task2",), priority=TaskPriority.HIGH)
                await manager.queue_task("task3", mock_task, args=("task3",), priority=TaskPriority.LOW)

                # Wait for completion
                await asyncio.sleep(0.5)

                # High priority should execute first
                assert len(executed) > 0
                # Note: Due to async nature, exact order might vary

    @pytest.mark.asyncio
    async def test_resource_monitoring(self, manager):
        """Test resource monitoring."""
        await manager.start_monitoring(interval_seconds=0.1)
        await asyncio.sleep(0.3)  # Let it monitor

        stats = manager.get_stats()
        assert "active_tasks" in stats
        assert "current_memory_mb" in stats
        assert "current_cpu_percent" in stats

        await manager.stop_monitoring()

    @pytest.mark.asyncio
    async def test_queue_overflow(self, manager):
        """Test queue overflow handling."""
        # Fill queue to capacity
        for i in range(manager.limits.max_queue_size):
            await manager.queue_task(f"task{i}", lambda: None)

        # Next queue should raise error
        with pytest.raises(ValueError, match="Task queue full"):
            await manager.queue_task("overflow", lambda: None)


class TestAsyncFFTProcessor:
    """Test AsyncFFTProcessor functionality."""

    @pytest.fixture
    def fft_processor(self):
        """Create FFT processor instance."""
        processor = AsyncAudioProcessor()
        return AsyncFFTProcessor(processor)

    @pytest.mark.asyncio
    async def test_compute_fft_async(self, fft_processor):
        """Test async FFT computation."""
        # Create test audio data
        audio_data = rng.standard_normal(44100)  # 1 second at 44.1kHz

        # Mock the essentia import and its functions
        mock_es = MagicMock()
        mock_windowing = MagicMock(return_value=lambda x: x)
        mock_fft = MagicMock(return_value=lambda x: np.fft.fft(x)[:1024])  # Return first half of FFT
        mock_es.Windowing = MagicMock(return_value=mock_windowing)
        mock_es.FFT = MagicMock(return_value=mock_fft)

        with patch.dict("sys.modules", {"essentia": MagicMock(), "essentia.standard": mock_es}):
            # Compute FFT
            result = await fft_processor.compute_fft_async(audio_data, fft_size=2048)

            assert result is not None
            assert len(result.shape) == 2  # Should be 2D array of frames

    @pytest.mark.asyncio
    async def test_compute_parallel_ffts(self, fft_processor):
        """Test parallel FFT computation."""

        # Create test segments
        segments = [rng.standard_normal(4096) for _ in range(3)]

        # Mock the essentia import and its functions
        mock_es = MagicMock()
        mock_windowing = MagicMock(return_value=lambda x: x)
        mock_fft = MagicMock(return_value=lambda x: np.fft.fft(x)[:2048])  # Return first half of FFT
        mock_es.Windowing = MagicMock(return_value=mock_windowing)
        mock_es.FFT = MagicMock(return_value=mock_fft)

        with patch.dict("sys.modules", {"essentia": MagicMock(), "essentia.standard": mock_es}):
            # Compute FFTs in parallel
            results = await fft_processor.compute_parallel_ffts(segments)

            assert len(results) == 3
            for result in results:
                assert result is not None
