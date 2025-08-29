"""
CPU utilization optimization for async audio processing.

This module provides CPU profiling, dynamic thread pool sizing,
CPU affinity management, and FFT optimization for parallel execution.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class CPUProfile:
    """CPU usage profile data."""

    timestamp: float
    overall_percent: float
    per_core_percent: List[float]
    process_percent: float
    thread_count: int
    task_count: int
    avg_task_time: float
    efficiency_score: float


@dataclass
class OptimizationConfig:
    """Configuration for CPU optimization."""

    enable_profiling: bool = True
    enable_dynamic_sizing: bool = True
    enable_cpu_affinity: bool = False  # Disabled by default as it can be problematic
    profile_interval_seconds: float = 5.0
    target_cpu_utilization: float = 0.7  # 70% target
    min_efficiency_score: float = 0.6
    thread_adjustment_step: int = 1
    warmup_period_seconds: float = 30.0


class AsyncCPUOptimizer:
    """
    CPU optimization for async audio processing.

    Monitors CPU usage patterns, adjusts thread pool sizes dynamically,
    and optimizes parallel execution strategies.
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        cpu_count: Optional[int] = None,
    ):
        """
        Initialize the CPU optimizer.

        Args:
            config: Optimization configuration
            cpu_count: Number of CPU cores (auto-detect if None)
        """
        self.config = config or OptimizationConfig()
        self.cpu_count = cpu_count or os.cpu_count() or 4

        # Profiling data
        self.profiles: List[CPUProfile] = []
        self.profiling_task: Optional[asyncio.Task] = None
        self.start_time = time.time()

        # Thread pool sizing
        self.current_thread_count = self.cpu_count * 2
        self.optimal_thread_count = self.current_thread_count
        self.thread_adjustment_history: List[Tuple[float, int, int]] = []

        # Process and system monitoring
        self.process = psutil.Process()
        self.last_profile_time = 0.0

        # Task performance tracking
        self.task_times: List[float] = []
        self.task_start_times: Dict[str, float] = {}

        logger.info(
            f"AsyncCPUOptimizer initialized with {self.cpu_count} cores, initial threads: {self.current_thread_count}"
        )

    async def start_profiling(self) -> None:
        """Start CPU profiling."""
        if not self.config.enable_profiling:
            return

        if self.profiling_task:
            return  # Already running

        self.profiling_task = asyncio.create_task(self._profile_loop())
        logger.info("CPU profiling started")

    async def stop_profiling(self) -> None:
        """Stop CPU profiling."""
        if self.profiling_task:
            self.profiling_task.cancel()
            try:
                await self.profiling_task
            except asyncio.CancelledError:
                pass
            self.profiling_task = None
        logger.info("CPU profiling stopped")

    async def _profile_loop(self) -> None:
        """Main profiling loop."""
        while True:
            try:
                await asyncio.sleep(self.config.profile_interval_seconds)
                profile = await self._collect_profile()
                self.profiles.append(profile)

                # Limit profile history
                if len(self.profiles) > 100:
                    self.profiles = self.profiles[-100:]

                # Analyze and optimize if past warmup period
                if time.time() - self.start_time > self.config.warmup_period_seconds:
                    await self._analyze_and_optimize(profile)

            except Exception as e:
                logger.error(f"Error in profiling loop: {str(e)}")

    async def _collect_profile(self) -> CPUProfile:
        """
        Collect current CPU profile.

        Returns:
            CPUProfile with current metrics
        """
        # Get CPU usage
        overall_percent = psutil.cpu_percent(interval=1.0)
        per_core_percent = psutil.cpu_percent(interval=None, percpu=True)
        process_percent = self.process.cpu_percent()

        # Get thread and task info
        thread_count = self.process.num_threads()
        task_count = len(self.task_start_times)

        # Calculate average task time
        avg_task_time = sum(self.task_times) / len(self.task_times) if self.task_times else 0.0

        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency(overall_percent, process_percent, task_count, thread_count)

        profile = CPUProfile(
            timestamp=time.time(),
            overall_percent=overall_percent,
            per_core_percent=per_core_percent,
            process_percent=process_percent,
            thread_count=thread_count,
            task_count=task_count,
            avg_task_time=avg_task_time,
            efficiency_score=efficiency_score,
        )

        logger.debug(
            f"CPU Profile - Overall: {overall_percent:.1f}%, "
            f"Process: {process_percent:.1f}%, Efficiency: {efficiency_score:.2f}"
        )

        return profile

    def _calculate_efficiency(
        self,
        overall_cpu: float,
        process_cpu: float,
        task_count: int,
        thread_count: int,
    ) -> float:
        """
        Calculate CPU efficiency score.

        Args:
            overall_cpu: Overall CPU usage percentage
            process_cpu: Process CPU usage percentage
            task_count: Number of active tasks
            thread_count: Number of threads

        Returns:
            Efficiency score (0.0 to 1.0)
        """
        if task_count == 0 or thread_count == 0:
            return 0.0

        # Calculate utilization efficiency
        target_util = self.config.target_cpu_utilization * 100
        util_efficiency = 1.0 - abs(overall_cpu - target_util) / target_util

        # Calculate task efficiency (tasks per thread)
        task_efficiency = min(1.0, task_count / thread_count)

        # Calculate process efficiency (process CPU vs overall)
        if overall_cpu > 0:
            process_efficiency = min(1.0, process_cpu / overall_cpu)
        else:
            process_efficiency = 0.0

        # Weighted average
        efficiency = util_efficiency * 0.4 + task_efficiency * 0.3 + process_efficiency * 0.3

        return max(0.0, min(1.0, efficiency))

    async def _analyze_and_optimize(self, profile: CPUProfile) -> None:
        """
        Analyze profile and optimize thread pool size.

        Args:
            profile: Current CPU profile
        """
        if not self.config.enable_dynamic_sizing:
            return

        # Check if adjustment is needed
        adjustment_needed = False
        new_thread_count = self.current_thread_count

        # Under-utilized: increase threads
        if profile.overall_percent < (self.config.target_cpu_utilization - 0.1) * 100 and profile.task_count > 0:
            new_thread_count = min(
                self.current_thread_count + self.config.thread_adjustment_step,
                self.cpu_count * 4,  # Max 4x CPU count
            )
            adjustment_needed = True
            logger.debug(
                f"CPU under-utilized ({profile.overall_percent:.1f}%), increasing threads to {new_thread_count}"
            )

        # Over-utilized: decrease threads
        elif profile.overall_percent > (self.config.target_cpu_utilization + 0.1) * 100:
            new_thread_count = max(
                self.current_thread_count - self.config.thread_adjustment_step,
                self.cpu_count,  # Min 1x CPU count
            )
            adjustment_needed = True
            logger.debug(
                f"CPU over-utilized ({profile.overall_percent:.1f}%), decreasing threads to {new_thread_count}"
            )

        # Low efficiency: adjust based on task/thread ratio
        elif profile.efficiency_score < self.config.min_efficiency_score:
            if profile.task_count < self.current_thread_count / 2:
                # Too many threads for tasks
                new_thread_count = max(profile.task_count * 2, self.cpu_count)
                adjustment_needed = True
                logger.debug(
                    f"Low efficiency ({profile.efficiency_score:.2f}), adjusting threads to {new_thread_count}"
                )

        if adjustment_needed and new_thread_count != self.current_thread_count:
            self.thread_adjustment_history.append((time.time(), self.current_thread_count, new_thread_count))
            self.current_thread_count = new_thread_count
            self.optimal_thread_count = new_thread_count

            # Limit history
            if len(self.thread_adjustment_history) > 50:
                self.thread_adjustment_history = self.thread_adjustment_history[-50:]

    def record_task_start(self, task_id: str) -> None:
        """
        Record task start time.

        Args:
            task_id: Task identifier
        """
        self.task_start_times[task_id] = time.time()

    def record_task_completion(self, task_id: str) -> None:
        """
        Record task completion and duration.

        Args:
            task_id: Task identifier
        """
        if task_id in self.task_start_times:
            duration = time.time() - self.task_start_times[task_id]
            self.task_times.append(duration)
            del self.task_start_times[task_id]

            # Limit history
            if len(self.task_times) > 1000:
                self.task_times = self.task_times[-1000:]

    def get_optimal_thread_count(self) -> int:
        """
        Get current optimal thread count.

        Returns:
            Optimal number of threads
        """
        return self.optimal_thread_count

    def get_recent_profiles(self, count: int = 10) -> List[CPUProfile]:
        """
        Get recent CPU profiles.

        Args:
            count: Number of profiles to return

        Returns:
            List of recent profiles
        """
        return self.profiles[-count:] if self.profiles else []

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
            Dictionary with optimization stats
        """
        avg_cpu = sum(p.overall_percent for p in self.profiles[-10:]) / 10 if len(self.profiles) >= 10 else 0.0

        avg_efficiency = sum(p.efficiency_score for p in self.profiles[-10:]) / 10 if len(self.profiles) >= 10 else 0.0

        return {
            "current_threads": self.current_thread_count,
            "optimal_threads": self.optimal_thread_count,
            "cpu_count": self.cpu_count,
            "avg_cpu_percent": avg_cpu,
            "avg_efficiency": avg_efficiency,
            "profile_count": len(self.profiles),
            "adjustment_count": len(self.thread_adjustment_history),
            "avg_task_time": sum(self.task_times) / len(self.task_times) if self.task_times else 0.0,
        }


class ParallelFFTOptimizer:
    """
    Optimized FFT operations for parallel execution.
    """

    def __init__(self, cpu_count: Optional[int] = None):
        """
        Initialize FFT optimizer.

        Args:
            cpu_count: Number of CPU cores
        """
        self.cpu_count = cpu_count or os.cpu_count() or 4

        # Pre-compute common FFT sizes
        self.common_fft_sizes = [512, 1024, 2048, 4096, 8192]
        self.fft_plans: Dict[int, Any] = {}

        logger.info(f"ParallelFFTOptimizer initialized with {self.cpu_count} cores")

    async def compute_parallel_stft(
        self,
        audio_data: np.ndarray,
        fft_size: int = 2048,
        hop_size: Optional[int] = None,
        window: str = "hann",
        n_parallel: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform in parallel.

        Args:
            audio_data: Audio signal
            fft_size: FFT size
            hop_size: Hop size (default: fft_size // 2)
            window: Window function name
            n_parallel: Number of parallel operations

        Returns:
            STFT magnitude spectrogram
        """
        hop_size = hop_size or fft_size // 2
        n_parallel = n_parallel or self.cpu_count

        # Split audio into chunks for parallel processing
        chunk_size = len(audio_data) // n_parallel
        chunks = []

        for i in range(n_parallel):
            start = i * chunk_size
            # Add overlap for continuity
            end = min((i + 1) * chunk_size + fft_size, len(audio_data))
            chunks.append(audio_data[start:end])

        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._compute_stft_chunk(chunk, fft_size, hop_size, window))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Combine results
        combined = np.concatenate(results, axis=0)
        return combined

    async def _compute_stft_chunk(
        self,
        audio_chunk: np.ndarray,
        fft_size: int,
        hop_size: int,
        window: str,
    ) -> np.ndarray:
        """
        Compute STFT for a single chunk.

        Args:
            audio_chunk: Audio data chunk
            fft_size: FFT size
            hop_size: Hop size
            window: Window function

        Returns:
            STFT magnitude for chunk
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_stft_sync, audio_chunk, fft_size, hop_size, window)

    def _compute_stft_sync(
        self,
        audio_chunk: np.ndarray,
        fft_size: int,
        hop_size: int,
        window: str,
    ) -> np.ndarray:
        """
        Synchronous STFT computation.

        Args:
            audio_chunk: Audio data
            fft_size: FFT size
            hop_size: Hop size
            window: Window function

        Returns:
            STFT magnitude
        """
        # Create window
        if window == "hann":
            window_func = np.hanning(fft_size)
        elif window == "hamming":
            window_func = np.hamming(fft_size)
        elif window == "blackman":
            window_func = np.blackman(fft_size)
        else:
            window_func = np.ones(fft_size)

        # Compute STFT frames
        frames = []
        for i in range(0, len(audio_chunk) - fft_size, hop_size):
            frame = audio_chunk[i : i + fft_size]
            windowed = frame * window_func
            fft_result = np.fft.rfft(windowed)
            frames.append(np.abs(fft_result))

        return np.array(frames) if frames else np.array([[]])

    def optimize_fft_size(self, sample_rate: int, target_resolution_hz: float) -> int:
        """
        Find optimal FFT size for target frequency resolution.

        Args:
            sample_rate: Audio sample rate
            target_resolution_hz: Target frequency resolution

        Returns:
            Optimal FFT size
        """
        # FFT size = sample_rate / frequency_resolution
        ideal_size = int(sample_rate / target_resolution_hz)

        # Round to nearest power of 2 for efficiency
        fft_size = 2 ** int(np.ceil(np.log2(ideal_size)))

        # Ensure it's in reasonable range
        fft_size = max(512, min(16384, fft_size))

        return fft_size

    async def benchmark_fft_performance(self, test_duration_seconds: float = 1.0) -> Dict[str, float]:
        """
        Benchmark FFT performance with different configurations.

        Args:
            test_duration_seconds: Duration of test audio

        Returns:
            Performance metrics
        """
        sample_rate = 44100
        test_audio = np.random.randn(int(sample_rate * test_duration_seconds))
        results = {}

        for fft_size in self.common_fft_sizes:
            start = time.time()
            await self.compute_parallel_stft(test_audio, fft_size=fft_size)
            duration = time.time() - start
            results[f"fft_{fft_size}"] = duration

        return results
