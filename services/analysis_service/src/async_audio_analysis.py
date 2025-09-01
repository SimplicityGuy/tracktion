"""
Async wrappers for audio analysis functions.

This module provides async interfaces for BPM detection, key detection,
mood analysis, and other audio processing operations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]  # types-aiofiles not installed in this environment
import essentia.standard as es
import numpy as np

from services.analysis_service.src.async_audio_processor import (
    AsyncAudioProcessor,
    TaskPriority,
)
from services.analysis_service.src.bpm_detector import BPMDetector
from services.analysis_service.src.key_detector import KeyDetector
from services.analysis_service.src.mood_analyzer import MoodAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisResult:
    """Complete audio analysis results."""

    file_path: str
    bpm: dict[str, Any] | None = None
    key: dict[str, Any] | None = None
    mood: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    errors: dict[str, str] | None = None
    processing_time_ms: float = 0.0


class AsyncAudioAnalyzer:
    """
    Async audio analysis coordinator.

    Manages parallel execution of various audio analysis tasks using
    the AsyncAudioProcessor infrastructure.
    """

    def __init__(
        self,
        processor: AsyncAudioProcessor | None = None,
        bpm_detector: BPMDetector | None = None,
        key_detector: KeyDetector | None = None,
        mood_analyzer: MoodAnalyzer | None = None,
        enable_buffering: bool = True,
        buffer_size: int = 8192,
    ):
        """
        Initialize the async audio analyzer.

        Args:
            processor: AsyncAudioProcessor instance
            bpm_detector: BPMDetector instance
            key_detector: KeyDetector instance
            mood_analyzer: MoodAnalyzer instance
            enable_buffering: Enable buffered audio reading
            buffer_size: Buffer size for audio reading
        """
        self.processor = processor or AsyncAudioProcessor()
        self.bpm_detector = bpm_detector or BPMDetector()
        self.key_detector = key_detector or KeyDetector()
        self.mood_analyzer = mood_analyzer or MoodAnalyzer()
        self.enable_buffering = enable_buffering
        self.buffer_size = buffer_size

        logger.info("AsyncAudioAnalyzer initialized")

    async def analyze_bpm_async(self, audio_file: str, priority: TaskPriority = TaskPriority.NORMAL) -> dict[str, Any]:
        """
        Detect BPM asynchronously.

        Args:
            audio_file: Path to audio file
            priority: Task priority

        Returns:
            BPM detection results
        """
        logger.debug(f"Starting async BPM detection for {audio_file}")

        try:
            result = await self.processor.process_audio_async(
                audio_file,
                self.bpm_detector.detect_bpm,
                priority=priority,
                task_id=f"bpm_{Path(audio_file).stem}",
            )
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Async BPM detection failed for {audio_file}: {e!s}")
            raise

    async def analyze_key_async(
        self, audio_file: str, priority: TaskPriority = TaskPriority.NORMAL
    ) -> dict[str, Any] | None:
        """
        Detect musical key asynchronously.

        Args:
            audio_file: Path to audio file
            priority: Task priority

        Returns:
            Key detection results
        """
        logger.debug(f"Starting async key detection for {audio_file}")

        try:
            # Convert KeyDetectionResult to dict for serialization
            def detect_and_convert(file_path: str) -> dict[str, Any] | None:
                result = self.key_detector.detect_key(file_path)
                if result:
                    return {
                        "key": result.key,
                        "scale": result.scale,
                        "confidence": result.confidence,
                        "alternative_key": result.alternative_key,
                        "alternative_scale": result.alternative_scale,
                        "agreement": result.agreement,
                        "needs_review": result.needs_review,
                    }
                return None

            result = await self.processor.process_audio_async(
                audio_file,
                detect_and_convert,
                priority=priority,
                task_id=f"key_{Path(audio_file).stem}",
            )
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Async key detection failed for {audio_file}: {e!s}")
            raise

    async def analyze_mood_async(
        self, audio_file: str, priority: TaskPriority = TaskPriority.NORMAL
    ) -> dict[str, Any] | None:
        """
        Analyze mood and genre asynchronously.

        Args:
            audio_file: Path to audio file
            priority: Task priority

        Returns:
            Mood analysis results
        """
        logger.debug(f"Starting async mood analysis for {audio_file}")

        try:
            # Convert MoodAnalysisResult to dict for serialization
            def analyze_and_convert(file_path: str) -> dict[str, Any] | None:
                result = self.mood_analyzer.analyze_mood(file_path)
                if result:
                    return {
                        "mood_scores": result.mood_scores,
                        "genres": result.genres,
                        "primary_genre": result.primary_genre,
                        "genre_confidence": result.genre_confidence,
                        "danceability": result.danceability,
                        "energy": result.energy,
                        "valence": result.valence,
                        "arousal": result.arousal,
                        "voice_instrumental": result.voice_instrumental,
                        "voice_confidence": result.voice_confidence,
                        "overall_confidence": result.overall_confidence,
                        "needs_review": result.needs_review,
                    }
                return None

            result = await self.processor.process_audio_async(
                audio_file,
                analyze_and_convert,
                priority=priority,
                task_id=f"mood_{Path(audio_file).stem}",
            )
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Async mood analysis failed for {audio_file}: {e!s}")
            raise

    async def analyze_audio_complete(
        self,
        audio_file: str,
        enable_bpm: bool = True,
        enable_key: bool = True,
        enable_mood: bool = True,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> AudioAnalysisResult:
        """
        Perform complete audio analysis with all components in parallel.

        Args:
            audio_file: Path to audio file
            enable_bpm: Enable BPM detection
            enable_key: Enable key detection
            enable_mood: Enable mood analysis
            priority: Task priority

        Returns:
            Complete AudioAnalysisResult
        """
        start_time = time.time()
        result = AudioAnalysisResult(file_path=audio_file)

        # Create tasks for parallel execution
        tasks = self._create_analysis_tasks(audio_file, enable_bpm, enable_key, enable_mood, priority)

        # Execute and process results
        errors = await self._execute_and_process_tasks(tasks, result)

        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Add errors if any
        if errors:
            result.errors = errors

        logger.info(f"Complete analysis for {audio_file} finished in {result.processing_time_ms:.1f}ms")

        return result

    def _create_analysis_tasks(
        self,
        audio_file: str,
        enable_bpm: bool,
        enable_key: bool,
        enable_mood: bool,
        priority: TaskPriority,
    ) -> list[tuple[str, asyncio.Task[dict[str, Any] | None]]]:
        """Create analysis tasks based on enabled components."""
        tasks: list[tuple[str, asyncio.Task[dict[str, Any] | None]]] = []

        if enable_bpm:
            tasks.append(("bpm", asyncio.create_task(self.analyze_bpm_async(audio_file, priority))))

        if enable_key:
            tasks.append(("key", asyncio.create_task(self.analyze_key_async(audio_file, priority))))

        if enable_mood:
            tasks.append(("mood", asyncio.create_task(self.analyze_mood_async(audio_file, priority))))

        return tasks

    async def _execute_and_process_tasks(
        self,
        tasks: list[tuple[str, asyncio.Task[dict[str, Any] | None]]],
        result: AudioAnalysisResult,
    ) -> dict[str, str]:
        """Execute tasks and process their results."""
        errors: dict[str, str] = {}

        if not tasks:
            return errors

        # Gather results
        task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results
        for i, (task_name, _) in enumerate(tasks):
            task_result = task_results[i]
            if isinstance(task_result, Exception):
                errors[task_name] = str(task_result)
                logger.error(f"{task_name} analysis failed: {task_result}")
            elif task_name == "bpm":
                result.bpm = task_result  # type: ignore[assignment]
            elif task_name == "key":
                result.key = task_result  # type: ignore[assignment]
            elif task_name == "mood":
                result.mood = task_result  # type: ignore[assignment]

        return errors

    async def analyze_batch(
        self,
        audio_files: list[str],
        enable_bpm: bool = True,
        enable_key: bool = True,
        enable_mood: bool = True,
        max_concurrent: int | None = None,
    ) -> dict[str, AudioAnalysisResult]:
        """
        Analyze multiple audio files in parallel.

        Args:
            audio_files: List of audio file paths
            enable_bpm: Enable BPM detection
            enable_key: Enable key detection
            enable_mood: Enable mood analysis
            max_concurrent: Maximum concurrent analyses

        Returns:
            Dictionary mapping file paths to results
        """
        max_concurrent = max_concurrent or self.processor.cpu_count * 2
        results = {}

        # Process in batches
        for i in range(0, len(audio_files), max_concurrent):
            batch = audio_files[i : i + max_concurrent]

            # Create tasks for this batch
            batch_tasks = []
            for audio_file in batch:
                task = self.analyze_audio_complete(
                    audio_file,
                    enable_bpm=enable_bpm,
                    enable_key=enable_key,
                    enable_mood=enable_mood,
                )
                batch_tasks.append((audio_file, task))

            # Wait for batch completion
            for audio_file, task in batch_tasks:
                try:
                    result = await task
                    results[audio_file] = result
                except Exception as e:
                    logger.error(f"Batch analysis failed for {audio_file}: {e!s}")
                    results[audio_file] = AudioAnalysisResult(file_path=audio_file, errors={"general": str(e)})

        return results

    async def read_audio_buffered(self, audio_file: str) -> np.ndarray:
        """
        Read audio file with async buffering.

        Args:
            audio_file: Path to audio file

        Returns:
            Audio signal as numpy array
        """
        if not self.enable_buffering:
            # Fall back to synchronous loading in thread pool
            def load_audio() -> np.ndarray:
                loader = es.MonoLoader(filename=audio_file)
                audio_data = loader()
                return np.array(audio_data, dtype=np.float32)  # Explicit numpy array conversion

            result = await self.processor._run_in_executor(load_audio)
            return np.array(result, dtype=np.float32)  # Ensure proper numpy array type

        # Async buffered reading implementation
        # This is a simplified version - in production, you'd want
        # to stream chunks asynchronously

        # Read file asynchronously
        async with aiofiles.open(audio_file, "rb") as f:
            # Read in chunks for large files
            chunks = []
            while True:
                chunk = await f.read(self.buffer_size)
                if not chunk:
                    break
                chunks.append(chunk)

        # Process audio data
        audio_data = b"".join(chunks)

        # Decode audio in thread pool
        def decode_audio(data: bytes) -> np.ndarray:
            # This would need a proper implementation to decode from bytes
            # For now, falling back to file path loading
            loader = es.MonoLoader(filename=audio_file)
            audio_data = loader()
            return np.array(audio_data, dtype=np.float32)  # Explicit numpy array conversion

        result = await self.processor._run_in_executor(decode_audio, audio_data)
        return np.array(result, dtype=np.float32)  # Ensure proper numpy array type


class AsyncFFTProcessor:
    """
    Async FFT processing for parallel spectral analysis.
    """

    def __init__(self, processor: AsyncAudioProcessor):
        """
        Initialize FFT processor.

        Args:
            processor: AsyncAudioProcessor instance
        """
        self.processor = processor

    async def compute_fft_async(
        self,
        audio_data: np.ndarray,
        fft_size: int = 2048,
        hop_size: int | None = None,
        window_type: str = "hann",
    ) -> np.ndarray:
        """
        Compute FFT asynchronously for parallel execution.

        Args:
            audio_data: Audio signal array
            fft_size: FFT size
            hop_size: Hop size for STFT
            window_type: Window function type

        Returns:
            FFT magnitude spectrum
        """
        hop_size = hop_size or fft_size // 2

        def compute_stft(data: np.ndarray) -> np.ndarray:
            """Compute Short-Time Fourier Transform."""
            try:
                # Create windowing function
                windowing = es.Windowing(type=window_type, size=fft_size)

                # Create FFT
                fft = es.FFT(size=fft_size)

                # Process frames
                frames = []
                for i in range(0, len(data) - fft_size, hop_size):
                    frame = data[i : i + fft_size]
                    windowed_data = windowing(frame)
                    windowed_array = np.array(windowed_data, dtype=np.complex64)  # Explicit conversion for mypy
                    spectrum_data = fft(windowed_array)
                    spectrum_array = np.array(spectrum_data, dtype=np.complex64)  # Explicit conversion for mypy
                    frames.append(np.abs(spectrum_array[: fft_size // 2 + 1]))

                return np.array(frames, dtype=np.float32)
            except ImportError:
                # Fallback to numpy FFT if essentia not available
                frames = []
                for i in range(0, len(data) - fft_size, hop_size):
                    frame = data[i : i + fft_size]
                    # Apply Hann window manually
                    window = np.hanning(fft_size)
                    windowed = frame * window
                    spectrum = np.fft.fft(windowed)
                    frames.append(np.abs(spectrum[: fft_size // 2 + 1]))

                return np.array(frames, dtype=np.float32)

        # Run FFT computation in thread pool
        result = await self.processor._run_in_executor(compute_stft, audio_data)
        return np.array(result, dtype=np.float32)  # Ensure proper numpy array type

    async def compute_parallel_ffts(self, audio_segments: list[np.ndarray], fft_size: int = 2048) -> list[np.ndarray]:
        """
        Compute FFTs for multiple audio segments in parallel.

        Args:
            audio_segments: List of audio segments
            fft_size: FFT size

        Returns:
            List of FFT results
        """
        tasks = []
        for segment in audio_segments:
            task = self.compute_fft_async(segment, fft_size)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return list(results)
