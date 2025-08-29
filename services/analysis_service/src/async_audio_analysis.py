"""
Async wrappers for audio analysis functions.

This module provides async interfaces for BPM detection, key detection,
mood analysis, and other audio processing operations.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    bpm: Optional[Dict[str, Any]] = None
    key: Optional[Dict[str, Any]] = None
    mood: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    errors: Optional[Dict[str, str]] = None
    processing_time_ms: float = 0.0


class AsyncAudioAnalyzer:
    """
    Async audio analysis coordinator.

    Manages parallel execution of various audio analysis tasks using
    the AsyncAudioProcessor infrastructure.
    """

    def __init__(
        self,
        processor: Optional[AsyncAudioProcessor] = None,
        bpm_detector: Optional[BPMDetector] = None,
        key_detector: Optional[KeyDetector] = None,
        mood_analyzer: Optional[MoodAnalyzer] = None,
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

    async def analyze_bpm_async(self, audio_file: str, priority: TaskPriority = TaskPriority.NORMAL) -> Dict[str, Any]:
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
            return result
        except Exception as e:
            logger.error(f"Async BPM detection failed for {audio_file}: {str(e)}")
            raise

    async def analyze_key_async(
        self, audio_file: str, priority: TaskPriority = TaskPriority.NORMAL
    ) -> Optional[Dict[str, Any]]:
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
            def detect_and_convert(file_path: str) -> Optional[Dict[str, Any]]:
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
            return result
        except Exception as e:
            logger.error(f"Async key detection failed for {audio_file}: {str(e)}")
            raise

    async def analyze_mood_async(
        self, audio_file: str, priority: TaskPriority = TaskPriority.NORMAL
    ) -> Optional[Dict[str, Any]]:
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
            def analyze_and_convert(file_path: str) -> Optional[Dict[str, Any]]:
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
            return result
        except Exception as e:
            logger.error(f"Async mood analysis failed for {audio_file}: {str(e)}")
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
        import time

        start_time = time.time()
        result = AudioAnalysisResult(file_path=audio_file)
        errors = {}

        # Create tasks for parallel execution
        tasks = []

        if enable_bpm:
            tasks.append(("bpm", self.analyze_bpm_async(audio_file, priority)))

        if enable_key:
            tasks.append(("key", self.analyze_key_async(audio_file, priority)))

        if enable_mood:
            tasks.append(("mood", self.analyze_mood_async(audio_file, priority)))

        # Execute all tasks in parallel
        if tasks:
            # Gather results
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            # Process results
            for i, (task_name, _) in enumerate(tasks):
                task_result = task_results[i]
                if isinstance(task_result, Exception):
                    errors[task_name] = str(task_result)
                    logger.error(f"{task_name} analysis failed: {task_result}")
                else:
                    if task_name == "bpm":
                        result.bpm = task_result
                    elif task_name == "key":
                        result.key = task_result
                    elif task_name == "mood":
                        result.mood = task_result

        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Add errors if any
        if errors:
            result.errors = errors

        logger.info(f"Complete analysis for {audio_file} finished in {result.processing_time_ms:.1f}ms")

        return result

    async def analyze_batch(
        self,
        audio_files: list[str],
        enable_bpm: bool = True,
        enable_key: bool = True,
        enable_mood: bool = True,
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, AudioAnalysisResult]:
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
                    logger.error(f"Batch analysis failed for {audio_file}: {str(e)}")
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
            import essentia.standard as es

            def load_audio() -> np.ndarray:
                loader = es.MonoLoader(filename=audio_file)
                return loader()

            return await self.processor._run_in_executor(load_audio)

        # Async buffered reading implementation
        # This is a simplified version - in production, you'd want
        # to stream chunks asynchronously
        import aiofiles
        import essentia.standard as es

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
            return loader()

        return await self.processor._run_in_executor(decode_audio, audio_data)


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
        hop_size: Optional[int] = None,
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
                import essentia.standard as es

                # Create windowing function
                windowing = es.Windowing(type=window_type, size=fft_size)

                # Create FFT
                fft = es.FFT(size=fft_size)

                # Process frames
                frames = []
                for i in range(0, len(data) - fft_size, hop_size):
                    frame = data[i : i + fft_size]
                    windowed = windowing(frame)
                    spectrum = fft(windowed)
                    frames.append(np.abs(spectrum[: fft_size // 2 + 1]))

                return np.array(frames)
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

                return np.array(frames)

        # Run FFT computation in thread pool
        result = await self.processor._run_in_executor(compute_stft, audio_data)
        return result

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
