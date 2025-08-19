"""
Temporal BPM analysis module for tracking tempo changes over time.

This module provides windowed BPM analysis to detect tempo variations
throughout a track, useful for DJ mixes and variable tempo music.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import essentia.standard as es
import numpy as np

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyzes BPM variations over time using windowed analysis.

    Detects tempo changes, calculates stability scores, and provides
    detailed temporal BPM information for tracks.
    """

    def __init__(
        self,
        window_size: float = 10.0,
        hop_size: Optional[float] = None,
        sample_rate: int = 44100,
        start_duration: float = 30.0,
        end_duration: float = 30.0,
    ):
        """
        Initialize temporal analyzer with configuration.

        Args:
            window_size: Size of analysis window in seconds
            hop_size: Hop between windows in seconds (default: window_size / 2)
            sample_rate: Sample rate for audio processing
            start_duration: Duration to analyze for start BPM (seconds)
            end_duration: Duration to analyze for end BPM (seconds)
        """
        self.window_size = window_size
        self.hop_size = hop_size or window_size / 2
        self.sample_rate = sample_rate
        self.start_duration = start_duration
        self.end_duration = end_duration

        # Initialize rhythm extractor for windowed analysis
        self.rhythm_extractor = es.RhythmExtractor2013()

        logger.info(f"TemporalAnalyzer initialized with window_size={window_size}s, hop_size={self.hop_size}s")

    def analyze_temporal_bpm(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform temporal BPM analysis on an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing:
                - average_bpm: Overall average BPM
                - start_bpm: BPM for first N seconds
                - end_bpm: BPM for last N seconds
                - stability_score: Tempo stability (0-1, higher is more stable)
                - temporal_bpm: List of BPM values over time windows
                - tempo_changes: Detected significant tempo changes
                - is_variable_tempo: Whether track has variable tempo

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If analysis fails
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio
            logger.debug(f"Loading audio for temporal analysis: {audio_path}")
            loader = es.MonoLoader(filename=str(audio_path), sampleRate=self.sample_rate)
            audio = loader()

            if len(audio) == 0:
                raise RuntimeError("Loaded audio is empty")

            audio_duration = len(audio) / self.sample_rate

            # Perform windowed BPM analysis
            temporal_data = self._analyze_windows(audio, audio_duration)

            # Calculate start and end BPM
            start_bpm = self._calculate_start_bpm(audio)
            end_bpm = self._calculate_end_bpm(audio)

            # Calculate average BPM and stability
            bpm_values = [w["bpm"] for w in temporal_data if w["confidence"] > 0.5]

            if not bpm_values:
                # No confident BPM values found
                return {
                    "average_bpm": None,
                    "start_bpm": None,
                    "end_bpm": None,
                    "stability_score": 0.0,
                    "temporal_bpm": temporal_data,
                    "tempo_changes": [],
                    "is_variable_tempo": False,
                }

            average_bpm = np.mean(bpm_values)
            stability_score = self._calculate_stability(bpm_values)

            # Detect tempo changes
            tempo_changes = self._detect_tempo_changes(temporal_data)
            is_variable = len(tempo_changes) > 0 or stability_score < 0.7

            result = {
                "average_bpm": round(float(average_bpm), 1),
                "start_bpm": round(float(start_bpm), 1) if start_bpm else None,
                "end_bpm": round(float(end_bpm), 1) if end_bpm else None,
                "stability_score": round(float(stability_score), 3),
                "temporal_bpm": temporal_data,
                "tempo_changes": tempo_changes,
                "is_variable_tempo": is_variable,
            }

            logger.info(
                f"Temporal analysis complete: avg={result['average_bpm']} BPM, stability={result['stability_score']}"
            )

            return result

        except Exception as e:
            logger.error(f"Temporal analysis failed for {audio_path}: {str(e)}")
            raise RuntimeError(f"Temporal analysis failed: {str(e)}") from e

    def _analyze_windows(self, audio: np.ndarray, duration: float) -> List[Dict[str, Any]]:
        """
        Analyze BPM in overlapping windows across the track.

        Args:
            audio: Audio signal array
            duration: Total duration in seconds

        Returns:
            List of window analysis results
        """
        temporal_data = []
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_size * self.sample_rate)

        # Calculate number of windows
        num_windows = int((len(audio) - window_samples) / hop_samples) + 1

        for i in range(num_windows):
            start_sample = i * hop_samples
            end_sample = min(start_sample + window_samples, len(audio))

            # Skip if window is too small
            if end_sample - start_sample < self.sample_rate * 2:  # At least 2 seconds
                continue

            window_audio = audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            try:
                # Analyze window
                bpm, beats, confidence, _, _ = self.rhythm_extractor(window_audio)

                temporal_data.append(
                    {
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "bpm": round(float(bpm), 1),
                        "confidence": round(float(confidence), 3),
                        "num_beats": len(beats),
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to analyze window {i}: {str(e)}")
                temporal_data.append(
                    {
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "bpm": None,  # type: ignore[dict-item]
                        "confidence": 0.0,
                        "num_beats": 0,
                    }
                )

        return temporal_data

    def _calculate_start_bpm(self, audio: np.ndarray) -> Optional[float]:
        """
        Calculate BPM for the start of the track.

        Args:
            audio: Audio signal array

        Returns:
            BPM value or None if detection fails
        """
        start_samples = int(self.start_duration * self.sample_rate)
        start_audio = audio[: min(start_samples, len(audio))]

        if len(start_audio) < self.sample_rate * 5:  # Need at least 5 seconds
            return None

        try:
            bpm, _, confidence, _, _ = self.rhythm_extractor(start_audio)
            if confidence > 0.5:
                return float(bpm)
        except Exception as e:
            logger.warning(f"Failed to calculate start BPM: {str(e)}")

        return None

    def _calculate_end_bpm(self, audio: np.ndarray) -> Optional[float]:
        """
        Calculate BPM for the end of the track.

        Args:
            audio: Audio signal array

        Returns:
            BPM value or None if detection fails
        """
        end_samples = int(self.end_duration * self.sample_rate)
        end_audio = audio[-min(end_samples, len(audio)) :]

        if len(end_audio) < self.sample_rate * 5:  # Need at least 5 seconds
            return None

        try:
            bpm, _, confidence, _, _ = self.rhythm_extractor(end_audio)
            if confidence > 0.5:
                return float(bpm)
        except Exception as e:
            logger.warning(f"Failed to calculate end BPM: {str(e)}")

        return None

    def _calculate_stability(self, bpm_values: List[float]) -> float:
        """
        Calculate tempo stability score.

        Args:
            bpm_values: List of BPM values over time

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(bpm_values) < 2:
            return 1.0

        # Calculate coefficient of variation
        mean_bpm = np.mean(bpm_values)
        if mean_bpm == 0:
            return 0.0

        std_bpm = np.std(bpm_values)
        cv = std_bpm / mean_bpm

        # Convert CV to stability score (0-1)
        # CV of 0 = perfect stability (1.0)
        # CV of 0.2 or higher = low stability (0.0)
        stability = max(0.0, min(1.0, 1.0 - (cv * 5)))

        return float(stability)

    def _detect_tempo_changes(
        self, temporal_data: List[Dict[str, Any]], threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Detect significant tempo changes in temporal data.

        Args:
            temporal_data: List of windowed BPM analysis results
            threshold: Minimum BPM difference to consider a change

        Returns:
            List of detected tempo changes with timestamps
        """
        tempo_changes: List[Dict[str, Any]] = []

        # Filter to confident detections only
        confident_windows = [w for w in temporal_data if w["bpm"] is not None and w["confidence"] > 0.5]

        if len(confident_windows) < 2:
            return tempo_changes

        for i in range(1, len(confident_windows)):
            prev_bpm = confident_windows[i - 1]["bpm"]
            curr_bpm = confident_windows[i]["bpm"]

            if abs(curr_bpm - prev_bpm) >= threshold:
                tempo_changes.append(
                    {
                        "timestamp": confident_windows[i]["start_time"],
                        "from_bpm": prev_bpm,
                        "to_bpm": curr_bpm,
                        "change": round(curr_bpm - prev_bpm, 1),
                    }
                )

        return tempo_changes
