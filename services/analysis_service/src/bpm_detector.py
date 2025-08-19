"""
BPM detection module using Essentia audio analysis library.

This module provides BPM (beats per minute) detection capabilities with
confidence scoring and fallback algorithms for improved accuracy.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import essentia.standard as es
import numpy as np

logger = logging.getLogger(__name__)


class BPMDetector:
    """
    BPM detector using Essentia's rhythm extraction algorithms.

    Primary algorithm: RhythmExtractor2013
    Fallback algorithm: PercivalBpmEstimator
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        confidence_threshold: float = 0.7,
        agreement_tolerance: float = 5.0,
        sample_rate: int = 44100,
    ) -> None:
        """
        Initialize BPM detector with configuration.

        Args:
            config: BPMConfig object (preferred) or None to use individual parameters
            confidence_threshold: Minimum confidence for primary algorithm (used if config is None)
            agreement_tolerance: BPM difference tolerance for algorithm agreement (used if config is None)
            sample_rate: Sample rate for audio loading (used if config is None)
        """
        if config is not None:
            # Use config object
            self.confidence_threshold = config.confidence_threshold
            self.agreement_tolerance = config.agreement_tolerance
            # BPMConfig doesn't have sample_rate, use default
            self.sample_rate = sample_rate
        else:
            # Use individual parameters
            self.confidence_threshold = confidence_threshold
            self.agreement_tolerance = agreement_tolerance
            self.sample_rate = sample_rate

        # Initialize extractors
        self.rhythm_extractor = es.RhythmExtractor2013()
        self.percival_estimator = es.PercivalBpmEstimator()

        logger.info(
            f"BPMDetector initialized with confidence_threshold={self.confidence_threshold}, "
            f"agreement_tolerance={self.agreement_tolerance}"
        )

    def detect_bpm(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect BPM from an audio file with confidence scoring.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing:
                - bpm: Detected BPM (float)
                - confidence: Confidence score (0-1)
                - beats: Beat positions in seconds
                - algorithm: Algorithm used (primary/fallback/consensus)
                - needs_review: Whether manual review is recommended

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio loading or processing fails
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio as mono
            logger.debug(f"Loading audio file: {audio_path}")
            loader = es.MonoLoader(filename=str(audio_path), sampleRate=self.sample_rate)
            audio = loader()

            if len(audio) == 0:
                raise RuntimeError("Loaded audio is empty")

            # Primary BPM detection using RhythmExtractor2013
            bpm, beats, confidence, _, beat_intervals = self._extract_rhythm(audio)

            # Normalize confidence to 0-1 range if needed
            # Essentia's RhythmExtractor2013 can return confidence > 1
            if confidence > 1.0:
                confidence = min(1.0, confidence / 5.0)  # Rough normalization based on observed values

            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

            # Determine if we need fallback
            algorithm_used = "primary"
            needs_review = False

            if confidence < self.confidence_threshold:
                logger.debug(f"Low confidence ({confidence:.2f}) from primary algorithm, trying fallback")

                # Try fallback algorithm
                bpm_fallback = self._estimate_bpm_percival(audio)

                # Check agreement between algorithms
                if abs(bpm - bpm_fallback) < self.agreement_tolerance:
                    # Algorithms agree, boost confidence
                    confidence = min(0.9, confidence + 0.3)
                    algorithm_used = "consensus"
                    logger.info(f"Algorithms agree: primary={bpm:.1f}, fallback={bpm_fallback:.1f}")
                else:
                    # Algorithms disagree, use the one with more stable tempo
                    if self._is_tempo_stable(beat_intervals):
                        # Primary is stable, keep it
                        needs_review = True
                    else:
                        # Use fallback
                        bpm = bpm_fallback
                        algorithm_used = "fallback"
                        needs_review = True
                        logger.warning(f"Algorithms disagree: primary={bpm:.1f}, fallback={bpm_fallback:.1f}")

            # Flag for review if confidence still low
            if confidence < 0.8:
                needs_review = True

            result = {
                "bpm": round(float(bpm), 1),
                "confidence": float(confidence),
                "beats": beats.tolist() if isinstance(beats, np.ndarray) else beats,
                "algorithm": algorithm_used,
                "needs_review": needs_review,
            }

            logger.info(f"BPM detection complete: {result['bpm']} BPM (confidence={result['confidence']:.2f})")

            return result

        except Exception as e:
            logger.error(f"BPM detection failed for {audio_path}: {str(e)}")
            raise RuntimeError(f"BPM detection failed: {str(e)}") from e

    def _extract_rhythm(self, audio: np.ndarray) -> Tuple[Any, ...]:
        """
        Extract rhythm using RhythmExtractor2013.

        Args:
            audio: Audio signal array

        Returns:
            Tuple of (bpm, beats, confidence, estimates, intervals)
        """
        return self.rhythm_extractor(audio)  # type: ignore[no-any-return]

    def _estimate_bpm_percival(self, audio: np.ndarray) -> Any:
        """
        Estimate BPM using Percival's algorithm as fallback.

        Args:
            audio: Audio signal array

        Returns:
            Estimated BPM
        """
        return self.percival_estimator(audio)

    def _is_tempo_stable(self, beat_intervals: np.ndarray, threshold: float = 0.15) -> bool:
        """
        Check if tempo is stable based on beat interval variance.

        Args:
            beat_intervals: Array of beat intervals
            threshold: Maximum coefficient of variation for stable tempo

        Returns:
            True if tempo is stable
        """
        if len(beat_intervals) < 2:
            return False

        # Calculate coefficient of variation
        mean_interval = np.mean(beat_intervals)
        if mean_interval == 0:
            return False

        std_interval = np.std(beat_intervals)
        cv = std_interval / mean_interval

        return bool(cv < threshold)  # Ensure we return a Python bool, not numpy bool

    def detect_bpm_with_confidence(self, audio_path: str) -> Dict[str, Any]:
        """
        Production-ready BPM detection matching Story 2.2 research pattern.

        This method provides a simplified interface matching the research
        recommendations from Story 2.2.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary with bpm, confidence, and needs_review
        """
        result = self.detect_bpm(audio_path)

        # Return simplified format matching Story 2.2 pattern
        return {"bpm": result["bpm"], "confidence": result["confidence"], "needs_review": result["needs_review"]}
