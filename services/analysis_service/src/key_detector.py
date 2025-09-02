"""
Musical key detection module using Essentia.

Implements key detection with multiple algorithms for validation
and confidence scoring.
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import essentia.standard as es
import numpy as np

from services.analysis_service.src.exceptions import (
    AnalysisServiceError,
    CorruptedFileError,
    InvalidAudioFileError,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)


@dataclass
class KeyDetectionResult:
    """Results from key detection analysis."""

    key: str  # e.g., "C", "F#", "Bb"
    scale: str  # "major" or "minor"
    confidence: float  # 0.0 to 1.0
    alternative_key: str | None = None
    alternative_scale: str | None = None
    agreement: bool = False
    needs_review: bool = False


class KeyDetector:
    """
    Musical key detection using Essentia algorithms.

    Provides primary detection with KeyExtractor and validation
    using HPCP-based alternative approach.
    """

    # Map Essentia key names to standard notation
    KEY_NAMES: ClassVar[list[str]] = [
        "C",
        "C#",
        "D",
        "Eb",
        "E",
        "F",
        "F#",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        agreement_boost: float = 1.2,
        disagreement_penalty: float = 0.8,
        needs_review_threshold: float = 0.7,
    ):
        """
        Initialize the key detector.

        Args:
            confidence_threshold: Minimum confidence for reliable detection
            agreement_boost: Multiplier when algorithms agree (max 1.0)
            disagreement_penalty: Multiplier when algorithms disagree
            needs_review_threshold: Threshold below which manual review is suggested
        """
        self.confidence_threshold = confidence_threshold
        self.agreement_boost = agreement_boost
        self.disagreement_penalty = disagreement_penalty
        self.needs_review_threshold = needs_review_threshold

    def detect_key(self, audio_file: str) -> KeyDetectionResult | None:
        """
        Detect the musical key of an audio file.

        Args:
            audio_file: Path to the audio file

        Returns:
            KeyDetectionResult with detected key, scale, and confidence
        """
        try:
            # Load audio file
            logger.info(f"Loading audio file: {audio_file}")
            audio = es.MonoLoader(filename=audio_file)()

            # Primary detection using KeyExtractor
            primary_result = self._detect_with_key_extractor(audio, es)

            # Validation with alternative HPCP-based algorithm
            alternative_result = self._detect_with_hpcp(audio, es)

            # Combine results and calculate final confidence
            final_result = self._combine_results(primary_result, alternative_result)

            logger.info(
                f"Key detection complete: {final_result.key} {final_result.scale} "
                f"(confidence: {final_result.confidence:.2f})"
            )

            return final_result

        except FileNotFoundError as e:
            logger.error(f"Audio file not found: {audio_file} - {e!s}")
            raise InvalidAudioFileError(f"Audio file not found: {audio_file}") from e
        except PermissionError as e:
            logger.error(f"Permission denied accessing audio file: {audio_file} - {e!s}")
            raise InvalidAudioFileError(f"Permission denied: {audio_file}") from e
        except (OSError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "corrupt" in error_msg or "invalid" in error_msg or "unsupported" in error_msg:
                logger.error(f"Corrupted or unsupported audio file: {audio_file} - {e!s}")
                if "format" in error_msg or "codec" in error_msg:
                    raise UnsupportedFormatError(f"Unsupported audio format: {audio_file}") from e
                raise CorruptedFileError(f"Corrupted audio file: {audio_file}") from e
            logger.error(f"Audio processing error for {audio_file}: {e!s}")
            raise InvalidAudioFileError(f"Audio processing failed: {audio_file}") from e
        except MemoryError as e:
            logger.error(f"Out of memory during key detection for {audio_file}: {e!s}")
            raise AnalysisServiceError(f"Out of memory analyzing {audio_file}: File too large") from e
        except Exception as e:
            logger.error(f"Unexpected error detecting key for {audio_file}: {e!s}", exc_info=True)
            raise AnalysisServiceError(f"Key detection failed for {audio_file}: {e!s}") from e

    def _detect_with_key_extractor(self, audio: np.ndarray, es: Any) -> tuple[str, str, float]:
        """
        Primary key detection using Essentia's KeyExtractor.

        Args:
            audio: Audio signal array
            es: Essentia standard module

        Returns:
            Tuple of (key, scale, strength)
        """
        try:
            # Use KeyExtractor for primary detection
            key, scale, strength = es.KeyExtractor()(audio)

            logger.debug(f"KeyExtractor result: {key} {scale} (strength: {strength:.3f})")

            return key, scale, strength

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "essentia" in error_msg or "algorithm" in error_msg:
                logger.error(f"KeyExtractor algorithm error: {e!s}")
                # Try to recover with simplified parameters or fallback
                try:
                    # Retry with simplified KeyExtractor configuration
                    logger.info("Retrying KeyExtractor with fallback configuration")
                    key_extractor_simple = es.KeyExtractor(profileType="temperley")
                    key_simple, scale_simple, strength_simple = key_extractor_simple(audio)
                    logger.debug(
                        f"KeyExtractor fallback result: {key_simple} {scale_simple} (strength: {strength_simple:.3f})"
                    )
                    return key_simple, scale_simple, strength_simple
                except Exception as fallback_error:
                    logger.error(f"KeyExtractor fallback also failed: {fallback_error!s}")
                    return "C", "major", 0.0
            else:
                logger.error(f"KeyExtractor runtime error: {e!s}")
                return "C", "major", 0.0
        except MemoryError as e:
            logger.error(f"Out of memory in KeyExtractor: {e!s}")
            # Try with reduced audio length for memory-constrained systems
            try:
                logger.info("Retrying KeyExtractor with reduced audio length")
                # Use first 60 seconds to reduce memory usage
                reduced_audio = audio[: min(len(audio), 60 * 22050)]  # 60s at 22kHz
                key_extractor = es.KeyExtractor()
                key_reduced, scale_reduced, strength_reduced = key_extractor(reduced_audio)
                logger.debug(
                    f"KeyExtractor reduced result: {key_reduced} {scale_reduced} (strength: {strength_reduced:.3f})"
                )
                return key_reduced, scale_reduced, strength_reduced * 0.8  # Penalize confidence for reduced analysis
            except Exception as memory_fallback_error:
                logger.error(f"KeyExtractor memory fallback failed: {memory_fallback_error!s}")
                return "C", "major", 0.0
        except Exception as e:
            logger.error(f"Unexpected KeyExtractor error: {e!s}", exc_info=True)
            return "C", "major", 0.0

    def _detect_with_hpcp(self, audio: np.ndarray, es: Any) -> tuple[str, str, float]:
        """
        Alternative key detection using HPCP (Harmonic Pitch Class Profile).

        Args:
            audio: Audio signal array
            es: Essentia standard module

        Returns:
            Tuple of (key, scale, strength)
        """
        try:
            # Compute spectrum
            spectrum = es.Spectrum()(audio)

            # Compute HPCP (chromagram)
            hpcp = es.HPCP()(spectrum)

            # Detect key from HPCP
            key, scale, strength, _ = es.Key()(hpcp)

            logger.debug(f"HPCP Key result: {key} {scale} (strength: {strength:.3f})")

            return key, scale, strength

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "essentia" in error_msg or "algorithm" in error_msg:
                logger.error(f"HPCP algorithm error: {e!s}")
                # Try alternative HPCP configuration
                try:
                    logger.info("Retrying HPCP with alternative configuration")
                    # Use simpler HPCP configuration
                    hpcp_simple = es.HPCP(size=12, windowSize=1.0)
                    spectrum = es.Spectrum()
                    peaks = es.SpectralPeaks()

                    # Simplified processing
                    hpcp_values = []
                    for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=2048):
                        spec = spectrum(frame)
                        freqs, mags = peaks(spec)
                        if len(freqs) > 0:
                            hpcp_frame = hpcp_simple(freqs, mags)
                            hpcp_values.append(hpcp_frame)

                    if hpcp_values:
                        hpcp_mean = np.mean(hpcp_values, axis=0)
                        # Use a simple template matching approach
                        major_template = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # C major template
                        correlation = np.correlate(hpcp_mean, major_template, mode="full")
                        best_key_idx = np.argmax(correlation) % 12
                        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                        return keys[best_key_idx], "major", 0.3  # Low confidence for fallback method
                    logger.warning("No HPCP values generated in fallback")
                    return "C", "major", 0.0
                except Exception as fallback_error:
                    logger.error(f"HPCP fallback failed: {fallback_error!s}")
                    return "C", "major", 0.0
            else:
                logger.error(f"HPCP runtime error: {e!s}")
                return "C", "major", 0.0
        except MemoryError as e:
            logger.error(f"Out of memory in HPCP key detection: {e!s}")
            # Return low confidence result to indicate reduced reliability
            return "C", "major", 0.0
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid data in HPCP processing: {e!s}")
            # Data processing error - likely due to audio characteristics
            return "C", "major", 0.0
        except Exception as e:
            logger.error(f"Unexpected HPCP error: {e!s}", exc_info=True)
            return "C", "major", 0.0

    def _combine_results(
        self,
        primary: tuple[str, str, float],
        alternative: tuple[str, str, float],
    ) -> KeyDetectionResult:
        """
        Combine results from multiple algorithms.

        Args:
            primary: Results from KeyExtractor (key, scale, strength)
            alternative: Results from HPCP method (key, scale, strength)

        Returns:
            Combined KeyDetectionResult with confidence scoring
        """
        key_primary, scale_primary, strength_primary = primary
        key_alt, scale_alt, strength_alt = alternative

        # Check agreement between algorithms
        agreement = (key_primary == key_alt) and (scale_primary == scale_alt)

        # Calculate confidence based on agreement
        if agreement:
            # Boost confidence when algorithms agree
            confidence = min(strength_primary * self.agreement_boost, 1.0)
            logger.debug(f"Algorithms agree, boosted confidence: {confidence:.3f}")
        else:
            # Penalize confidence when algorithms disagree
            # Use the stronger detection as primary
            if strength_primary >= strength_alt:
                confidence = strength_primary * self.disagreement_penalty
            else:
                # Switch to alternative if it's stronger
                key_primary, scale_primary = key_alt, scale_alt
                confidence = strength_alt * self.disagreement_penalty

            logger.debug(f"Algorithms disagree, reduced confidence: {confidence:.3f}")

        # Determine if manual review is needed
        needs_review = confidence < self.needs_review_threshold

        return KeyDetectionResult(
            key=key_primary,
            scale=scale_primary,
            confidence=confidence,
            alternative_key=key_alt if not agreement else None,
            alternative_scale=scale_alt if not agreement else None,
            agreement=agreement,
            needs_review=needs_review,
        )

    def detect_key_with_segments(self, audio_file: str, num_segments: int = 3) -> KeyDetectionResult | None:
        """
        Detect key using multiple segments for improved accuracy.

        Args:
            audio_file: Path to the audio file
            num_segments: Number of segments to analyze

        Returns:
            KeyDetectionResult based on segment analysis
        """
        try:
            # Load audio file
            audio = es.MonoLoader(filename=audio_file)()
            segment_length = len(audio) // num_segments

            segment_results = []

            # Analyze each segment
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length if i < num_segments - 1 else len(audio)
                segment = audio[start:end]

                # Get results for this segment
                primary = self._detect_with_key_extractor(segment, es)
                alternative = self._detect_with_hpcp(segment, es)
                result = self._combine_results(primary, alternative)

                segment_results.append(result)

            # Vote on the most common key
            return self._vote_on_segments(segment_results)

        except Exception as e:
            logger.error(f"Segment-based key detection failed: {e!s}")
            return None

    def _vote_on_segments(self, results: list[KeyDetectionResult]) -> KeyDetectionResult:
        """
        Vote on key detection from multiple segments.

        Args:
            results: List of KeyDetectionResult from segments

        Returns:
            Final KeyDetectionResult based on voting
        """
        # Count key/scale combinations weighted by confidence
        key_votes: dict[tuple[str, str], float] = {}

        for result in results:
            key_scale = (result.key, result.scale)
            if key_scale not in key_votes:
                key_votes[key_scale] = 0
            key_votes[key_scale] += result.confidence

        # Find the winning key/scale combination
        winner = max(key_votes.items(), key=lambda x: x[1])
        (key, scale), total_confidence = winner

        # Calculate average confidence for the winning key
        winning_results = [r for r in results if r.key == key and r.scale == scale]
        avg_confidence = sum(r.confidence for r in winning_results) / len(winning_results)

        # Check if there's strong agreement
        agreement_ratio = len(winning_results) / len(results)
        if agreement_ratio > 0.66:  # More than 2/3 segments agree
            avg_confidence = min(avg_confidence * self.agreement_boost, 1.0)

        return KeyDetectionResult(
            key=key,
            scale=scale,
            confidence=avg_confidence,
            agreement=agreement_ratio > 0.66,
            needs_review=avg_confidence < self.needs_review_threshold,
        )
