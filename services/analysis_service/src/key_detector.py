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
        Primary key detection using Essentia's KeyExtractor algorithm.

        This method uses Essentia's KeyExtractor which implements key detection
        based on harmonic pitch class profiles and template matching. The algorithm:

        1. Analyzes the harmonic content of the audio signal
        2. Creates a pitch class profile (chroma vector)
        3. Matches against major/minor key templates
        4. Returns the best matching key with confidence score

        Algorithm details:
        - Uses Krumhansl-Schmuckler key-finding algorithm
        - Default profile type: 'bgate' (optimized for electronic music)
        - Fallback profile type: 'temperley' (classical music oriented)
        - Handles memory constraints by analyzing first 60 seconds if needed

        Error handling:
        - Runtime errors: Tries fallback configuration with 'temperley' profile
        - Memory errors: Reduces audio length to 60 seconds with confidence penalty
        - All errors: Returns default 'C major' with zero confidence

        Args:
            audio: Audio signal array (mono, typically 44.1kHz)
            es: Essentia standard module reference

        Returns:
            Tuple of (key, scale, strength) where:
            - key: Musical key (e.g., 'C', 'F#', 'Bb')
            - scale: Either 'major' or 'minor'
            - strength: Algorithm confidence (0.0-1.0, higher is better)
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
        Alternative key detection using HPCP (Harmonic Pitch Class Profile) analysis.

        This method provides an independent key detection approach using direct
        HPCP computation and template matching for validation of primary results.

        Algorithm workflow:
        1. Compute spectrum from audio signal
        2. Extract HPCP (chromagram) representing pitch class distribution
        3. Apply Key() algorithm for template matching
        4. Fallback to manual template correlation if main algorithm fails

        HPCP (Harmonic Pitch Class Profile):
        - Represents the relative intensity of each of the 12 pitch classes
        - Aggregates harmonic content across all octaves
        - More robust to octave differences than raw pitch detection
        - Used for chord recognition and key analysis

        Fallback algorithm (when main HPCP fails):
        - Uses frame-by-frame spectral peak analysis
        - Computes HPCP manually with reduced configuration
        - Applies simple correlation with major scale template
        - Returns low confidence (0.3) to indicate reduced reliability

        Error handling strategy:
        - RuntimeError: Attempts simplified HPCP configuration with manual processing
        - MemoryError: Returns default result to prevent system instability
        - ValueError/IndexError: Handles invalid audio characteristics gracefully
        - All failures: Returns 'C major' with zero confidence

        Args:
            audio: Audio signal array (mono, float32 format expected)
            es: Essentia standard module reference for algorithm access

        Returns:
            Tuple of (key, scale, strength) where:
            - key: Detected musical key using standard notation
            - scale: Either 'major' or 'minor' scale type
            - strength: Confidence score (0.0-1.0, lower for fallback methods)
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

                    # Simplified processing: manually process audio frames
                    hpcp_values = []
                    # Process audio in frames: 4096 samples (~93ms at 44kHz) with 50% overlap
                    for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=2048):
                        spec = spectrum(frame)  # Convert frame to frequency domain
                        freqs, mags = peaks(spec)  # Find spectral peaks for pitch detection
                        if len(freqs) > 0:
                            # Compute HPCP for this frame (harmonic pitch class profile)
                            hpcp_frame = hpcp_simple(freqs, mags)
                            hpcp_values.append(hpcp_frame)

                    if hpcp_values:
                        # Average HPCP across all frames to get overall pitch class distribution
                        hpcp_mean = np.mean(hpcp_values, axis=0)
                        # Use template matching approach with C major scale pattern
                        major_template = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # C major: C-D-E-F-G-A-B
                        # Cross-correlate HPCP with major template to find best key
                        correlation = np.correlate(hpcp_mean, major_template, mode="full")
                        # Find index of maximum correlation (best matching key)
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
        Combine and validate results from multiple key detection algorithms.

        This method implements a sophisticated algorithm consensus system that:
        1. Compares results from KeyExtractor (primary) and HPCP (alternative)
        2. Applies confidence adjustments based on algorithm agreement
        3. Selects the most reliable result when algorithms disagree
        4. Determines if manual review is recommended

        Confidence adjustment strategy:
        - Agreement: Boost confidence by agreement_boost factor (default 1.2x, max 1.0)
        - Disagreement: Apply disagreement_penalty (default 0.8x) to selected result
        - Algorithm switching: Use alternative if it has higher strength when disagreeing

        Agreement detection:
        - Exact match required for both key and scale
        - No tolerance for enharmonic equivalents (F# ≠ Gb)
        - Helps identify ambiguous or difficult-to-analyze audio

        Selection when disagreeing:
        - Compare raw strength values from both algorithms
        - Choose algorithm with higher confidence for final result
        - Apply disagreement penalty to account for uncertainty
        - Preserve alternative result for manual review reference

        Manual review criteria:
        - Final confidence below needs_review_threshold (default 0.7)
        - Helps identify tracks requiring human verification
        - Common for: complex harmonic content, atonal music, modulating keys

        Args:
            primary: KeyExtractor results (key, scale, strength)
            alternative: HPCP method results (key, scale, strength)

        Returns:
            KeyDetectionResult containing:
            - key/scale: Selected primary result
            - confidence: Adjusted confidence score (0.0-1.0)
            - alternative_key/scale: Alternative result if disagreement occurs
            - agreement: Boolean indicating algorithm consensus
            - needs_review: Boolean suggesting manual verification if confidence low
        """
        key_primary, scale_primary, strength_primary = primary
        key_alt, scale_alt, strength_alt = alternative

        # Check agreement between algorithms
        agreement = (key_primary == key_alt) and (scale_primary == scale_alt)

        # Calculate confidence based on agreement between algorithms
        if agreement:
            # Both algorithms detected same key+scale: boost confidence
            # Apply agreement_boost multiplier (default 1.2x) but cap at 1.0
            confidence = min(strength_primary * self.agreement_boost, 1.0)
            logger.debug(f"Algorithms agree, boosted confidence: {confidence:.3f}")
        else:
            # Algorithms disagree: apply penalty and choose stronger result
            # Use the stronger detection as primary result for output
            if strength_primary >= strength_alt:
                # Primary algorithm is more confident, keep its result
                confidence = strength_primary * self.disagreement_penalty  # Default 0.8x penalty
            else:
                # Alternative algorithm is more confident, switch to its result
                key_primary, scale_primary = key_alt, scale_alt
                confidence = strength_alt * self.disagreement_penalty  # Apply same penalty

            logger.debug(f"Algorithms disagree, reduced confidence: {confidence:.3f}")

        # Determine if manual review is needed
        # TODO: Implement machine learning model to improve confidence threshold
        # Could learn from manual review feedback to optimize needs_review_threshold
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
