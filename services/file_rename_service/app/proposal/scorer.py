"""Confidence scoring module for rename proposals."""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculate confidence scores for rename proposals based on multiple factors."""

    def __init__(self, base_pattern_weight: float = 0.2, feedback_weight: float = 0.3) -> None:
        """Initialize confidence scorer.

        Args:
            base_pattern_weight: Weight for pattern frequency factor (0.0-1.0)
            feedback_weight: Weight for user feedback factor (0.0-1.0)
        """
        self.base_pattern_weight = max(0.0, min(1.0, base_pattern_weight))
        self.feedback_weight = max(0.0, min(1.0, feedback_weight))

        logger.debug(
            f"ConfidenceScorer initialized with pattern_weight={self.base_pattern_weight}, "
            f"feedback_weight={self.feedback_weight}"
        )

    def calculate_confidence(
        self,
        ml_confidence: float,
        pattern_frequency: dict[str, int],
        pattern_name: str | None = None,
        user_feedback: dict[str, Any] | None = None,
    ) -> float:
        """Calculate overall confidence score combining multiple factors.

        Args:
            ml_confidence: Base confidence from ML model (0.0-1.0)
            pattern_frequency: Dictionary mapping pattern names to usage counts
            pattern_name: Name of the pattern used for this proposal
            user_feedback: Dictionary containing user feedback history

        Returns:
            Final confidence score normalized to 0.0-1.0 range
        """
        # Validate and normalize ML confidence
        if not 0.0 <= ml_confidence <= 1.0:
            logger.warning(f"ML confidence {ml_confidence} outside valid range, clamping to [0,1]")
            ml_confidence = max(0.0, min(1.0, ml_confidence))

        base_score = ml_confidence
        logger.debug(f"Base ML confidence: {base_score:.3f}")

        # Factor in pattern frequency if pattern is specified
        if pattern_name and pattern_frequency:
            pattern_adjusted_score = self.factor_pattern_frequency(base_score, pattern_frequency, pattern_name)
            logger.debug(f"Pattern-adjusted score: {pattern_adjusted_score:.3f}")
        else:
            pattern_adjusted_score = base_score

        # Factor in user feedback history
        if user_feedback:
            final_score = self.factor_user_feedback(pattern_adjusted_score, user_feedback)
            logger.debug(f"Final score with feedback: {final_score:.3f}")
        else:
            final_score = pattern_adjusted_score

        # Ensure final score is in valid range
        final_score = self.normalize_score(final_score)

        logger.info(
            f"Confidence calculated: ML={ml_confidence:.3f} -> Final={final_score:.3f} "
            f"(pattern={pattern_name}, feedback_available={user_feedback is not None})"
        )

        return final_score

    def factor_pattern_frequency(self, base_score: float, pattern_counts: dict[str, int], pattern_name: str) -> float:
        """Adjust confidence based on how frequently a pattern has been used.

        More frequently used patterns get a confidence boost, indicating reliability.

        Args:
            base_score: Base confidence score to adjust
            pattern_counts: Dictionary mapping pattern names to usage counts
            pattern_name: Name of the pattern to factor in

        Returns:
            Adjusted confidence score
        """
        if not pattern_counts or pattern_name not in pattern_counts:
            logger.debug(f"Pattern '{pattern_name}' not found in frequency data, no adjustment")
            return base_score

        pattern_count = pattern_counts[pattern_name]
        total_patterns = sum(pattern_counts.values())

        if total_patterns == 0:
            logger.warning("Total pattern count is zero, no frequency adjustment applied")
            return base_score

        # Calculate frequency ratio (0.0 to 1.0)
        frequency_ratio = pattern_count / total_patterns

        # Use logarithmic scaling to prevent over-boosting popular patterns
        # This gives a more gradual increase based on frequency
        frequency_factor = math.log1p(frequency_ratio * 10) / math.log1p(10)  # Normalize log scale

        # Apply weighted adjustment
        adjustment = self.base_pattern_weight * frequency_factor
        adjusted_score = base_score + (adjustment * (1.0 - base_score))  # Boost toward 1.0

        logger.debug(
            f"Pattern frequency adjustment: {pattern_name} used {pattern_count}/{total_patterns} times "
            f"(ratio={frequency_ratio:.3f}, factor={frequency_factor:.3f}, "
            f"adjustment=+{adjustment:.3f}, {base_score:.3f} -> {adjusted_score:.3f})"
        )

        return adjusted_score

    def factor_user_feedback(self, base_score: float, feedback_history: dict[str, Any]) -> float:
        """Adjust confidence based on historical user feedback.

        Positive feedback increases confidence, negative feedback decreases it.

        Args:
            base_score: Base confidence score to adjust
            feedback_history: Dictionary containing feedback metrics like:
                - 'approval_rate': Float between 0.0-1.0 representing approval percentage
                - 'total_feedback_count': Integer count of total feedback instances
                - 'recent_approvals': Integer count of recent approvals (optional)
                - 'recent_rejections': Integer count of recent rejections (optional)

        Returns:
            Adjusted confidence score
        """
        if not feedback_history:
            logger.debug("No feedback history provided, no adjustment")
            return base_score

        # Extract feedback metrics with defaults
        approval_rate = feedback_history.get("approval_rate", 0.5)  # Default to neutral
        total_count = feedback_history.get("total_feedback_count", 0)

        # Validate approval rate
        approval_rate = max(0.0, min(1.0, approval_rate))

        if total_count == 0:
            logger.debug("No feedback samples available, no adjustment")
            return base_score

        # Calculate confidence in the feedback (more samples = more reliable)
        # Use sigmoid function to gradually increase confidence with sample size
        feedback_reliability = 1.0 / (1.0 + math.exp(-0.1 * (total_count - 20)))  # Sigmoid centered at 20 samples

        # Calculate feedback factor: approval_rate maps to [-1, 1] adjustment
        # 0.5 approval rate = no change, >0.5 = positive, <0.5 = negative
        feedback_factor = (approval_rate - 0.5) * 2.0  # Maps [0,1] to [-1,1]

        # Apply weighted adjustment based on reliability
        reliable_feedback_factor = feedback_factor * feedback_reliability
        adjustment = self.feedback_weight * reliable_feedback_factor

        if adjustment > 0:
            # Boost confidence toward 1.0
            adjusted_score: float = base_score + (adjustment * (1.0 - base_score))
        else:
            # Reduce confidence toward 0.0
            adjusted_score = base_score + (adjustment * base_score)

        logger.debug(
            f"User feedback adjustment: approval_rate={approval_rate:.3f}, "
            f"total_count={total_count}, reliability={feedback_reliability:.3f}, "
            f"factor={feedback_factor:.3f}, adjustment={adjustment:+.3f}, "
            f"{base_score:.3f} -> {adjusted_score:.3f}"
        )

        return adjusted_score

    def normalize_score(self, raw_score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a raw score to specified range.

        Args:
            raw_score: Raw score value to normalize
            min_val: Minimum value of target range (default: 0.0)
            max_val: Maximum value of target range (default: 1.0)

        Returns:
            Normalized score clamped to [min_val, max_val]
        """
        if min_val >= max_val:
            logger.error(f"Invalid normalization range: min_val={min_val} >= max_val={max_val}")
            raise ValueError("min_val must be less than max_val")

        normalized = max(min_val, min(max_val, raw_score))

        if normalized != raw_score:
            logger.debug(f"Score normalized: {raw_score:.6f} -> {normalized:.6f}")

        return normalized

    def calculate_batch_confidence(
        self,
        ml_confidences: list[float],
        pattern_frequency: dict[str, int],
        pattern_names: list[str | None],
        user_feedbacks: list[dict[str, Any] | None] | None = None,
    ) -> list[float]:
        """Calculate confidence scores for multiple proposals efficiently.

        Args:
            ml_confidences: List of ML confidence scores
            pattern_frequency: Dictionary mapping pattern names to usage counts
            pattern_names: List of pattern names corresponding to each proposal
            user_feedbacks: Optional list of user feedback dictionaries

        Returns:
            List of final confidence scores
        """
        if len(ml_confidences) != len(pattern_names):
            raise ValueError("ml_confidences and pattern_names must have same length")

        if user_feedbacks and len(user_feedbacks) != len(ml_confidences):
            raise ValueError("user_feedbacks length must match ml_confidences length")

        results = []

        for i, ml_conf in enumerate(ml_confidences):
            pattern_name = pattern_names[i]
            feedback = user_feedbacks[i] if user_feedbacks else None

            confidence = self.calculate_confidence(ml_conf, pattern_frequency, pattern_name, feedback)
            results.append(confidence)

        logger.info(f"Calculated batch confidence for {len(results)} proposals")
        return results

    def get_scoring_weights(self) -> dict[str, float]:
        """Get current scoring weights configuration.

        Returns:
            Dictionary containing current weight settings
        """
        return {
            "pattern_frequency_weight": self.base_pattern_weight,
            "user_feedback_weight": self.feedback_weight,
            "ml_confidence_weight": 1.0 - self.base_pattern_weight - self.feedback_weight,
        }

    def update_weights(self, pattern_weight: float | None = None, feedback_weight: float | None = None) -> None:
        """Update scoring weights.

        Args:
            pattern_weight: New weight for pattern frequency factor (0.0-1.0)
            feedback_weight: New weight for user feedback factor (0.0-1.0)
        """
        if pattern_weight is not None:
            self.base_pattern_weight = max(0.0, min(1.0, pattern_weight))
            logger.info(f"Updated pattern frequency weight to {self.base_pattern_weight}")

        if feedback_weight is not None:
            self.feedback_weight = max(0.0, min(1.0, feedback_weight))
            logger.info(f"Updated user feedback weight to {self.feedback_weight}")
