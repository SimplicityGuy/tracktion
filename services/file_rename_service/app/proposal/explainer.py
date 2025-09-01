"""
Human-readable explanation system for rename proposals.

This module generates clear, user-friendly explanations for rename proposals,
showing which patterns matched, confidence factors, and reasoning for alternatives.
"""

import logging
from typing import Any

from services.file_rename_service.app.tokenizer.models import TokenCategory
from services.file_rename_service.app.tokenizer.patterns import PatternMatcher

from .models import RenameProposal

logger = logging.getLogger(__name__)


class RenameExplainer:
    """Generates human-readable explanations for rename proposals."""

    def __init__(self, pattern_matcher: PatternMatcher | None = None) -> None:
        """Initialize the explainer.

        Args:
            pattern_matcher: Optional PatternMatcher instance for pattern details.
                           If not provided, a new instance will be created.
        """
        self.pattern_matcher = pattern_matcher or PatternMatcher()

        # Category descriptions for user-friendly explanations
        self.category_descriptions = {
            TokenCategory.ARTIST: "artist/band name",
            TokenCategory.DATE: "date information",
            TokenCategory.VENUE: "venue or location",
            TokenCategory.QUALITY: "audio quality indicator",
            TokenCategory.FORMAT: "file format",
            TokenCategory.SOURCE: "recording source",
            TokenCategory.TRACK: "track information",
            TokenCategory.SET: "set or disc number",
            TokenCategory.TOUR: "tour information",
            TokenCategory.LABEL: "record label",
            TokenCategory.CATALOG: "catalog number",
            TokenCategory.UNKNOWN: "unidentified element",
        }

    def generate_explanation(self, proposal: RenameProposal, ml_predictions: dict[str, Any] | None = None) -> str:
        """Generate a comprehensive human-readable explanation for a rename proposal.

        Args:
            proposal: The RenameProposal to explain
            ml_predictions: Optional ML model predictions and metadata

        Returns:
            Formatted explanation string

        Example:
            >>> proposal = RenameProposal(
            ...     original_filename="phish_1999-07-04_boston.flac",
            ...     proposed_filename="Phish - 1999-07-04 - Boston - FLAC.flac",
            ...     confidence_score=0.85,
            ...     patterns_used=["artist_pattern", "date_iso", "venue_city", "format_audio"],
            ...     alternatives=["Phish - July 4 1999 - Boston.flac", "1999-07-04 Phish Boston.flac"]
            ... )
            >>> explainer = RenameExplainer()
            >>> print(explainer.generate_explanation(proposal))
            Rename Analysis for 'phish_1999-07-04_boston.flac'

            Proposed rename: Phish - 1999-07-04 - Boston - FLAC.flac
            Confidence: 85% (High confidence)

            Pattern Matches Found:
            • Artist: 'phish' → Recognized as artist/band name
            • Date: '1999-07-04' → ISO date format (YYYY-MM-DD)
            • Venue: 'boston' → Location/venue identifier
            • Format: 'flac' → Audio format indicator

            Confidence Factors:
            • Strong pattern recognition (4/4 key elements identified)
            • High-confidence date format match
            • Known artist name detected
            • Standard audio format recognized

            Alternative Options:
            1. Phish - July 4 1999 - Boston.flac (Different date format)
            2. 1999-07-04 Phish Boston.flac (Artist-first ordering)
        """
        components = {
            "header": self._generate_header(proposal),
            "confidence": self._generate_confidence_summary(proposal),
            "patterns": self._generate_pattern_explanation(proposal),
            "confidence_factors": self._generate_confidence_factors(proposal, ml_predictions),
            "alternatives": self._generate_alternatives_explanation(proposal),
        }

        return self.format_explanation(components)

    def explain_pattern_match(self, pattern_name: str, matched_tokens: list[str]) -> str:
        """Explain how a specific pattern matched and what it means.

        Args:
            pattern_name: Name/ID of the pattern that matched
            matched_tokens: List of text tokens that were matched by this pattern

        Returns:
            Human-readable explanation of the pattern match

        Example:
            >>> explainer = RenameExplainer()
            >>> explainer.explain_pattern_match("date_iso", ["1999-07-04"])
            "Date: '1999-07-04' → ISO date format (YYYY-MM-DD) - Clear date structure"
        """
        if not matched_tokens:
            return f"Pattern '{pattern_name}': No matches found"

        # Find the pattern in our pattern matcher
        pattern_details = self._get_pattern_details(pattern_name)

        if not pattern_details:
            return f"Pattern '{pattern_name}': {', '.join(matched_tokens)} (Unknown pattern type)"

        category = pattern_details.get("category", "unknown")
        description = pattern_details.get("description", "Pattern match")

        # Get user-friendly category name
        category_name = self.category_descriptions.get(category, str(category))

        # Format the explanation based on number of matches
        if len(matched_tokens) == 1:
            return f"{category_name.title()}: '{matched_tokens[0]}' → {description}"
        tokens_str = "', '".join(matched_tokens)
        return f"{category_name.title()}: '{tokens_str}' → {description}"

    def explain_confidence_factors(self, confidence_score: float, contributing_factors: dict[str, Any]) -> str:
        """Explain what factors contributed to the confidence score.

        Args:
            confidence_score: The overall confidence score (0.0-1.0)
            contributing_factors: Dictionary of factors that influenced the score

        Returns:
            Human-readable explanation of confidence factors

        Example:
            >>> factors = {
            ...     "pattern_matches": 4,
            ...     "ml_confidence": 0.82,
            ...     "pattern_frequency_boost": 0.05,
            ...     "user_feedback_adjustment": -0.02
            ... }
            >>> explainer = RenameExplainer()
            >>> explainer.explain_confidence_factors(0.85, factors)
            "High confidence (85%) based on:
            • Strong pattern recognition (4 key elements identified)
            • ML model prediction: 82%
            • Pattern frequency boost: +5% (reliable patterns)
            • User feedback adjustment: -2% (mixed historical feedback)"
        """
        confidence_level = self._get_confidence_level_description(confidence_score)

        explanation_parts = [f"{confidence_level} ({confidence_score:.0%}) based on:"]

        # Analyze each contributing factor
        if "pattern_matches" in contributing_factors:
            count = contributing_factors["pattern_matches"]
            explanation_parts.append(f"• Pattern recognition ({count} key elements identified)")

        if "ml_confidence" in contributing_factors:
            ml_score = contributing_factors["ml_confidence"]
            explanation_parts.append(f"• ML model prediction: {ml_score:.0%}")

        if "pattern_frequency_boost" in contributing_factors:
            boost = contributing_factors["pattern_frequency_boost"]
            if boost > 0:
                explanation_parts.append(f"• Pattern reliability bonus: +{boost:.1%}")
            elif boost < 0:
                explanation_parts.append(f"• Pattern reliability penalty: {boost:.1%}")

        if "user_feedback_adjustment" in contributing_factors:
            adjustment = contributing_factors["user_feedback_adjustment"]
            if adjustment > 0:
                explanation_parts.append(f"• User feedback bonus: +{adjustment:.1%} (positive history)")
            elif adjustment < 0:
                explanation_parts.append(f"• User feedback penalty: {adjustment:.1%} (mixed history)")

        if "coverage_ratio" in contributing_factors:
            coverage = contributing_factors["coverage_ratio"]
            if coverage > 0.8:
                explanation_parts.append(f"• High filename coverage ({coverage:.0%} recognized)")
            elif coverage > 0.6:
                explanation_parts.append(f"• Good filename coverage ({coverage:.0%} recognized)")
            else:
                explanation_parts.append(f"• Partial filename coverage ({coverage:.0%} recognized)")

        return "\n".join(explanation_parts)

    def explain_alternatives(self, alternatives: list[str], primary_proposal: str) -> list[str]:
        """Generate explanations for alternative rename proposals.

        Args:
            alternatives: List of alternative filename proposals
            primary_proposal: The main/preferred proposal for comparison

        Returns:
            List of explanations for each alternative

        Example:
            >>> alternatives = [
            ...     "Phish - July 4 1999 - Boston.flac",
            ...     "1999-07-04 Phish Boston.flac"
            ... ]
            >>> primary = "Phish - 1999-07-04 - Boston - FLAC.flac"
            >>> explainer = RenameExplainer()
            >>> explainer.explain_alternatives(alternatives, primary)
            [
                "Phish - July 4 1999 - Boston.flac (Human-readable date format)",
                "1999-07-04 Phish Boston.flac (Date-first chronological ordering)"
            ]
        """
        if not alternatives:
            return ["No alternative options available"]

        explanations = []

        for _i, alternative in enumerate(alternatives, 1):
            # Analyze differences between alternative and primary proposal
            differences = self._analyze_proposal_differences(primary_proposal, alternative)

            explanation = f"{alternative} ({differences})" if differences else f"{alternative} (Alternative formatting)"

            explanations.append(explanation)

        return explanations

    def format_explanation(self, components: dict[str, str]) -> str:
        """Format explanation components into a cohesive, readable explanation.

        Args:
            components: Dictionary containing different explanation sections

        Returns:
            Formatted complete explanation
        """
        sections = []

        # Add header
        if components.get("header"):
            sections.append(components["header"])

        # Add confidence summary
        if components.get("confidence"):
            sections.append(components["confidence"])

        # Add pattern matches
        if components.get("patterns"):
            sections.append("Pattern Matches Found:")
            sections.append(components["patterns"])

        # Add confidence factors
        if components.get("confidence_factors"):
            sections.append("Confidence Analysis:")
            sections.append(components["confidence_factors"])

        # Add alternatives
        if components.get("alternatives"):
            sections.append("Alternative Options:")
            sections.append(components["alternatives"])

        return "\n\n".join(sections)

    def _generate_header(self, proposal: RenameProposal) -> str:
        """Generate the explanation header."""
        return f"Rename Analysis for '{proposal.original_filename}'\n" + "=" * (len(proposal.original_filename) + 20)

    def _generate_confidence_summary(self, proposal: RenameProposal) -> str:
        """Generate confidence summary section."""
        confidence_level = self._get_confidence_level_description(proposal.confidence_score)
        return (
            f"Proposed rename: {proposal.proposed_filename}\n"
            f"Confidence: {proposal.confidence_score:.0%} ({confidence_level})"
        )

    def _generate_pattern_explanation(self, proposal: RenameProposal) -> str:
        """Generate pattern matches explanation."""
        if not proposal.patterns_used:
            return "• No specific patterns identified"

        explanations = []
        for pattern_name in proposal.patterns_used:
            # For this example, we'll create reasonable explanations
            # In practice, you'd want to match against actual pattern data
            pattern_explanation = self._create_pattern_explanation(pattern_name)
            explanations.append(f"• {pattern_explanation}")

        return "\n".join(explanations)

    def _generate_confidence_factors(self, proposal: RenameProposal, ml_predictions: dict[str, Any] | None) -> str:
        """Generate confidence factors explanation."""
        factors = []

        # Pattern-based factors
        if proposal.patterns_used:
            pattern_count = len(proposal.patterns_used)
            if pattern_count >= 3:
                factors.append(f"• Strong pattern recognition ({pattern_count} key elements identified)")
            elif pattern_count >= 2:
                factors.append(f"• Good pattern recognition ({pattern_count} elements identified)")
            else:
                factors.append(f"• Basic pattern recognition ({pattern_count} element identified)")

        # ML prediction factors
        if ml_predictions and "confidence" in ml_predictions:
            ml_conf = ml_predictions["confidence"]
            factors.append(f"• ML model prediction: {ml_conf:.0%}")

        # Confidence level assessment
        if proposal.confidence_score > 0.8:
            factors.append("• High pattern match reliability")
        elif proposal.confidence_score > 0.6:
            factors.append("• Moderate pattern match reliability")
        else:
            factors.append("• Lower pattern match reliability")

        return "\n".join(factors) if factors else "• Standard confidence assessment applied"

    def _generate_alternatives_explanation(self, proposal: RenameProposal) -> str:
        """Generate alternatives explanation."""
        if not proposal.alternatives:
            return "No alternative options available"

        explanations = self.explain_alternatives(proposal.alternatives, proposal.proposed_filename)

        formatted_alternatives = []
        for i, explanation in enumerate(explanations, 1):
            formatted_alternatives.append(f"{i}. {explanation}")

        return "\n".join(formatted_alternatives)

    def _get_confidence_level_description(self, confidence_score: float) -> str:
        """Get human-readable confidence level description."""
        if confidence_score >= 0.9:
            return "Very high confidence"
        if confidence_score >= 0.8:
            return "High confidence"
        if confidence_score >= 0.7:
            return "Good confidence"
        if confidence_score >= 0.6:
            return "Moderate confidence"
        if confidence_score >= 0.5:
            return "Low confidence"
        return "Very low confidence"

    def _get_pattern_details(self, pattern_name: str) -> dict[str, Any] | None:
        """Get pattern details from the pattern matcher."""
        # Look through patterns to find matching name/description
        for pattern in self.pattern_matcher.patterns:
            if pattern.description and (
                pattern_name.lower() in pattern.description.lower() or pattern_name in pattern.regex
            ):
                return {
                    "category": pattern.category,
                    "description": pattern.description,
                    "priority": pattern.priority,
                    "examples": pattern.examples,
                }
        return None

    def _create_pattern_explanation(self, pattern_name: str) -> str:
        """Create explanation for a pattern name."""
        # Map common pattern names to explanations
        pattern_explanations = {
            "artist_pattern": "Artist name recognized",
            "date_iso": "ISO date format detected",
            "date_us": "US date format detected",
            "venue_city": "Location/venue identified",
            "venue_at": "Venue with 'at' keyword found",
            "format_audio": "Audio format indicator found",
            "quality_bitrate": "Audio quality/bitrate detected",
            "source_type": "Recording source identified",
            "track_number": "Track numbering detected",
            "set_indicator": "Set/disc information found",
            "catalog_number": "Catalog/label number found",
        }

        if pattern_name in pattern_explanations:
            return pattern_explanations[pattern_name]

        # Try to extract category from pattern name
        for category in TokenCategory:
            if category.value in pattern_name.lower():
                desc = self.category_descriptions.get(category, "pattern match")
                return f"{desc.title()} pattern matched"

        return f"Pattern '{pattern_name}' matched"

    def _analyze_proposal_differences(self, primary: str, alternative: str) -> str:
        """Analyze differences between primary and alternative proposals."""
        # Simple difference analysis - in practice you might want more sophisticated comparison

        if "july" in alternative.lower() and "07" in primary:
            return "Human-readable month format"
        if alternative.startswith(("19", "20")):
            return "Date-first chronological ordering"
        if len(alternative) < len(primary):
            return "Shorter format"
        if len(alternative) > len(primary):
            return "More detailed format"
        if "-" in primary and "_" in alternative:
            return "Underscore separators"
        if "_" in primary and "-" in alternative:
            return "Dash separators"
        return "Alternative structure"
