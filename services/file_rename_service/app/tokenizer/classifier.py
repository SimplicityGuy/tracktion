"""
Token classification and categorization logic.
"""

import re

from .models import Token, TokenCategory


class TokenClassifier:
    """Classifies tokens into categories with confidence scoring."""

    def __init__(self) -> None:
        """Initialize the classifier with category rules."""
        self.category_rules = self._initialize_rules()
        self.ambiguous_resolution = self._initialize_ambiguous_rules()

    def _initialize_rules(self) -> dict[TokenCategory, list[tuple[re.Pattern, float]]]:
        """
        Initialize classification rules for each category.

        Returns:
            Dict mapping categories to list of (pattern, confidence) tuples
        """
        return {
            TokenCategory.ARTIST: [
                (
                    re.compile(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$"),
                    0.7,
                ),  # Proper names - lower confidence to avoid venue conflicts
                (re.compile(r"^[A-Z]{2,}$"), 0.7),  # All caps (band abbreviations)
                (re.compile(r"^\w+\s*(and|&)\s*\w+"), 0.8),  # "X and Y" format
                (re.compile(r"^The\s+\w+"), 0.85),  # "The X" format
            ],
            TokenCategory.VENUE: [
                (
                    re.compile(r"\b(Theatre|Theater|Arena|Stadium|Hall|Club|Ballroom|Pavilion|Center|Centre)\b", re.I),
                    0.95,
                ),  # Higher confidence for venue keywords
                (re.compile(r"^(The\s+)?\w+\s+(Room|Stage|Lounge|Bar|Pub)$", re.I), 0.85),
                (re.compile(r"^\w+\'s(\s+\w+)?$", re.I), 0.75),  # "X's" or "X's Y"
                (
                    re.compile(
                        r"\w+\s+(Theatre|Theater|Arena|Stadium|Hall|Club|Ballroom|Pavilion|Center|Centre)", re.I
                    ),
                    0.92,
                ),  # "X Arena" format - high confidence
            ],
            TokenCategory.DATE: [
                (re.compile(r"^\d{4}$"), 0.6),  # Year only
                (re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.I), 0.8),  # Month names
                (re.compile(r"^\d{1,2}(st|nd|rd|th)$", re.I), 0.7),  # Day with suffix
            ],
            TokenCategory.QUALITY: [
                (re.compile(r"\b(HD|HQ|LQ|high|low|good|poor|excellent)\b", re.I), 0.8),
                (re.compile(r"\b(master|remaster|remix|edit)\b", re.I), 0.75),
                (re.compile(r"\b(clean|dirty|raw|processed)\b", re.I), 0.7),
            ],
            TokenCategory.SOURCE: [
                (re.compile(r"\b(bootleg|official|promo|demo)\b", re.I), 0.85),
                (re.compile(r"\b(live|studio|rehearsal|soundcheck)\b", re.I), 0.9),
                (re.compile(r"\b(radio|tv|video|broadcast)\b", re.I), 0.85),
            ],
            TokenCategory.TOUR: [
                (re.compile(r"\b\d{4}\s*tour\b", re.I), 0.95),
                (re.compile(r"\b(world|european|american|asian)\s*tour\b", re.I), 0.9),
                (re.compile(r"\b(spring|summer|fall|autumn|winter)\s*\d{4}\b", re.I), 0.85),
            ],
        }

    def _initialize_ambiguous_rules(self) -> dict[str, TokenCategory]:
        """
        Initialize rules for resolving ambiguous tokens.

        Returns:
            Dict mapping ambiguous values to preferred categories
        """
        return {
            # Common ambiguous terms
            "live": TokenCategory.SOURCE,
            "studio": TokenCategory.SOURCE,
            "demo": TokenCategory.SOURCE,
            "set": TokenCategory.SET,
            "disc": TokenCategory.SET,
            "cd": TokenCategory.SET,
            "track": TokenCategory.TRACK,
            "part": TokenCategory.SET,
            # Months that might be confused with names
            "may": TokenCategory.DATE,
            "june": TokenCategory.DATE,
            "april": TokenCategory.DATE,
            "august": TokenCategory.DATE,
        }

    def classify(self, text: str, context: list[Token] | None = None) -> tuple[TokenCategory, float]:
        """
        Classify a text segment into a category.

        Args:
            text: Text to classify
            context: Optional list of surrounding tokens for context

        Returns:
            Tuple of (category, confidence)
        """
        # Check for exact matches in ambiguous resolution
        text_lower = text.lower()
        if text_lower in self.ambiguous_resolution:
            return self.ambiguous_resolution[text_lower], 0.95

        # Try each category's rules
        best_category = TokenCategory.UNKNOWN
        best_confidence = 0.0

        for category, rules in self.category_rules.items():
            for pattern, confidence in rules:
                if pattern.search(text) and confidence > best_confidence:
                    best_category = category
                    best_confidence = confidence

        # Use context to boost confidence if available
        if context and best_category != TokenCategory.UNKNOWN:
            best_confidence = self._adjust_confidence_with_context(text, best_category, best_confidence, context)

        return best_category, best_confidence

    def _adjust_confidence_with_context(
        self, text: str, category: TokenCategory, confidence: float, context: list[Token]
    ) -> float:
        """
        Adjust confidence based on surrounding context.

        Args:
            text: Text being classified
            category: Proposed category
            confidence: Base confidence
            context: Surrounding tokens

        Returns:
            Adjusted confidence score
        """
        # Look for supporting context
        context_categories = [t.category for t in context]

        # Boost confidence if similar categories are nearby
        if category in context_categories:
            confidence = min(confidence + 0.1, 1.0)

        # Check for specific patterns
        if category == TokenCategory.VENUE:
            # Venues often follow dates
            if TokenCategory.DATE in context_categories:
                confidence = min(confidence + 0.15, 1.0)

        elif category == TokenCategory.ARTIST:
            # Artists often precede dates or venues
            if TokenCategory.DATE in context_categories or TokenCategory.VENUE in context_categories:
                confidence = min(confidence + 0.1, 1.0)

        elif category == TokenCategory.TRACK:
            # Track numbers often precede artist names
            artist_after = any(t.category == TokenCategory.ARTIST for t in context if t.position and t.position > 0)
            if artist_after:
                confidence = min(confidence + 0.1, 1.0)

        return confidence

    def resolve_ambiguity(self, token: Token, candidates: list[tuple[TokenCategory, float]]) -> TokenCategory:
        """
        Resolve ambiguity when multiple categories match.

        Args:
            token: Token to resolve
            candidates: List of (category, confidence) candidates

        Returns:
            Best category for the token
        """
        if not candidates:
            return TokenCategory.UNKNOWN

        # Sort by confidence
        sorted_candidates = sorted(candidates, key=lambda c: c[1], reverse=True)

        # If top confidence is significantly higher, use it
        if len(sorted_candidates) > 1 and sorted_candidates[0][1] - sorted_candidates[1][1] > 0.2:
            return sorted_candidates[0][0]

        # Check ambiguous resolution rules
        token_lower = token.value.lower()
        if token_lower in self.ambiguous_resolution:
            preferred = self.ambiguous_resolution[token_lower]
            # Check if preferred category is among candidates
            for category, _ in candidates:
                if category == preferred:
                    return preferred

        # Default to highest confidence
        best_category, _ = sorted_candidates[0]
        return best_category

    def calculate_category_confidence(self, tokens: list[Token], category: TokenCategory) -> float:
        """
        Calculate overall confidence for tokens in a category.

        Args:
            tokens: List of tokens
            category: Category to calculate confidence for

        Returns:
            Average confidence score
        """
        category_tokens = [t for t in tokens if t.category == category]

        if not category_tokens:
            return 0.0

        total_confidence = sum(t.confidence for t in category_tokens)
        return total_confidence / len(category_tokens)

    def suggest_category(self, token: Token, vocabulary_stats: dict) -> TokenCategory | None:
        """
        Suggest a category for an unknown token based on vocabulary statistics.

        Args:
            token: Token to suggest category for
            vocabulary_stats: Statistics from vocabulary manager

        Returns:
            Suggested category or None
        """
        # This could be enhanced with ML in the future
        # For now, use simple heuristics

        value = token.value.lower()

        # Check if it looks like a number (possible track/year)
        if value.isdigit():
            num = int(value)
            if 1900 <= num <= 2100:
                return TokenCategory.DATE
            elif 1 <= num <= 99:
                return TokenCategory.TRACK

        # Check for common patterns
        if any(char in value for char in ["@", "#", "!"]):
            return None  # Special characters, likely not a valid token

        # If mostly uppercase, might be an abbreviation or format
        if value.isupper() and len(value) <= 5:
            return TokenCategory.FORMAT

        # Default to artist for proper-looking names
        if value[0].isupper() and value[1:].islower():
            return TokenCategory.ARTIST

        return None
