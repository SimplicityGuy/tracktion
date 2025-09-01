"""
Vocabulary management for dynamic token discovery and tracking.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .models import Token, TokenCategory


class VocabularyManager:
    """Manages the dynamic vocabulary of discovered tokens."""

    def __init__(self, vocabulary_path: Path | None = None):
        """
        Initialize the vocabulary manager.

        Args:
            vocabulary_path: Path to persist vocabulary data
        """
        self.vocabulary_path = vocabulary_path
        self.vocabulary: dict[str, dict[TokenCategory, Token]] = defaultdict(dict)
        self.frequency_index: dict[TokenCategory, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.discovery_threshold = 3  # Min occurrences to consider a token "discovered"

        if vocabulary_path and vocabulary_path.exists():
            self.load_vocabulary()

    def add_token(self, token: Token) -> None:
        """Add or update a token in the vocabulary."""
        key = token.value.lower()

        if token.category in self.vocabulary[key]:
            # Update existing token
            existing = self.vocabulary[key][token.category]
            existing.frequency += 1
            existing.last_seen = datetime.now(tz=UTC)
            existing.confidence = max(existing.confidence, token.confidence)
        else:
            # Add new token (but store with lowercase value for consistency)
            token_copy = Token(
                value=key,  # Store as lowercase
                category=token.category,
                frequency=token.frequency,
                confidence=token.confidence,
                first_seen=token.first_seen,
                last_seen=token.last_seen,
                position=token.position,
                original_text=token.original_text,
            )
            self.vocabulary[key][token.category] = token_copy

        # Update frequency index
        self.frequency_index[token.category][key] = self.vocabulary[key][token.category].frequency

    def add_tokens(self, tokens: list[Token]) -> None:
        """Add multiple tokens to the vocabulary."""
        for token in tokens:
            self.add_token(token)

    def get_token(self, value: str, category: TokenCategory | None = None) -> Token | None:
        """
        Retrieve a token from the vocabulary.

        Args:
            value: Token value to look up
            category: Optional category filter

        Returns:
            Token if found, None otherwise
        """
        key = value.lower()

        if key not in self.vocabulary:
            return None

        if category:
            return self.vocabulary[key].get(category)

        # Return the token with highest frequency across all categories
        tokens = list(self.vocabulary[key].values())
        return max(tokens, key=lambda t: t.frequency) if tokens else None

    def discover_new_tokens(self, tokens: list[Token]) -> list[Token]:
        """
        Identify potentially new tokens based on frequency patterns.

        Returns:
            List of newly discovered tokens
        """
        new_discoveries = []

        for token in tokens:
            key = token.value.lower()

            # Check previous frequency before adding
            prev_frequency = 0
            if key in self.vocabulary and token.category in self.vocabulary[key]:
                prev_frequency = self.vocabulary[key][token.category].frequency

            # Add the token
            self.add_token(token)

            # Check if this addition crossed the discovery threshold
            current_frequency = self.vocabulary[key][token.category].frequency
            if prev_frequency < self.discovery_threshold <= current_frequency:
                new_discoveries.append(self.vocabulary[key][token.category])

        return new_discoveries

    def get_frequent_tokens(self, category: TokenCategory | None = None, min_frequency: int = 5) -> list[Token]:
        """
        Get tokens that appear frequently.

        Args:
            category: Optional category filter
            min_frequency: Minimum frequency threshold

        Returns:
            List of frequent tokens
        """
        frequent_tokens = []

        if category:
            # Get frequent tokens for specific category
            for value, freq in self.frequency_index[category].items():
                if freq >= min_frequency:
                    token = self.vocabulary[value].get(category)
                    if token:
                        frequent_tokens.append(token)
        else:
            # Get frequent tokens across all categories
            frequent_tokens.extend(
                token
                for value_dict in self.vocabulary.values()
                for token in value_dict.values()
                if token.frequency >= min_frequency
            )

        # Sort by frequency (descending)
        return sorted(frequent_tokens, key=lambda t: t.frequency, reverse=True)

    def get_statistics(self) -> dict:
        """Get vocabulary statistics."""
        total_tokens = sum(len(categories) for categories in self.vocabulary.values())

        category_counts: dict[str, int] = defaultdict(int)
        total_frequency = 0

        for value_dict in self.vocabulary.values():
            for category, token in value_dict.items():
                category_counts[category.value] += 1
                total_frequency += token.frequency

        return {
            "total_unique_tokens": len(self.vocabulary),
            "total_token_instances": total_tokens,
            "total_frequency": total_frequency,
            "tokens_by_category": dict(category_counts),
            "discovery_threshold": self.discovery_threshold,
        }

    def get_ambiguous_tokens(self) -> list[str]:
        """
        Get tokens that appear in multiple categories.

        Returns:
            List of ambiguous token values
        """
        ambiguous = []

        for value, categories in self.vocabulary.items():
            if len(categories) > 1:
                ambiguous.append(value)

        return ambiguous

    def calculate_token_confidence(self, token: Token) -> float:
        """
        Calculate confidence score for a token based on vocabulary knowledge.

        Args:
            token: Token to calculate confidence for

        Returns:
            Confidence score between 0 and 1
        """
        key = token.value.lower()

        if key not in self.vocabulary:
            # Unknown token, lower confidence
            return token.confidence * 0.7

        if token.category not in self.vocabulary[key]:
            # Known value but different category
            return token.confidence * 0.8

        # Known token, boost confidence based on frequency
        known_token = self.vocabulary[key][token.category]
        frequency_boost = min(known_token.frequency / 100, 0.2)

        return min(token.confidence + frequency_boost, 1.0)

    def save_vocabulary(self) -> None:
        """Save vocabulary to disk."""
        if not self.vocabulary_path:
            return

        # Convert to serializable format
        vocab_data: dict[str, dict] = {}
        for value, categories in self.vocabulary.items():
            vocab_data[value] = {}
            for category, token in categories.items():
                vocab_data[value][category.value] = {
                    "frequency": token.frequency,
                    "confidence": token.confidence,
                    "first_seen": token.first_seen.isoformat(),
                    "last_seen": token.last_seen.isoformat(),
                }

        # Save to file
        self.vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(self.vocabulary_path).open("w") as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocabulary(self) -> None:
        """Load vocabulary from disk."""
        if not self.vocabulary_path or not self.vocabulary_path.exists():
            return

        try:
            with self.vocabulary_path.open() as f:
                vocab_data = json.load(f)

            # Convert from serialized format
            for value, categories in vocab_data.items():
                for category_str, token_data in categories.items():
                    category = TokenCategory(category_str)
                    token = Token(
                        value=value,
                        category=category,
                        frequency=token_data["frequency"],
                        confidence=token_data["confidence"],
                        first_seen=datetime.fromisoformat(token_data["first_seen"]),
                        last_seen=datetime.fromisoformat(token_data["last_seen"]),
                    )
                    self.vocabulary[value][category] = token
                    self.frequency_index[category][value] = token.frequency

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading vocabulary: {e}")

    def merge_vocabulary(self, other: "VocabularyManager") -> None:
        """Merge another vocabulary into this one."""
        for categories in other.vocabulary.values():
            for token in categories.values():
                self.add_token(token)
