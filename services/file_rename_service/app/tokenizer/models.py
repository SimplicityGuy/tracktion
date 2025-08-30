"""
Data models for the tokenizer module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TokenCategory(Enum):
    """Categories for token classification."""

    ARTIST = "artist"
    DATE = "date"
    VENUE = "venue"
    QUALITY = "quality"
    FORMAT = "format"
    SOURCE = "source"
    TRACK = "track"
    SET = "set"
    TOUR = "tour"
    LABEL = "label"
    CATALOG = "catalog"
    UNKNOWN = "unknown"


@dataclass
class Token:
    """Represents a tokenized component from a filename."""

    value: str
    category: TokenCategory
    frequency: int = 1
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    position: int | None = None
    original_text: str | None = None

    def update_frequency(self) -> None:
        """Increment frequency and update last seen timestamp."""
        self.frequency += 1
        self.last_seen = datetime.now()


@dataclass
class Pattern:
    """Represents a regex pattern for matching filename components."""

    regex: str
    category: TokenCategory
    priority: int = 0
    match_count: int = 0
    description: str | None = None
    examples: list[str] = field(default_factory=list)

    def increment_match_count(self) -> None:
        """Increment the match count for this pattern."""
        self.match_count += 1


@dataclass
class TokenizedFilename:
    """Result of tokenizing a filename."""

    original_filename: str
    tokens: list[Token]
    unmatched_segments: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0

    @property
    def token_count(self) -> int:
        """Get the number of tokens extracted."""
        return len(self.tokens)

    @property
    def coverage_ratio(self) -> float:
        """Calculate the ratio of matched text to total filename length."""
        if not self.original_filename:
            return 0.0

        matched_length = sum(len(t.original_text or t.value) for t in self.tokens)
        return matched_length / len(self.original_filename)

    def get_tokens_by_category(self, category: TokenCategory) -> list[Token]:
        """Get all tokens of a specific category."""
        return [t for t in self.tokens if t.category == category]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "original_filename": self.original_filename,
            "tokens": [
                {
                    "value": t.value,
                    "category": t.category.value,
                    "confidence": t.confidence,
                    "position": t.position,
                    "original_text": t.original_text,
                }
                for t in self.tokens
            ],
            "unmatched_segments": self.unmatched_segments,
            "confidence_score": self.confidence_score,
            "coverage_ratio": self.coverage_ratio,
            "token_count": self.token_count,
            "processing_time_ms": self.processing_time_ms,
        }
