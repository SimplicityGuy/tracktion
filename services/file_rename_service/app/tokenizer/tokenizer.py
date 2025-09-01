"""
Main tokenizer implementation for filename analysis.
"""

import re
import time
import unicodedata
from pathlib import Path

from .classifier import TokenClassifier
from .models import Token, TokenCategory, TokenizedFilename
from .patterns import PatternMatcher
from .vocabulary import VocabularyManager

# Constants
FILE_EXTENSION_PARTS = 2  # Number of parts when splitting by extension
MAX_FILE_EXTENSION_LENGTH = 5  # Maximum length for valid file extensions
CONFIDENCE_THRESHOLD_FOR_UNKNOWN = 0.5  # Minimum confidence to include unknown tokens
MAX_NOISE_LENGTH = 2  # Maximum length for text considered noise


class Tokenizer:
    """Main tokenizer for analyzing and extracting patterns from filenames."""

    def __init__(
        self,
        vocabulary_path: Path | None = None,
        enable_caching: bool = True,
        batch_size: int = 100,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocabulary_path: Path to persist vocabulary
            enable_caching: Enable caching for performance
            batch_size: Default batch size for processing
        """
        self.pattern_matcher = PatternMatcher()
        self.vocabulary_manager = VocabularyManager(vocabulary_path)
        self.classifier = TokenClassifier()
        self.enable_caching = enable_caching
        self.batch_size = batch_size
        self._cache: dict[str, TokenizedFilename] = {}
        self._stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "total_time_ms": 0.0,
        }

    def tokenize(self, filename: str) -> TokenizedFilename:
        """
        Tokenize a single filename.

        Args:
            filename: Filename to tokenize

        Returns:
            TokenizedFilename object with extracted tokens
        """
        start_time = time.time()

        # Check cache
        if self.enable_caching and filename in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[filename]

        # Clean and prepare filename
        cleaned = self._clean_filename(filename)

        # Extract tokens using pattern matching
        matches = self.pattern_matcher.match(cleaned)

        # Process matches and create tokens
        tokens = self._process_matches(matches, cleaned)

        # Extract unmatched segments
        unmatched = self.pattern_matcher.extract_unmatched(cleaned, matches)

        # Try to classify unmatched segments
        additional_tokens = self._classify_unmatched(unmatched)
        tokens.extend(additional_tokens)

        # Calculate confidence scores
        for token in tokens:
            token.confidence = self.vocabulary_manager.calculate_token_confidence(token)

        # Update vocabulary
        self.vocabulary_manager.add_tokens(tokens)

        # Create result
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result = TokenizedFilename(
            original_filename=filename,
            tokens=tokens,
            unmatched_segments=[s for s in unmatched if not self._is_noise(s)],
            confidence_score=self._calculate_overall_confidence(tokens),
            processing_time_ms=processing_time,
        )

        # Cache result
        if self.enable_caching:
            self._cache[filename] = result

        # Update stats
        self._stats["total_processed"] += 1
        self._stats["total_time_ms"] += processing_time

        return result

    def tokenize_batch(self, filenames: list[str]) -> list[TokenizedFilename]:
        """
        Tokenize multiple filenames in batch.

        Args:
            filenames: List of filenames to tokenize

        Returns:
            List of TokenizedFilename objects
        """
        results = []

        # Process in batches for efficiency
        for i in range(0, len(filenames), self.batch_size):
            batch = filenames[i : i + self.batch_size]
            batch_results = [self.tokenize(f) for f in batch]
            results.extend(batch_results)

            # Periodically save vocabulary
            if i % (self.batch_size * 10) == 0:
                self.vocabulary_manager.save_vocabulary()

        # Final vocabulary save
        self.vocabulary_manager.save_vocabulary()

        return results

    def _clean_filename(self, filename: str) -> str:
        """
        Clean and normalize filename for processing.

        Args:
            filename: Raw filename

        Returns:
            Cleaned filename
        """

        # First, check if there's a file extension to handle
        if "." in filename:
            # Check if last part after . looks like a file extension (short, alphanumeric)
            parts = filename.rsplit(".", 1)
            if len(parts) == FILE_EXTENSION_PARTS and len(parts[1]) <= MAX_FILE_EXTENSION_LENGTH and parts[1].isalnum():
                name, ext = parts
                # Keep extension if it's a format indicator
                filename = f"{name} {ext}" if ext.upper() in ["FLAC", "MP3", "WAV", "APE", "SHN", "M4A"] else name

        # Normalize unicode
        filename = unicodedata.normalize("NFKD", filename)

        # Replace underscores and tildes with spaces
        filename = filename.replace("_", " ").replace("~", " ")

        # For dashes, keep them if part of dates, otherwise replace with space
        # Preserve date patterns like 2023-07-14 or 2023.07.14

        # Check if there are date patterns with dashes
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}-\d{2}-\d{2}")
        # Find all date matches
        date_matches = list(date_pattern.finditer(filename))

        if date_matches:
            # Replace dashes outside of date patterns
            new_filename = []
            last_end = 0
            for match in date_matches:
                # Add the part before the date (with dashes replaced)
                new_filename.append(filename[last_end : match.start()].replace("-", " "))
                # Add the date pattern as-is
                new_filename.append(filename[match.start() : match.end()])
                last_end = match.end()
            # Add any remaining part
            new_filename.append(filename[last_end:].replace("-", " "))
            filename = "".join(new_filename)
        else:
            # No date patterns, replace all dashes
            filename = filename.replace("-", " ")

        # Normalize whitespace
        return " ".join(filename.split())

    def _process_matches(self, matches: list[tuple[Token, int, int]], text: str) -> list[Token]:
        """
        Process pattern matches and resolve overlaps.

        Args:
            matches: List of (Token, start, end) tuples
            text: Original text

        Returns:
            List of processed tokens
        """
        if not matches:
            return []

        # Sort by position and priority
        sorted_matches = sorted(matches, key=lambda m: (m[1], -m[0].confidence))

        # Remove overlapping matches (keep higher priority)
        filtered = []
        last_end = -1

        for token, start, end in sorted_matches:
            if start >= last_end:
                filtered.append(token)
                last_end = end

        return filtered

    def _classify_unmatched(self, segments: list[str]) -> list[Token]:
        """
        Attempt to classify unmatched text segments.

        Args:
            segments: List of unmatched text segments

        Returns:
            List of classified tokens
        """
        tokens = []

        for segment in segments:
            # Skip noise
            if self._is_noise(segment):
                continue

            # Try to classify
            category, confidence = self.classifier.classify(segment)

            if category != TokenCategory.UNKNOWN or confidence > CONFIDENCE_THRESHOLD_FOR_UNKNOWN:
                token = Token(
                    value=segment,
                    category=category,
                    confidence=confidence,
                    original_text=segment,
                )
                tokens.append(token)

        return tokens

    def _is_noise(self, text: str) -> bool:
        """
        Check if text segment is likely noise.

        Args:
            text: Text segment to check

        Returns:
            True if likely noise
        """
        # Single characters (except meaningful ones)
        if len(text) == 1 and text not in "0123456789":
            return True

        # Only special characters
        if all(c in "()[]{}!@#$%^&*-_=+|\\:;\"'<>,.?/" for c in text):
            return True

        # Very short and no alphanumeric
        return len(text) <= MAX_NOISE_LENGTH and not any(c.isalnum() for c in text)

    def _calculate_overall_confidence(self, tokens: list[Token]) -> float:
        """
        Calculate overall confidence score for tokenization.

        Args:
            tokens: List of extracted tokens

        Returns:
            Overall confidence score (0-1)
        """
        if not tokens:
            return 0.0

        # Weight by token category importance
        category_weights = {
            TokenCategory.ARTIST: 1.5,
            TokenCategory.DATE: 1.3,
            TokenCategory.VENUE: 1.2,
            TokenCategory.QUALITY: 1.0,
            TokenCategory.FORMAT: 1.0,
            TokenCategory.SOURCE: 1.0,
            TokenCategory.TRACK: 0.8,
            TokenCategory.SET: 0.8,
            TokenCategory.TOUR: 1.1,
            TokenCategory.LABEL: 0.9,
            TokenCategory.CATALOG: 0.9,
            TokenCategory.UNKNOWN: 0.3,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for token in tokens:
            weight = category_weights.get(token.category, 0.5)
            weighted_sum += token.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def analyze_patterns(self, filenames: list[str]) -> dict:
        """
        Analyze patterns across multiple filenames.

        Args:
            filenames: List of filenames to analyze

        Returns:
            Analysis results including pattern frequencies
        """
        # Tokenize all filenames
        results = self.tokenize_batch(filenames)

        # Collect frequency data
        category_freq: dict[str, int] = {}
        token_freq: dict[str, int] = {}

        for result in results:
            for token in result.tokens:
                # Category frequency
                cat_key = token.category.value
                category_freq[cat_key] = category_freq.get(cat_key, 0) + 1

                # Token value frequency
                token_key = f"{token.category.value}:{token.value}"
                token_freq[token_key] = token_freq.get(token_key, 0) + 1

        # Get pattern statistics
        pattern_stats = self.pattern_matcher.get_pattern_statistics()

        # Get vocabulary statistics
        vocab_stats = self.vocabulary_manager.get_statistics()

        # Calculate accuracy (based on coverage)
        total_coverage = sum(r.coverage_ratio for r in results)
        avg_coverage = total_coverage / len(results) if results else 0.0

        return {
            "total_files": len(filenames),
            "total_tokens": sum(r.token_count for r in results),
            "average_tokens_per_file": (sum(r.token_count for r in results) / len(results) if results else 0),
            "average_confidence": (sum(r.confidence_score for r in results) / len(results) if results else 0),
            "average_coverage": avg_coverage,
            "estimated_accuracy": avg_coverage * 100,  # Rough accuracy estimate
            "category_frequencies": category_freq,
            "top_tokens": sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:20],
            "pattern_statistics": pattern_stats,
            "vocabulary_statistics": vocab_stats,
            "processing_stats": self.get_statistics(),
        }

    def get_statistics(self) -> dict:
        """Get tokenizer performance statistics."""
        avg_time = (
            self._stats["total_time_ms"] / self._stats["total_processed"] if self._stats["total_processed"] > 0 else 0
        )

        return {
            "total_processed": self._stats["total_processed"],
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["total_processed"] if self._stats["total_processed"] > 0 else 0
            ),
            "average_time_ms": avg_time,
            "total_time_ms": self._stats["total_time_ms"],
        }

    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        self._cache.clear()
        self._stats["cache_hits"] = 0
