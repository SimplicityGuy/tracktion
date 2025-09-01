"""Feature extraction and engineering for ML models."""

import hashlib
import re
from collections import Counter
from typing import Any

import numpy as np


class FeatureExtractor:
    """Extract features from tokenized filenames for ML model input."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize feature extractor with configuration."""
        self.config = config or self._default_config()
        self.vocabulary: dict[str, int] = {}
        self.max_sequence_length = self.config.get("max_sequence_length", 50)
        self.embedding_dim = self.config.get("embedding_dim", 100)
        self.use_embeddings = self.config.get("use_embeddings", False)

    def _default_config(self) -> dict[str, Any]:
        """Return default feature extraction configuration."""
        return {
            "max_sequence_length": 50,
            "max_vocabulary_size": 10000,
            "embedding_dim": 100,
            "use_embeddings": False,
            "include_metadata": True,
            "include_statistics": True,
        }

    def fit(self, training_data: list[dict[str, Any]]) -> None:
        """Build vocabulary from training data."""
        token_counter: Counter[str] = Counter()

        for sample in training_data:
            tokens = sample.get("tokens", [])
            for token in tokens:
                token_value = token.get("value", "")
                token_type = token.get("type", "")
                # Create a unique key for token+type combination
                token_key = f"{token_type}:{token_value}"
                token_counter[token_key] += 1

        # Build vocabulary with most common tokens
        max_vocab = self.config.get("max_vocabulary_size", 10000)
        self.vocabulary = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}

        for token_key, _ in token_counter.most_common(max_vocab - 4):
            self.vocabulary[token_key] = len(self.vocabulary)

    def extract_features(self, tokens: list[dict[str, Any]], filename: str) -> dict[str, Any]:
        """Extract features from tokenized filename."""
        features: dict[str, Any] = {}

        # Token-based features
        if self.use_embeddings:
            features["token_embeddings"] = self._create_embeddings(tokens)
        else:
            features["token_encoding"] = self._encode_tokens(tokens)

        # Statistical features
        if self.config.get("include_statistics", True):
            stats = self._extract_statistics(tokens, filename)
            features.update(stats)

        # Metadata features
        if self.config.get("include_metadata", True):
            features.update(self._extract_metadata(tokens, filename))

        return features

    def _encode_tokens(self, tokens: list[dict[str, Any]]) -> np.ndarray:
        """Encode tokens as one-hot or integer sequences."""
        encoded = []

        # Add start token
        encoded.append(self.vocabulary.get("<START>", 1))

        for token in tokens[: self.max_sequence_length - 2]:
            token_value = token.get("value", "")
            token_type = token.get("type", "")
            token_key = f"{token_type}:{token_value}"
            encoded.append(self.vocabulary.get(token_key, self.vocabulary["<UNK>"]))

        # Add end token
        encoded.append(self.vocabulary.get("<END>", 3))

        # Pad sequence
        while len(encoded) < self.max_sequence_length:
            encoded.append(self.vocabulary["<PAD>"])

        return np.array(encoded[: self.max_sequence_length])

    def _create_embeddings(self, tokens: list[dict[str, Any]]) -> np.ndarray:
        """Create embedding representations for tokens."""
        # For now, use random embeddings - in production, use pre-trained
        embeddings = []

        for token in tokens[: self.max_sequence_length]:
            token_value = token.get("value", "")
            token_type = token.get("type", "")
            token_key = f"{token_type}:{token_value}"

            # Generate consistent embedding for each token
            if token_key in self.vocabulary:
                # Use hash for consistent random embedding
                seed = int(hashlib.md5(token_key.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                embedding = rng.standard_normal(self.embedding_dim)
            else:
                # Unknown token embedding
                embedding = np.zeros(self.embedding_dim)

            embeddings.append(embedding)

        # Pad with zeros
        while len(embeddings) < self.max_sequence_length:
            embeddings.append(np.zeros(self.embedding_dim))

        return np.array(embeddings[: self.max_sequence_length])

    def _extract_statistics(self, tokens: list[dict[str, Any]], filename: str) -> dict[str, float]:
        """Extract statistical features from tokens."""
        stats: dict[str, float] = {}

        # Token counts by type
        type_counts = Counter(token.get("type", "") for token in tokens)
        stats["num_words"] = type_counts.get("word", 0)
        stats["num_numbers"] = type_counts.get("number", 0)
        stats["num_separators"] = type_counts.get("separator", 0)
        stats["num_extensions"] = type_counts.get("extension", 0)
        stats["num_dates"] = type_counts.get("date", 0)
        stats["num_versions"] = type_counts.get("version", 0)

        # Filename characteristics
        stats["filename_length"] = float(len(filename))
        stats["num_tokens"] = float(len(tokens))
        stats["avg_token_length"] = float(np.mean([len(t.get("value", "")) for t in tokens]) if tokens else 0)

        # Pattern detection
        stats["has_camelcase"] = float(bool(re.search(r"[a-z][A-Z]", filename)))
        stats["has_snakecase"] = float("_" in filename)
        stats["has_kebabcase"] = float("-" in filename)
        stats["has_spaces"] = float(" " in filename)

        return stats

    def _extract_metadata(self, tokens: list[dict[str, Any]], filename: str) -> dict[str, Any]:
        """Extract metadata features from tokens."""
        metadata = {}

        # Extension information
        extension_tokens = [t for t in tokens if t.get("type") == "extension"]
        if extension_tokens:
            metadata["extension"] = extension_tokens[0].get("value", "")
        else:
            metadata["extension"] = ""

        # Date information
        date_tokens = [t for t in tokens if t.get("type") == "date"]
        metadata["has_date"] = len(date_tokens) > 0
        metadata["num_dates"] = len(date_tokens)

        # Version information
        version_tokens = [t for t in tokens if t.get("type") == "version"]
        metadata["has_version"] = len(version_tokens) > 0

        # Common patterns
        metadata["starts_with_number"] = tokens[0].get("type") == "number" if tokens else False
        metadata["ends_with_number"] = tokens[-2].get("type") == "number" if len(tokens) > 1 else False

        return metadata

    def transform_batch(self, samples: list[dict[str, Any]]) -> dict[str, np.ndarray | None]:
        """Transform batch of samples into feature arrays."""
        batch_features: dict[str, list[Any]] = {
            "token_encodings": [],
            "statistics": [],
            "metadata": [],
        }

        for sample in samples:
            tokens = sample.get("tokens", [])
            filename = sample.get("filename_original", "")
            features = self.extract_features(tokens, filename)

            if "token_encoding" in features:
                batch_features["token_encodings"].append(features["token_encoding"])

            # Combine statistical features
            stats = [
                features[key]
                for key in sorted(features.keys())
                if key.startswith(("num_", "has_")) or key in {"filename_length", "avg_token_length"}
            ]
            batch_features["statistics"].append(stats)

        # Convert to numpy arrays
        return {
            "token_encodings": (
                np.array(batch_features["token_encodings"]) if batch_features["token_encodings"] else None
            ),
            "statistics": (np.array(batch_features["statistics"]) if batch_features["statistics"] else None),
        }
