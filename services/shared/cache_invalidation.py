"""
Cache invalidation strategies and utilities for the Tracktion services.

This module provides centralized cache invalidation logic with different
strategies for various use cases and data types.
"""

import hashlib
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar, Protocol

logger = logging.getLogger(__name__)


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""

    IMMEDIATE = "immediate"  # Invalidate immediately
    TIME_BASED = "time_based"  # Invalidate after TTL expires
    EVENT_BASED = "event_based"  # Invalidate on specific events
    LAZY = "lazy"  # Invalidate on next access if conditions met
    PATTERN_BASED = "pattern_based"  # Invalidate by key patterns


class CacheBackend(Protocol):
    """Protocol for cache backends."""

    def delete(self, key: str) -> bool:
        """Delete a single key."""
        ...

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching a pattern."""
        ...

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...


class CacheInvalidationManager:
    """Manages cache invalidation strategies across services."""

    # TTL configurations by data type (in seconds)
    TTL_CONFIG: ClassVar[dict[str, dict[str, int]]] = {
        # Audio analysis results - longer TTL since they're expensive to compute
        "audio_analysis": {
            "bpm": 30 * 24 * 60 * 60,  # 30 days
            "temporal": 30 * 24 * 60 * 60,  # 30 days
            "key": 30 * 24 * 60 * 60,  # 30 days
            "mood": 7 * 24 * 60 * 60,  # 7 days (may change more often)
        },
        # ML predictions - medium TTL
        "ml_predictions": {
            "high_confidence": 24 * 60 * 60,  # 24 hours
            "low_confidence": 4 * 60 * 60,  # 4 hours
            "training_jobs": 7 * 24 * 60 * 60,  # 7 days
        },
        # CUE generation - medium TTL
        "cue_generation": {
            "content": 7200,  # 2 hours
            "validation": 1800,  # 30 minutes
            "format_capabilities": 24 * 60 * 60,  # 24 hours
        },
        # Search results - short TTL due to dynamic nature
        "search_results": {
            "successful": 60 * 60,  # 1 hour
            "failed": 10 * 60,  # 10 minutes
        },
        # Configuration data - long TTL
        "configuration": {
            "static": 7 * 24 * 60 * 60,  # 7 days
            "dynamic": 60 * 60,  # 1 hour
        },
    }

    def __init__(self, cache_backend: CacheBackend):
        """Initialize with a cache backend."""
        self.cache = cache_backend
        self._invalidation_rules: list[dict[str, Any]] = []

    def add_invalidation_rule(
        self,
        pattern: str,
        strategy: InvalidationStrategy,
        condition: str | None = None,
        ttl: int | None = None,
    ) -> None:
        """Add an invalidation rule."""
        rule = {
            "pattern": pattern,
            "strategy": strategy,
            "condition": condition,
            "ttl": ttl,
            "created_at": datetime.now(UTC),
        }
        self._invalidation_rules.append(rule)
        logger.debug(f"Added invalidation rule: {rule}")

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        try:
            deleted = self.cache.clear_pattern(pattern)
            logger.info(f"Invalidated {deleted} cache entries matching pattern: {pattern}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {e}")
            return 0

    def invalidate_audio_analysis(self, file_path: str, analysis_types: list[str] | None = None) -> int:
        """Invalidate audio analysis cache entries for a file."""
        analysis_types = analysis_types or ["bpm", "temporal", "key", "mood"]
        total_deleted = 0

        # Generate file hash for cache keys (simplified)
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]

        for analysis_type in analysis_types:
            pattern = f"{analysis_type}:{file_hash}:*"
            deleted = self.invalidate_by_pattern(pattern)
            total_deleted += deleted

        logger.info(f"Invalidated {total_deleted} audio analysis entries for {file_path}")
        return total_deleted

    def invalidate_ml_predictions(self, model_version: str | None = None, filename_pattern: str | None = None) -> int:
        """Invalidate ML prediction cache entries."""
        total_deleted = 0

        if model_version:
            # Invalidate all predictions for a specific model version
            pattern = f"ml_predictions:{model_version}:*"
            deleted = self.invalidate_by_pattern(pattern)
            total_deleted += deleted

        if filename_pattern:
            # Invalidate predictions for specific filename patterns
            pattern = f"ml_predictions:*:{filename_pattern}:*"
            deleted = self.invalidate_by_pattern(pattern)
            total_deleted += deleted

        logger.info(f"Invalidated {total_deleted} ML prediction entries")
        return total_deleted

    def invalidate_tracklist_cache(self, tracklist_id: str) -> int:
        """Invalidate all cache entries for a tracklist."""
        patterns = [
            f"cue_content:{tracklist_id}:*",
            f"validation:*:{tracklist_id}:*",
            f"tracklist:search:*:{tracklist_id}:*",
        ]

        total_deleted = 0
        for pattern in patterns:
            deleted = self.invalidate_by_pattern(pattern)
            total_deleted += deleted

        logger.info(f"Invalidated {total_deleted} tracklist entries for {tracklist_id}")
        return total_deleted

    def invalidate_search_cache(self, search_type: str | None = None, query_pattern: str | None = None) -> int:
        """Invalidate search cache entries."""
        total_deleted = 0

        if search_type and query_pattern:
            pattern = f"tracklist:search:{search_type}:{query_pattern}:*"
        elif search_type:
            pattern = f"tracklist:search:{search_type}:*"
        elif query_pattern:
            pattern = f"tracklist:search:*:{query_pattern}:*"
        else:
            pattern = "tracklist:search:*"

        deleted = self.invalidate_by_pattern(pattern)
        total_deleted += deleted

        # Also invalidate failed searches
        failed_pattern = pattern.replace("tracklist:search:", "tracklist:searchfailed:")
        deleted = self.invalidate_by_pattern(failed_pattern)
        total_deleted += deleted

        logger.info(f"Invalidated {total_deleted} search cache entries")
        return total_deleted

    def set_adaptive_ttl(self, key: str, data_type: str, subtype: str, confidence: float = 1.0) -> bool:
        """Set adaptive TTL based on data type and confidence."""
        try:
            base_ttl = self.TTL_CONFIG.get(data_type, {}).get(subtype, 3600)  # Default 1 hour

            # Adjust TTL based on confidence
            if confidence < 0.5:
                adjusted_ttl = max(base_ttl // 4, 300)  # Min 5 minutes
            elif confidence < 0.7:
                adjusted_ttl = base_ttl // 2
            else:
                adjusted_ttl = base_ttl

            return self.cache.expire(key, adjusted_ttl)
        except Exception as e:
            logger.error(f"Failed to set adaptive TTL for {key}: {e}")
            return False

    def schedule_cleanup(self, data_type: str, age_threshold_hours: int = 24) -> int:
        """Schedule cleanup of old cache entries."""
        # This would typically be implemented with a background task
        # For now, we'll implement immediate cleanup logic

        cleanup_patterns = {
            "audio_analysis": ["bpm:*", "temporal:*", "key:*", "mood:*"],
            "ml_predictions": ["ml_predictions:*"],
            "cue_generation": ["cue_content:*", "validation:*", "format_capabilities:*"],
            "search_results": ["tracklist:search:*", "tracklist:searchfailed:*"],
        }

        patterns = cleanup_patterns.get(data_type, [])
        total_deleted = 0

        for pattern in patterns:
            deleted = self.invalidate_by_pattern(pattern)
            total_deleted += deleted

        logger.info(f"Cleanup removed {total_deleted} entries for data type: {data_type}")
        return total_deleted

    def bulk_invalidate(self, invalidation_requests: list[dict[str, Any]]) -> dict[str, int]:
        """Process multiple invalidation requests."""
        results = {}

        for request in invalidation_requests:
            request_type = request.get("type")
            params = request.get("params", {})

            try:
                if request_type == "pattern":
                    deleted = self.invalidate_by_pattern(params["pattern"])
                elif request_type == "audio_analysis":
                    deleted = self.invalidate_audio_analysis(params["file_path"], params.get("analysis_types"))
                elif request_type == "ml_predictions":
                    deleted = self.invalidate_ml_predictions(
                        params.get("model_version"), params.get("filename_pattern")
                    )
                elif request_type == "tracklist":
                    deleted = self.invalidate_tracklist_cache(params["tracklist_id"])
                elif request_type == "search":
                    deleted = self.invalidate_search_cache(params.get("search_type"), params.get("query_pattern"))
                else:
                    logger.warning(f"Unknown invalidation request type: {request_type}")
                    deleted = 0

                results[f"{request_type}_{params}"] = deleted

            except Exception as e:
                logger.error(f"Failed to process invalidation request {request}: {e}")
                results[f"{request_type}_{params}"] = 0

        return results

    def get_invalidation_stats(self) -> dict[str, Any]:
        """Get statistics about cache invalidation."""
        return {
            "rules_count": len(self._invalidation_rules),
            "ttl_config": self.TTL_CONFIG,
            "strategies": [strategy.value for strategy in InvalidationStrategy],
        }


# Utility functions
def get_ttl_for_confidence(base_ttl: int, confidence: float) -> int:
    """Get TTL adjusted for confidence level."""
    if confidence < 0.3:
        return max(base_ttl // 8, 300)  # Very low confidence - 5 min minimum
    if confidence < 0.5:
        return max(base_ttl // 4, 600)  # Low confidence - 10 min minimum
    if confidence < 0.7:
        return base_ttl // 2  # Medium confidence
    if confidence < 0.9:
        return int(base_ttl * 0.8)  # Good confidence
    return base_ttl  # High confidence


def get_ttl_for_data_age(base_ttl: int, age_hours: float) -> int:
    """Get TTL adjusted for data age."""
    if age_hours < 1:
        return base_ttl  # Fresh data
    if age_hours < 24:
        return max(int(base_ttl * 0.8), 1800)  # Slightly aged - min 30 min
    if age_hours < 168:  # 1 week
        return max(int(base_ttl * 0.6), 900)  # Week old - min 15 min
    return max(int(base_ttl * 0.3), 300)  # Very old - min 5 min
