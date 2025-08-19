"""
Configuration management for the analysis service.

Provides centralized configuration for BPM detection, temporal analysis,
caching, and other service settings.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BPMConfig:
    """Configuration for BPM detection."""

    confidence_threshold: float = 0.7
    fallback_threshold: float = 0.5
    agreement_tolerance: float = 5.0
    max_file_size_mb: int = 500
    supported_formats: list = field(default_factory=lambda: [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac"])


@dataclass
class TemporalConfig:
    """Configuration for temporal BPM analysis."""

    window_size_seconds: float = 10.0
    start_window_seconds: float = 30.0
    end_window_seconds: float = 30.0
    min_windows_for_analysis: int = 3
    stability_threshold: float = 0.8
    enable_temporal_storage: bool = True


@dataclass
class CacheConfig:
    """Configuration for Redis caching."""

    enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl_days: int = 30
    failed_ttl_hours: int = 1
    low_confidence_ttl_days: int = 7
    algorithm_version: str = "1.0"
    use_xxh128: bool = True


@dataclass
class MessageQueueConfig:
    """Configuration for RabbitMQ messaging."""

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    queue_name: str = "analysis_queue"
    exchange_name: str = "tracktion_exchange"
    routing_key: str = "file.analyze"
    max_retries: int = 5
    base_delay_seconds: float = 2.0
    prefetch_count: int = 1


@dataclass
class StorageConfig:
    """Configuration for database storage."""

    postgres_url: Optional[str] = None
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    store_temporal_array: bool = False
    batch_size: int = 100


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    enable_streaming: bool = True
    streaming_threshold_mb: int = 100
    chunk_size_bytes: int = 8192
    parallel_workers: int = 1
    memory_limit_mb: int = 500
    processing_timeout_seconds: int = 300


@dataclass
class ModelConfig:
    """Configuration for TensorFlow model management."""

    models_dir: str = "services/analysis_service/models"
    auto_download: bool = True
    verify_checksum: bool = True
    lazy_load: bool = True
    preload_models: list = field(default_factory=list)
    model_repo_base: str = "https://essentia.upf.edu/models/"


@dataclass
class KeyDetectionConfig:
    """Configuration for musical key detection."""

    confidence_threshold: float = 0.7
    agreement_boost: float = 1.2
    disagreement_penalty: float = 0.8
    needs_review_threshold: float = 0.7


@dataclass
class MoodAnalysisConfig:
    """Configuration for mood and genre analysis."""

    enable_mood_detection: bool = True
    enable_genre_detection: bool = True
    enable_danceability: bool = True
    ensemble_voting_threshold: float = 0.5
    confidence_threshold: float = 0.6
    mood_dimensions: list = field(
        default_factory=lambda: ["happy", "sad", "aggressive", "relaxed", "acoustic", "electronic", "party"]
    )


@dataclass
class ServiceConfig:
    """Main configuration for the analysis service."""

    bpm: BPMConfig = field(default_factory=BPMConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    message_queue: MessageQueueConfig = field(default_factory=MessageQueueConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    key_detection: KeyDetectionConfig = field(default_factory=KeyDetectionConfig)
    mood_analysis: MoodAnalysisConfig = field(default_factory=MoodAnalysisConfig)

    # Service-level settings
    enable_temporal_analysis: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_port: int = 8080

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Create configuration from environment variables.

        Environment variables follow the pattern:
        TRACKTION_<SECTION>_<SETTING>

        Examples:
        - TRACKTION_BPM_CONFIDENCE_THRESHOLD=0.8
        - TRACKTION_CACHE_ENABLED=false
        - TRACKTION_TEMPORAL_WINDOW_SIZE_SECONDS=15.0
        """
        config = cls()

        # BPM configuration
        if val := os.getenv("TRACKTION_BPM_CONFIDENCE_THRESHOLD"):
            config.bpm.confidence_threshold = float(val)
        if val := os.getenv("TRACKTION_BPM_FALLBACK_THRESHOLD"):
            config.bpm.fallback_threshold = float(val)
        if val := os.getenv("TRACKTION_BPM_AGREEMENT_TOLERANCE"):
            config.bpm.agreement_tolerance = float(val)
        if val := os.getenv("TRACKTION_BPM_MAX_FILE_SIZE_MB"):
            config.bpm.max_file_size_mb = int(val)

        # Temporal configuration
        if val := os.getenv("TRACKTION_TEMPORAL_WINDOW_SIZE_SECONDS"):
            config.temporal.window_size_seconds = float(val)
        if val := os.getenv("TRACKTION_TEMPORAL_START_WINDOW_SECONDS"):
            config.temporal.start_window_seconds = float(val)
        if val := os.getenv("TRACKTION_TEMPORAL_END_WINDOW_SECONDS"):
            config.temporal.end_window_seconds = float(val)
        if val := os.getenv("TRACKTION_TEMPORAL_STABILITY_THRESHOLD"):
            config.temporal.stability_threshold = float(val)
        if val := os.getenv("TRACKTION_TEMPORAL_ENABLE_STORAGE"):
            config.temporal.enable_temporal_storage = val.lower() in ("true", "1", "yes")

        # Cache configuration
        if val := os.getenv("TRACKTION_CACHE_ENABLED"):
            config.cache.enabled = val.lower() in ("true", "1", "yes")
        config.cache.redis_host = os.getenv("TRACKTION_CACHE_REDIS_HOST", config.cache.redis_host)
        if val := os.getenv("TRACKTION_CACHE_REDIS_PORT"):
            config.cache.redis_port = int(val)
        if val := os.getenv("TRACKTION_CACHE_REDIS_DB"):
            config.cache.redis_db = int(val)
        config.cache.redis_password = os.getenv("TRACKTION_CACHE_REDIS_PASSWORD")
        if val := os.getenv("TRACKTION_CACHE_DEFAULT_TTL_DAYS"):
            config.cache.default_ttl_days = int(val)
        if val := os.getenv("TRACKTION_CACHE_ALGORITHM_VERSION"):
            config.cache.algorithm_version = val

        # Message queue configuration
        config.message_queue.rabbitmq_url = os.getenv("TRACKTION_MQ_RABBITMQ_URL", config.message_queue.rabbitmq_url)
        config.message_queue.queue_name = os.getenv("TRACKTION_MQ_QUEUE_NAME", config.message_queue.queue_name)
        config.message_queue.exchange_name = os.getenv("TRACKTION_MQ_EXCHANGE_NAME", config.message_queue.exchange_name)
        config.message_queue.routing_key = os.getenv("TRACKTION_MQ_ROUTING_KEY", config.message_queue.routing_key)
        if val := os.getenv("TRACKTION_MQ_MAX_RETRIES"):
            config.message_queue.max_retries = int(val)

        # Storage configuration
        config.storage.postgres_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        config.storage.neo4j_uri = os.getenv("NEO4J_URI")
        config.storage.neo4j_user = os.getenv("NEO4J_USER")
        config.storage.neo4j_password = os.getenv("NEO4J_PASSWORD")
        if val := os.getenv("TRACKTION_STORAGE_STORE_TEMPORAL_ARRAY"):
            config.storage.store_temporal_array = val.lower() in ("true", "1", "yes")

        # Performance configuration
        if val := os.getenv("TRACKTION_PERF_ENABLE_STREAMING"):
            config.performance.enable_streaming = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_PERF_STREAMING_THRESHOLD_MB"):
            config.performance.streaming_threshold_mb = int(val)
        if val := os.getenv("TRACKTION_PERF_PARALLEL_WORKERS"):
            config.performance.parallel_workers = int(val)
        if val := os.getenv("TRACKTION_PERF_MEMORY_LIMIT_MB"):
            config.performance.memory_limit_mb = int(val)
        if val := os.getenv("TRACKTION_PERF_PROCESSING_TIMEOUT_SECONDS"):
            config.performance.processing_timeout_seconds = int(val)

        # Model configuration
        config.models.models_dir = os.getenv("TRACKTION_MODELS_DIR", config.models.models_dir)
        if val := os.getenv("TRACKTION_MODELS_AUTO_DOWNLOAD"):
            config.models.auto_download = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_MODELS_VERIFY_CHECKSUM"):
            config.models.verify_checksum = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_MODELS_LAZY_LOAD"):
            config.models.lazy_load = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_MODELS_PRELOAD"):
            config.models.preload_models = val.split(",")

        # Key detection configuration
        if val := os.getenv("TRACKTION_KEY_CONFIDENCE_THRESHOLD"):
            config.key_detection.confidence_threshold = float(val)
        if val := os.getenv("TRACKTION_KEY_AGREEMENT_BOOST"):
            config.key_detection.agreement_boost = float(val)
        if val := os.getenv("TRACKTION_KEY_DISAGREEMENT_PENALTY"):
            config.key_detection.disagreement_penalty = float(val)
        if val := os.getenv("TRACKTION_KEY_NEEDS_REVIEW_THRESHOLD"):
            config.key_detection.needs_review_threshold = float(val)

        # Mood analysis configuration
        if val := os.getenv("TRACKTION_MOOD_ENABLE_MOOD"):
            config.mood_analysis.enable_mood_detection = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_MOOD_ENABLE_GENRE"):
            config.mood_analysis.enable_genre_detection = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_MOOD_ENABLE_DANCEABILITY"):
            config.mood_analysis.enable_danceability = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_MOOD_ENSEMBLE_THRESHOLD"):
            config.mood_analysis.ensemble_voting_threshold = float(val)
        if val := os.getenv("TRACKTION_MOOD_CONFIDENCE_THRESHOLD"):
            config.mood_analysis.confidence_threshold = float(val)

        # Service-level settings
        if val := os.getenv("TRACKTION_ENABLE_TEMPORAL_ANALYSIS"):
            config.enable_temporal_analysis = val.lower() in ("true", "1", "yes")
        config.log_level = os.getenv("TRACKTION_LOG_LEVEL", config.log_level)
        if val := os.getenv("TRACKTION_METRICS_ENABLED"):
            config.metrics_enabled = val.lower() in ("true", "1", "yes")
        if val := os.getenv("TRACKTION_HEALTH_CHECK_PORT"):
            config.health_check_port = int(val)

        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        """Create configuration from a dictionary.

        Useful for loading from JSON or YAML files.
        """
        config = cls()

        # Update BPM config
        if "bpm" in data:
            for key, value in data["bpm"].items():
                if hasattr(config.bpm, key):
                    setattr(config.bpm, key, value)

        # Update Temporal config
        if "temporal" in data:
            for key, value in data["temporal"].items():
                if hasattr(config.temporal, key):
                    setattr(config.temporal, key, value)

        # Update Cache config
        if "cache" in data:
            for key, value in data["cache"].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)

        # Update Message Queue config
        if "message_queue" in data:
            for key, value in data["message_queue"].items():
                if hasattr(config.message_queue, key):
                    setattr(config.message_queue, key, value)

        # Update Storage config
        if "storage" in data:
            for key, value in data["storage"].items():
                if hasattr(config.storage, key):
                    setattr(config.storage, key, value)

        # Update Performance config
        if "performance" in data:
            for key, value in data["performance"].items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)

        # Update Model config
        if "models" in data:
            for key, value in data["models"].items():
                if hasattr(config.models, key):
                    setattr(config.models, key, value)

        # Update Key Detection config
        if "key_detection" in data:
            for key, value in data["key_detection"].items():
                if hasattr(config.key_detection, key):
                    setattr(config.key_detection, key, value)

        # Update Mood Analysis config
        if "mood_analysis" in data:
            for key, value in data["mood_analysis"].items():
                if hasattr(config.mood_analysis, key):
                    setattr(config.mood_analysis, key, value)

        # Update service-level settings
        for key in ["enable_temporal_analysis", "log_level", "metrics_enabled", "health_check_port"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "bpm": {
                "confidence_threshold": self.bpm.confidence_threshold,
                "fallback_threshold": self.bpm.fallback_threshold,
                "agreement_tolerance": self.bpm.agreement_tolerance,
                "max_file_size_mb": self.bpm.max_file_size_mb,
                "supported_formats": self.bpm.supported_formats,
            },
            "temporal": {
                "window_size_seconds": self.temporal.window_size_seconds,
                "start_window_seconds": self.temporal.start_window_seconds,
                "end_window_seconds": self.temporal.end_window_seconds,
                "min_windows_for_analysis": self.temporal.min_windows_for_analysis,
                "stability_threshold": self.temporal.stability_threshold,
                "enable_temporal_storage": self.temporal.enable_temporal_storage,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "redis_host": self.cache.redis_host,
                "redis_port": self.cache.redis_port,
                "redis_db": self.cache.redis_db,
                "default_ttl_days": self.cache.default_ttl_days,
                "failed_ttl_hours": self.cache.failed_ttl_hours,
                "low_confidence_ttl_days": self.cache.low_confidence_ttl_days,
                "algorithm_version": self.cache.algorithm_version,
                "use_xxh128": self.cache.use_xxh128,
            },
            "message_queue": {
                "rabbitmq_url": self.message_queue.rabbitmq_url,
                "queue_name": self.message_queue.queue_name,
                "exchange_name": self.message_queue.exchange_name,
                "routing_key": self.message_queue.routing_key,
                "max_retries": self.message_queue.max_retries,
                "base_delay_seconds": self.message_queue.base_delay_seconds,
                "prefetch_count": self.message_queue.prefetch_count,
            },
            "storage": {
                "postgres_url": self.storage.postgres_url,
                "neo4j_uri": self.storage.neo4j_uri,
                "neo4j_user": self.storage.neo4j_user,
                "store_temporal_array": self.storage.store_temporal_array,
                "batch_size": self.storage.batch_size,
            },
            "performance": {
                "enable_streaming": self.performance.enable_streaming,
                "streaming_threshold_mb": self.performance.streaming_threshold_mb,
                "chunk_size_bytes": self.performance.chunk_size_bytes,
                "parallel_workers": self.performance.parallel_workers,
                "memory_limit_mb": self.performance.memory_limit_mb,
                "processing_timeout_seconds": self.performance.processing_timeout_seconds,
            },
            "models": {
                "models_dir": self.models.models_dir,
                "auto_download": self.models.auto_download,
                "verify_checksum": self.models.verify_checksum,
                "lazy_load": self.models.lazy_load,
                "preload_models": self.models.preload_models,
                "model_repo_base": self.models.model_repo_base,
            },
            "key_detection": {
                "confidence_threshold": self.key_detection.confidence_threshold,
                "agreement_boost": self.key_detection.agreement_boost,
                "disagreement_penalty": self.key_detection.disagreement_penalty,
                "needs_review_threshold": self.key_detection.needs_review_threshold,
            },
            "mood_analysis": {
                "enable_mood_detection": self.mood_analysis.enable_mood_detection,
                "enable_genre_detection": self.mood_analysis.enable_genre_detection,
                "enable_danceability": self.mood_analysis.enable_danceability,
                "ensemble_voting_threshold": self.mood_analysis.ensemble_voting_threshold,
                "confidence_threshold": self.mood_analysis.confidence_threshold,
                "mood_dimensions": self.mood_analysis.mood_dimensions,
            },
            "enable_temporal_analysis": self.enable_temporal_analysis,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
            "health_check_port": self.health_check_port,
        }

    def validate(self) -> list:
        """Validate configuration and return any errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate BPM settings
        if not 0 < self.bpm.confidence_threshold <= 1:
            errors.append("BPM confidence_threshold must be between 0 and 1")
        if not 0 < self.bpm.fallback_threshold <= 1:
            errors.append("BPM fallback_threshold must be between 0 and 1")
        if self.bpm.agreement_tolerance < 0:
            errors.append("BPM agreement_tolerance must be non-negative")
        if self.bpm.max_file_size_mb <= 0:
            errors.append("BPM max_file_size_mb must be positive")

        # Validate Temporal settings
        if self.temporal.window_size_seconds <= 0:
            errors.append("Temporal window_size_seconds must be positive")
        if self.temporal.start_window_seconds <= 0:
            errors.append("Temporal start_window_seconds must be positive")
        if self.temporal.end_window_seconds <= 0:
            errors.append("Temporal end_window_seconds must be positive")
        if not 0 <= self.temporal.stability_threshold <= 1:
            errors.append("Temporal stability_threshold must be between 0 and 1")

        # Validate Cache settings
        if self.cache.redis_port <= 0 or self.cache.redis_port > 65535:
            errors.append("Cache redis_port must be between 1 and 65535")
        if self.cache.default_ttl_days <= 0:
            errors.append("Cache default_ttl_days must be positive")

        # Validate Performance settings
        if self.performance.streaming_threshold_mb <= 0:
            errors.append("Performance streaming_threshold_mb must be positive")
        if self.performance.parallel_workers <= 0:
            errors.append("Performance parallel_workers must be positive")
        if self.performance.memory_limit_mb <= 0:
            errors.append("Performance memory_limit_mb must be positive")
        if self.performance.processing_timeout_seconds <= 0:
            errors.append("Performance processing_timeout_seconds must be positive")

        return errors


# Global configuration instance
_config: Optional[ServiceConfig] = None


def get_config() -> ServiceConfig:
    """Get the global configuration instance.

    Creates the configuration from environment variables on first call.
    """
    global _config
    if _config is None:
        _config = ServiceConfig.from_env()
    return _config


def set_config(config: ServiceConfig) -> None:
    """Set the global configuration instance.

    Useful for testing or when loading configuration from files.
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance.

    The next call to get_config() will recreate from environment.
    """
    global _config
    _config = None
