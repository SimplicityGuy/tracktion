"""
Unit tests for configuration management.

Tests configuration loading, validation, and management.
"""

import os
from unittest.mock import patch

from services.analysis_service.src.config import (
    BPMConfig,
    CacheConfig,
    ConfigManager,
    KeyDetectionConfig,
    MessageQueueConfig,
    ModelConfig,
    MoodAnalysisConfig,
    PerformanceConfig,
    ServiceConfig,
    StorageConfig,
    TemporalConfig,
    get_config,
    reset_config,
    set_config,
)


class TestServiceConfig:
    """Test suite for ServiceConfig."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ServiceConfig()

        # Test BPM defaults
        assert config.bpm.confidence_threshold == 0.7
        assert config.bpm.fallback_threshold == 0.5
        assert config.bpm.agreement_tolerance == 5.0
        assert config.bpm.max_file_size_mb == 500
        assert ".mp3" in config.bpm.supported_formats

        # Test Temporal defaults
        assert config.temporal.window_size_seconds == 10.0
        assert config.temporal.start_window_seconds == 30.0
        assert config.temporal.end_window_seconds == 30.0
        assert config.temporal.stability_threshold == 0.8
        assert config.temporal.enable_temporal_storage is True

        # Test Cache defaults
        assert config.cache.enabled is True
        assert config.cache.redis_host == "localhost"
        assert config.cache.redis_port == 6379
        assert config.cache.default_ttl_days == 30
        assert config.cache.algorithm_version == "1.0"

        # Test service-level defaults
        assert config.enable_temporal_analysis is True
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "TRACKTION_BPM_CONFIDENCE_THRESHOLD": "0.9",
            "TRACKTION_BPM_MAX_FILE_SIZE_MB": "1000",
            "TRACKTION_TEMPORAL_WINDOW_SIZE_SECONDS": "15.0",
            "TRACKTION_TEMPORAL_ENABLE_STORAGE": "false",
            "TRACKTION_CACHE_ENABLED": "false",
            "TRACKTION_CACHE_REDIS_HOST": "redis.example.com",
            "TRACKTION_CACHE_REDIS_PORT": "6380",
            "TRACKTION_CACHE_ALGORITHM_VERSION": "2.0",
            "TRACKTION_MQ_RABBITMQ_URL": "amqp://user:pass@rabbitmq:5672/",
            "TRACKTION_MQ_MAX_RETRIES": "10",
            "DATABASE_URL": "postgresql://user:pass@localhost/tracktion",
            "NEO4J_URI": "bolt://neo4j:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
            "TRACKTION_PERF_PARALLEL_WORKERS": "4",
            "TRACKTION_ENABLE_TEMPORAL_ANALYSIS": "false",
            "TRACKTION_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()

            # Verify BPM settings
            assert config.bpm.confidence_threshold == 0.9
            assert config.bpm.max_file_size_mb == 1000

            # Verify Temporal settings
            assert config.temporal.window_size_seconds == 15.0
            assert config.temporal.enable_temporal_storage is False

            # Verify Cache settings
            assert config.cache.enabled is False
            assert config.cache.redis_host == "redis.example.com"
            assert config.cache.redis_port == 6380
            assert config.cache.algorithm_version == "2.0"

            # Verify Message Queue settings
            assert config.message_queue.rabbitmq_url == "amqp://user:pass@rabbitmq:5672/"
            assert config.message_queue.max_retries == 10

            # Verify Storage settings
            assert config.storage.postgres_url == "postgresql://user:pass@localhost/tracktion"
            assert config.storage.neo4j_uri == "bolt://neo4j:7687"
            assert config.storage.neo4j_user == "neo4j"
            assert config.storage.neo4j_password == "secret"

            # Verify Performance settings
            assert config.performance.parallel_workers == 4

            # Verify service-level settings
            assert config.enable_temporal_analysis is False
            assert config.log_level == "DEBUG"

    def test_from_dict(self):
        """Test configuration from dictionary."""
        data = {
            "bpm": {
                "confidence_threshold": 0.85,
                "max_file_size_mb": 750,
            },
            "temporal": {
                "window_size_seconds": 20.0,
                "stability_threshold": 0.9,
            },
            "cache": {
                "enabled": False,
                "redis_port": 6380,
            },
            "enable_temporal_analysis": False,
            "log_level": "WARNING",
        }

        config = ServiceConfig.from_dict(data)

        assert config.bpm.confidence_threshold == 0.85
        assert config.bpm.max_file_size_mb == 750
        assert config.temporal.window_size_seconds == 20.0
        assert config.temporal.stability_threshold == 0.9
        assert config.cache.enabled is False
        assert config.cache.redis_port == 6380
        assert config.enable_temporal_analysis is False
        assert config.log_level == "WARNING"

        # Unchanged defaults
        assert config.bpm.fallback_threshold == 0.5
        assert config.cache.redis_host == "localhost"

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ServiceConfig()
        config.bpm.confidence_threshold = 0.85
        config.cache.redis_port = 6380
        config.enable_temporal_analysis = False

        data = config.to_dict()

        assert data["bpm"]["confidence_threshold"] == 0.85
        assert data["cache"]["redis_port"] == 6380
        assert data["enable_temporal_analysis"] is False

        # Check structure
        assert "bpm" in data
        assert "temporal" in data
        assert "cache" in data
        assert "message_queue" in data
        assert "storage" in data
        assert "performance" in data

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ServiceConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_bpm_config(self):
        """Test validation of invalid BPM configuration."""
        config = ServiceConfig()
        config.bpm.confidence_threshold = 1.5  # > 1
        config.bpm.max_file_size_mb = -10  # negative

        errors = config.validate()
        assert "BPM confidence_threshold must be between 0 and 1" in errors
        assert "BPM max_file_size_mb must be positive" in errors

    def test_validate_invalid_temporal_config(self):
        """Test validation of invalid temporal configuration."""
        config = ServiceConfig()
        config.temporal.window_size_seconds = 0  # not positive
        config.temporal.stability_threshold = 1.5  # > 1

        errors = config.validate()
        assert "Temporal window_size_seconds must be positive" in errors
        assert "Temporal stability_threshold must be between 0 and 1" in errors

    def test_validate_invalid_cache_config(self):
        """Test validation of invalid cache configuration."""
        config = ServiceConfig()
        config.cache.redis_port = 70000  # > 65535
        config.cache.default_ttl_days = 0  # not positive

        errors = config.validate()
        assert "Cache redis_port must be between 1 and 65535" in errors
        assert "Cache default_ttl_days must be positive" in errors

    def test_validate_invalid_performance_config(self):
        """Test validation of invalid performance configuration."""
        config = ServiceConfig()
        config.performance.parallel_workers = 0  # not positive
        config.performance.memory_limit_mb = -100  # negative

        errors = config.validate()
        assert "Performance parallel_workers must be positive" in errors
        assert "Performance memory_limit_mb must be positive" in errors


class TestGlobalConfig:
    """Test suite for global configuration management."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_get_config_singleton(self):
        """Test that get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Test setting custom configuration."""
        custom_config = ServiceConfig()
        custom_config.bpm.confidence_threshold = 0.95

        set_config(custom_config)

        config = get_config()
        assert config.bpm.confidence_threshold == 0.95

    def test_reset_config(self):
        """Test resetting configuration."""
        # Set custom config
        custom_config = ServiceConfig()
        custom_config.bpm.confidence_threshold = 0.95
        set_config(custom_config)

        # Reset
        reset_config()

        # Get new config (should be from env)
        config = get_config()
        assert config.bpm.confidence_threshold == 0.7  # default value

    def test_get_config_from_env(self):
        """Test that get_config loads from environment."""
        env_vars = {
            "TRACKTION_BPM_CONFIDENCE_THRESHOLD": "0.88",
        }

        with patch.dict(os.environ, env_vars):
            reset_config()  # Force reload
            config = get_config()
            assert config.bpm.confidence_threshold == 0.88


class TestIndividualConfigs:
    """Test suite for individual configuration classes."""

    def test_bpm_config_defaults(self):
        """Test BPMConfig default values."""
        config = BPMConfig()
        assert config.confidence_threshold == 0.7
        assert config.fallback_threshold == 0.5
        assert config.agreement_tolerance == 5.0
        assert config.max_file_size_mb == 500
        assert len(config.supported_formats) > 0
        assert ".mp3" in config.supported_formats

    def test_temporal_config_defaults(self):
        """Test TemporalConfig default values."""
        config = TemporalConfig()
        assert config.window_size_seconds == 10.0
        assert config.start_window_seconds == 30.0
        assert config.end_window_seconds == 30.0
        assert config.min_windows_for_analysis == 3
        assert config.stability_threshold == 0.8
        assert config.enable_temporal_storage is True

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert config.redis_password is None
        assert config.default_ttl_days == 30
        assert config.failed_ttl_hours == 1
        assert config.low_confidence_ttl_days == 7
        assert config.algorithm_version == "1.0"
        assert config.use_xxh128 is True

    def test_message_queue_config_defaults(self):
        """Test MessageQueueConfig default values."""
        config = MessageQueueConfig()
        assert config.rabbitmq_url == "amqp://guest:guest@localhost:5672/"
        assert config.queue_name == "analysis_queue"
        assert config.exchange_name == "tracktion_exchange"
        assert config.routing_key == "file.analyze"
        assert config.max_retries == 5
        assert config.base_delay_seconds == 2.0
        assert config.prefetch_count == 1

    def test_storage_config_defaults(self):
        """Test StorageConfig default values."""
        config = StorageConfig()
        assert config.postgres_url is None
        assert config.neo4j_uri is None
        assert config.neo4j_user is None
        assert config.neo4j_password is None
        assert config.store_temporal_array is False
        assert config.batch_size == 100

    def test_performance_config_defaults(self):
        """Test PerformanceConfig default values."""
        config = PerformanceConfig()
        assert config.enable_streaming is True
        assert config.streaming_threshold_mb == 100
        assert config.chunk_size_bytes == 8192
        assert config.parallel_workers == 1
        assert config.memory_limit_mb == 500
        assert config.processing_timeout_seconds == 300


class TestConfigManager:
    """Test suite for ConfigManager singleton class."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_config_manager_singleton(self):
        """Test ConfigManager is properly implemented as singleton."""

        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_config_manager_get_config_creates_from_env(self):
        """Test ConfigManager.get_config creates config from env on first call."""

        manager = ConfigManager()
        # Ensure config is not set
        manager._config = None

        env_vars = {"TRACKTION_BPM_CONFIDENCE_THRESHOLD": "0.95"}
        with patch.dict(os.environ, env_vars):
            config = manager.get_config()
            assert config.bpm.confidence_threshold == 0.95

    def test_config_manager_get_config_returns_cached(self):
        """Test ConfigManager.get_config returns cached config."""

        manager = ConfigManager()
        config1 = manager.get_config()
        config2 = manager.get_config()
        assert config1 is config2

    def test_config_manager_set_config(self):
        """Test ConfigManager.set_config."""

        manager = ConfigManager()
        custom_config = ServiceConfig()
        custom_config.bpm.confidence_threshold = 0.88

        manager.set_config(custom_config)
        retrieved_config = manager.get_config()
        assert retrieved_config is custom_config

    def test_config_manager_reset_config(self):
        """Test ConfigManager.reset_config."""

        manager = ConfigManager()
        # Set a custom config
        custom_config = ServiceConfig()
        manager.set_config(custom_config)

        # Reset should clear the config
        manager.reset_config()
        assert manager._config is None

        # Next get_config should create from env
        config = manager.get_config()
        assert config is not custom_config


class TestConfigurationValidationEdgeCases:
    """Test edge cases for configuration validation."""

    def test_validate_boundary_values(self):
        """Test validation at boundary values."""
        config = ServiceConfig()

        # Test boundary values that should be valid
        config.bpm.confidence_threshold = 0.001  # Just above 0
        config.bpm.fallback_threshold = 1.0  # Exactly 1
        config.temporal.stability_threshold = 0.0  # Exactly 0
        config.cache.redis_port = 1  # Minimum valid port

        errors = config.validate()
        assert errors == []

    def test_validate_multiple_errors(self):
        """Test validation collects multiple errors."""
        config = ServiceConfig()

        # Set multiple invalid values
        config.bpm.confidence_threshold = -0.1
        config.bpm.max_file_size_mb = -50
        config.temporal.window_size_seconds = -10
        config.cache.redis_port = 70000
        config.performance.parallel_workers = 0

        errors = config.validate()
        assert len(errors) >= 5  # Should catch all errors

    def test_validate_edge_case_redis_port(self):
        """Test Redis port validation edge cases."""
        config = ServiceConfig()

        # Test maximum valid port
        config.cache.redis_port = 65535
        errors = config.validate()
        assert not any("redis_port" in error for error in errors)

        # Test minimum invalid port
        config.cache.redis_port = 0
        errors = config.validate()
        assert any("redis_port must be between 1 and 65535" in error for error in errors)


class TestEnvironmentVariableHandling:
    """Test suite for environment variable handling edge cases."""

    def test_from_env_boolean_variations(self):
        """Test boolean environment variable parsing variations."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("NO", False),
            ("invalid", False),  # Invalid values default to False
        ]

        for env_value, expected in test_cases:
            env_vars = {"TRACKTION_CACHE_ENABLED": env_value}
            with patch.dict(os.environ, env_vars):
                config = ServiceConfig.from_env()
                assert config.cache.enabled == expected

    def test_from_env_numeric_conversions(self):
        """Test numeric environment variable conversions."""
        env_vars = {
            "TRACKTION_BPM_CONFIDENCE_THRESHOLD": "0.85",
            "TRACKTION_BPM_MAX_FILE_SIZE_MB": "750",
            "TRACKTION_CACHE_REDIS_PORT": "6380",
            "TRACKTION_PERF_PARALLEL_WORKERS": "8",
        }

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()
            assert config.bpm.confidence_threshold == 0.85
            assert config.bpm.max_file_size_mb == 750
            assert config.cache.redis_port == 6380
            assert config.performance.parallel_workers == 8

    def test_from_env_string_list_parsing(self):
        """Test parsing comma-separated lists from environment."""
        env_vars = {"TRACKTION_MODELS_PRELOAD": "model1,model2,model3"}

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()
            assert config.models.preload_models == ["model1", "model2", "model3"]

    def test_from_env_empty_preload_models(self):
        """Test empty preload models environment variable."""
        env_vars = {"TRACKTION_MODELS_PRELOAD": ""}

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()
            # Empty string fails the walrus operator check, so default value is used
            assert config.models.preload_models == []

    def test_from_env_none_password(self):
        """Test None password handling."""
        # Test when password environment variable is not set
        env_vars = {}
        with patch.dict(os.environ, env_vars, clear=True):
            config = ServiceConfig.from_env()
            assert config.cache.redis_password is None

        # Test when password is set
        env_vars = {"TRACKTION_CACHE_REDIS_PASSWORD": "secret123"}
        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()
            assert config.cache.redis_password == "secret123"


class TestConfigDictOperations:
    """Test suite for dictionary operations on configuration."""

    def test_from_dict_nested_updates(self):
        """Test from_dict only updates existing attributes."""
        data = {
            "bpm": {
                "confidence_threshold": 0.9,
                "nonexistent_field": "should_be_ignored",
            },
            "nonexistent_section": {"some_field": "ignored"},
        }

        config = ServiceConfig.from_dict(data)
        # Existing field should be updated
        assert config.bpm.confidence_threshold == 0.9
        # Non-existent fields should be ignored without error
        assert not hasattr(config.bpm, "nonexistent_field")
        assert not hasattr(config, "nonexistent_section")

    def test_from_dict_partial_sections(self):
        """Test from_dict with partial section updates."""
        data = {
            "cache": {
                "redis_port": 6380,
                # Other cache fields not specified
            }
        }

        config = ServiceConfig.from_dict(data)
        # Specified field should be updated
        assert config.cache.redis_port == 6380
        # Unspecified fields should retain defaults
        assert config.cache.redis_host == "localhost"
        assert config.cache.enabled is True

    def test_to_dict_completeness(self):
        """Test to_dict returns complete configuration structure."""
        config = ServiceConfig()
        data = config.to_dict()

        expected_sections = [
            "bpm",
            "temporal",
            "cache",
            "message_queue",
            "storage",
            "performance",
            "models",
            "key_detection",
            "mood_analysis",
        ]

        for section in expected_sections:
            assert section in data

        expected_service_fields = ["enable_temporal_analysis", "log_level", "metrics_enabled", "health_check_port"]

        for field in expected_service_fields:
            assert field in data

    def test_to_dict_values_match_config(self):
        """Test to_dict values match the configuration object."""
        config = ServiceConfig()
        config.bpm.confidence_threshold = 0.85
        config.cache.redis_port = 6380
        config.log_level = "DEBUG"

        data = config.to_dict()

        assert data["bpm"]["confidence_threshold"] == 0.85
        assert data["cache"]["redis_port"] == 6380
        assert data["log_level"] == "DEBUG"


class TestNewConfigurationSections:
    """Test suite for newer configuration sections not fully covered."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""

        config = ModelConfig()
        assert config.models_dir == "services/analysis_service/models"
        assert config.auto_download is True
        assert config.verify_checksum is True
        assert config.lazy_load is True
        assert config.preload_models == []
        assert config.model_repo_base == "https://essentia.upf.edu/models/"

    def test_key_detection_config_defaults(self):
        """Test KeyDetectionConfig default values."""

        config = KeyDetectionConfig()
        assert config.confidence_threshold == 0.7
        assert config.agreement_boost == 1.2
        assert config.disagreement_penalty == 0.8
        assert config.needs_review_threshold == 0.7

    def test_mood_analysis_config_defaults(self):
        """Test MoodAnalysisConfig default values."""

        config = MoodAnalysisConfig()
        assert config.enable_mood_detection is True
        assert config.enable_genre_detection is True
        assert config.enable_danceability is True
        assert config.ensemble_voting_threshold == 0.5
        assert config.confidence_threshold == 0.6
        assert "happy" in config.mood_dimensions
        assert "sad" in config.mood_dimensions
        assert len(config.mood_dimensions) == 7
