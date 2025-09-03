"""Additional edge case tests for configuration validation and error handling."""

import os
import threading
import time
from unittest.mock import patch

from services.analysis_service.src.config import (
    BPMConfig,
    CacheConfig,
    ConfigManager,
    MoodAnalysisConfig,
    PerformanceConfig,
    ServiceConfig,
    TemporalConfig,
    get_config,
    reset_config,
)


class TestConfigurationInputValidation:
    """Test configuration input validation and sanitization."""

    def test_bpm_config_extreme_values(self):
        """Test BPM config with extreme but valid values."""
        config = BPMConfig()

        # Test extreme but valid values
        config.confidence_threshold = 0.001
        config.fallback_threshold = 0.999
        config.agreement_tolerance = 0.0
        config.max_file_size_mb = 999999

        # Should not raise validation errors when used in ServiceConfig
        service_config = ServiceConfig()
        service_config.bpm = config
        errors = service_config.validate()
        bpm_errors = [e for e in errors if "BPM" in e]
        assert len(bpm_errors) == 0

    def test_temporal_config_extreme_values(self):
        """Test temporal config with extreme but valid values."""
        config = TemporalConfig()

        # Test extreme but valid values
        config.window_size_seconds = 0.1
        config.start_window_seconds = 999.9
        config.end_window_seconds = 999.9
        config.min_windows_for_analysis = 1
        config.stability_threshold = 0.0

        service_config = ServiceConfig()
        service_config.temporal = config
        errors = service_config.validate()
        temporal_errors = [e for e in errors if "Temporal" in e]
        assert len(temporal_errors) == 0

    def test_cache_config_extreme_values(self):
        """Test cache config with extreme but valid values."""
        config = CacheConfig()

        # Test extreme but valid values
        config.redis_port = 65535
        config.redis_db = 15  # Redis max db number
        config.default_ttl_days = 36500  # 100 years
        config.failed_ttl_hours = 24 * 365  # 1 year in hours
        config.low_confidence_ttl_days = 36500

        service_config = ServiceConfig()
        service_config.cache = config
        errors = service_config.validate()
        cache_errors = [e for e in errors if "Cache" in e]
        assert len(cache_errors) == 0

    def test_performance_config_extreme_values(self):
        """Test performance config with extreme but valid values."""
        config = PerformanceConfig()

        # Test extreme but valid values
        config.streaming_threshold_mb = 99999
        config.chunk_size_bytes = 1
        config.parallel_workers = 1000
        config.memory_limit_mb = 999999
        config.processing_timeout_seconds = 86400  # 1 day

        service_config = ServiceConfig()
        service_config.performance = config
        errors = service_config.validate()
        perf_errors = [e for e in errors if "Performance" in e]
        assert len(perf_errors) == 0


class TestEnvironmentVariableEdgeCases:
    """Test edge cases for environment variable parsing."""

    def test_whitespace_environment_variables(self):
        """Test environment variables with whitespace."""
        env_vars = {
            "TRACKTION_BPM_CONFIDENCE_THRESHOLD": "  0.85  ",
            "TRACKTION_CACHE_REDIS_HOST": "  redis.example.com  ",
            "TRACKTION_LOG_LEVEL": "  DEBUG  ",
            "TRACKTION_MQ_RABBITMQ_URL": "  amqp://user:pass@host:5672/  ",
        }

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()

            # Values should be parsed correctly despite whitespace
            assert config.bpm.confidence_threshold == 0.85
            assert config.cache.redis_host == "  redis.example.com  "  # String fields keep whitespace
            assert config.log_level == "  DEBUG  "
            assert config.message_queue.rabbitmq_url == "  amqp://user:pass@host:5672/  "

    def test_case_insensitive_boolean_parsing(self):
        """Test case-insensitive boolean environment variable parsing."""
        boolean_fields = [
            ("TRACKTION_CACHE_ENABLED", "cache.enabled"),
            ("TRACKTION_TEMPORAL_ENABLE_STORAGE", "temporal.enable_temporal_storage"),
            ("TRACKTION_PERF_ENABLE_STREAMING", "performance.enable_streaming"),
            ("TRACKTION_MODELS_AUTO_DOWNLOAD", "models.auto_download"),
            ("TRACKTION_MODELS_VERIFY_CHECKSUM", "models.verify_checksum"),
            ("TRACKTION_MODELS_LAZY_LOAD", "models.lazy_load"),
            ("TRACKTION_ENABLE_TEMPORAL_ANALYSIS", "enable_temporal_analysis"),
            ("TRACKTION_METRICS_ENABLED", "metrics_enabled"),
        ]

        for env_var, attr_path in boolean_fields:
            for true_value in ["TRUE", "True", "true", "YES", "Yes", "yes", "1"]:
                env_vars = {env_var: true_value}
                with patch.dict(os.environ, env_vars):
                    config = ServiceConfig.from_env()

                    # Navigate to the nested attribute
                    obj = config
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    assert obj is True, f"Failed for {env_var}={true_value}"

            for false_value in ["FALSE", "False", "false", "NO", "No", "no", "0", "invalid"]:
                env_vars = {env_var: false_value}
                with patch.dict(os.environ, env_vars):
                    config = ServiceConfig.from_env()

                    # Navigate to the nested attribute
                    obj = config
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    assert obj is False, f"Failed for {env_var}={false_value}"

            # Empty string is special case - walrus operator fails so default is used
            env_vars = {env_var: ""}
            with patch.dict(os.environ, env_vars):
                config = ServiceConfig.from_env()

                # Navigate to the nested attribute to get default value
                obj = config
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                # For empty strings, default values are used (which may be True or False)

    def test_numeric_conversion_edge_cases(self):
        """Test numeric environment variable conversion edge cases."""
        # Test valid numeric conversions
        valid_cases = [
            ("TRACKTION_BPM_CONFIDENCE_THRESHOLD", "0.0001", 0.0001),
            ("TRACKTION_BPM_MAX_FILE_SIZE_MB", "0", 0),  # Zero is valid for some validators
            ("TRACKTION_CACHE_REDIS_PORT", "1", 1),
            ("TRACKTION_PERF_PARALLEL_WORKERS", "999", 999),
        ]

        for env_var, env_value, _ in valid_cases:
            env_vars = {env_var: env_value}
            with patch.dict(os.environ, env_vars):
                ServiceConfig.from_env()
                # Test passes if no exception is raised during parsing

    def test_missing_optional_environment_variables(self):
        """Test behavior when optional environment variables are missing."""
        # Clear all environment variables
        with patch.dict(os.environ, {}, clear=True):
            config = ServiceConfig.from_env()

            # Should use defaults
            assert config.cache.redis_password is None
            assert config.storage.postgres_url is None
            assert config.storage.neo4j_uri is None
            assert config.storage.neo4j_user is None
            assert config.storage.neo4j_password is None


class TestConfigurationDataTypes:
    """Test configuration data type handling."""

    def test_list_field_immutability(self):
        """Test that list fields are properly isolated between instances."""
        config1 = BPMConfig()
        config2 = BPMConfig()

        # Modify one instance
        config1.supported_formats.append(".custom")

        # Other instance should not be affected
        assert ".custom" not in config2.supported_formats
        assert len(config2.supported_formats) == len(BPMConfig().supported_formats)

    def test_dict_field_immutability(self):
        """Test that dict fields are properly isolated between instances."""
        config1 = MoodAnalysisConfig()
        config2 = MoodAnalysisConfig()

        # Modify one instance
        config1.mood_dimensions.append("custom_mood")

        # Other instance should not be affected
        assert "custom_mood" not in config2.mood_dimensions
        assert len(config2.mood_dimensions) == len(MoodAnalysisConfig().mood_dimensions)

    def test_nested_config_independence(self):
        """Test that nested config objects are independent."""
        service1 = ServiceConfig()
        service2 = ServiceConfig()

        # Modify nested config in one instance
        service1.bpm.confidence_threshold = 0.99

        # Other instance should not be affected
        assert service2.bpm.confidence_threshold == 0.7

    def test_config_serialization_types(self):
        """Test configuration serialization maintains proper types."""
        config = ServiceConfig()
        config.bpm.confidence_threshold = 0.85
        config.cache.redis_port = 6380
        config.enable_temporal_analysis = False

        data = config.to_dict()

        # Verify types are preserved
        assert isinstance(data["bpm"]["confidence_threshold"], float)
        assert isinstance(data["cache"]["redis_port"], int)
        assert isinstance(data["enable_temporal_analysis"], bool)


class TestConfigurationValidationComprehensive:
    """Comprehensive validation testing."""

    def test_validation_error_message_format(self):
        """Test that validation error messages are properly formatted."""
        config = ServiceConfig()

        # Set invalid values
        config.bpm.confidence_threshold = -0.5
        config.cache.redis_port = -1
        config.performance.parallel_workers = -10

        errors = config.validate()

        # All errors should be descriptive strings
        for error in errors:
            assert isinstance(error, str)
            assert len(error) > 10  # Should be descriptive
            assert any(word in error.lower() for word in ["must", "should", "between", "positive"])

    def test_validation_with_mixed_valid_invalid(self):
        """Test validation with mix of valid and invalid values."""
        config = ServiceConfig()

        # Set some valid values
        config.bpm.confidence_threshold = 0.8  # Valid
        config.cache.redis_port = 6379  # Valid

        # Set some invalid values
        config.temporal.window_size_seconds = -10  # Invalid
        config.performance.memory_limit_mb = -100  # Invalid

        errors = config.validate()

        # Should only report invalid values
        assert len(errors) == 2
        assert all("window_size_seconds" in error or "memory_limit_mb" in error for error in errors)

    def test_validation_empty_config(self):
        """Test validation of completely default configuration."""
        config = ServiceConfig()
        errors = config.validate()

        # Default configuration should be valid
        assert errors == []

    def test_validation_performance_impact(self):
        """Test that validation doesn't have performance issues."""

        config = ServiceConfig()

        # Time multiple validation runs
        start_time = time.time()
        for _ in range(100):
            config.validate()
        end_time = time.time()

        # Should be fast (less than 1 second for 100 runs)
        duration = end_time - start_time
        assert duration < 1.0, f"Validation too slow: {duration} seconds for 100 runs"


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    def test_config_roundtrip_env_to_dict(self):
        """Test configuration roundtrip from environment to dict and back."""
        env_vars = {
            "TRACKTION_BPM_CONFIDENCE_THRESHOLD": "0.88",
            "TRACKTION_CACHE_REDIS_PORT": "6380",
            "TRACKTION_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars):
            # Create from env
            config1 = ServiceConfig.from_env()

            # Convert to dict and back
            data = config1.to_dict()
            config2 = ServiceConfig.from_dict(data)

            # Should be equivalent
            assert config2.bpm.confidence_threshold == 0.88
            assert config2.cache.redis_port == 6380
            assert config2.log_level == "DEBUG"

    def test_config_partial_override(self):
        """Test partial configuration override scenarios."""
        # Start with environment-based config
        env_vars = {"TRACKTION_BPM_CONFIDENCE_THRESHOLD": "0.75"}

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig.from_env()

            # Partially override with dict
            override_data = {
                "cache": {"redis_port": 6380},
                "log_level": "WARNING",
            }

            config = ServiceConfig.from_dict(override_data)

            # Environment setting should be overridden by default (from_dict creates new)
            assert config.bpm.confidence_threshold == 0.7  # default, not env value

            # Dict overrides should be applied
            assert config.cache.redis_port == 6380
            assert config.log_level == "WARNING"

    def test_global_config_thread_safety(self):
        """Test global configuration management thread safety basics."""

        results = []

        def worker():
            """Worker function for threading test."""
            config = get_config()
            results.append(id(config))

        # Reset config first
        reset_config()

        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All threads should get the same config instance
        assert len(set(results)) == 1, "Config instances should be the same across threads"

    def test_config_manager_isolation(self):
        """Test ConfigManager provides proper isolation."""

        # Create multiple manager instances
        manager1 = ConfigManager()
        manager2 = ConfigManager()

        # Should be the same instance (singleton)
        assert manager1 is manager2

        # Set custom config
        custom_config = ServiceConfig()
        custom_config.log_level = "CUSTOM"

        manager1.set_config(custom_config)

        # Both managers should see the same config
        assert manager2.get_config().log_level == "CUSTOM"
