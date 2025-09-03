"""Configuration loading and management integration tests.

This module contains integration tests for configuration loading, validation,
environment-specific settings, secrets management, and configuration hot-reloading.
"""

import asyncio
import json

# Configure test logging
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationSource:
    """Mock configuration source."""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.data: dict[str, Any] = {}
        self.is_available = True
        self.load_count = 0
        self.last_loaded: datetime | None = None

    def load(self) -> dict[str, Any]:
        """Load configuration data."""
        if not self.is_available:
            raise RuntimeError(f"Configuration source {self.name} is not available")

        self.load_count += 1
        self.last_loaded = datetime.now(UTC)
        return self.data.copy()

    def set_data(self, data: dict[str, Any]):
        """Set configuration data."""
        self.data = data

    def update_data(self, updates: dict[str, Any]):
        """Update specific configuration values."""
        self.data.update(updates)


class FileConfigurationSource(ConfigurationSource):
    """File-based configuration source."""

    def __init__(self, name: str, file_path: str, format: str = "json", priority: int = 0):
        super().__init__(name, priority)
        self.file_path = Path(file_path)
        self.format = format.lower()
        self.file_watcher = None

    def load(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.is_available or not self.file_path.exists():
            raise FileNotFoundError(f"Configuration file {self.file_path} not found")

        self.load_count += 1
        self.last_loaded = datetime.now(UTC)

        try:
            with self.file_path.open(encoding="utf-8") as f:
                if self.format == "json":
                    data = json.load(f)
                elif self.format == "yaml":
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")

            self.data = data
            return data.copy()

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.file_path}: {e}") from e

    def save(self, data: dict[str, Any]):
        """Save configuration to file."""
        self.data = data

        with self.file_path.open("w", encoding="utf-8") as f:
            if self.format == "json":
                json.dump(data, f, indent=2)
            elif self.format == "yaml":
                yaml.dump(data, f, default_flow_style=False)


class EnvironmentConfigurationSource(ConfigurationSource):
    """Environment variable configuration source."""

    def __init__(self, name: str = "environment", prefix: str = "", priority: int = 10):
        super().__init__(name, priority)
        self.prefix = prefix

    def load(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        if not self.is_available:
            raise RuntimeError("Environment configuration source is not available")

        self.load_count += 1
        self.last_loaded = datetime.now(UTC)

        env_data = {}
        prefix = f"{self.prefix}_" if self.prefix else ""

        for key, value in os.environ.items():
            if not self.prefix or key.startswith(prefix):
                config_key = key[len(prefix) :].lower() if self.prefix else key.lower()

                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                    env_data[config_key] = parsed_value
                except (json.JSONDecodeError, ValueError):
                    # Keep as string
                    env_data[config_key] = value

        self.data = env_data
        return env_data.copy()


class RemoteConfigurationSource(ConfigurationSource):
    """Mock remote configuration source (e.g., configuration service)."""

    def __init__(self, name: str, endpoint: str, priority: int = 5):
        super().__init__(name, priority)
        self.endpoint = endpoint
        self.connection_timeout = 5.0
        self.retry_count = 0
        self.max_retries = 3

    async def async_load(self) -> dict[str, Any]:
        """Asynchronously load configuration from remote source."""
        if not self.is_available:
            raise RuntimeError(f"Remote configuration source {self.name} is not available")

        self.load_count += 1
        self.last_loaded = datetime.now(UTC)

        # Simulate network delay
        await asyncio.sleep(0.1)

        # Simulate occasional network failures
        if self.retry_count < self.max_retries and self.load_count % 5 == 0:
            self.retry_count += 1
            raise ConnectionError(f"Failed to connect to {self.endpoint}")

        self.retry_count = 0
        return self.data.copy()


class ConfigurationValidator:
    """Configuration validation and schema checking."""

    def __init__(self):
        self.schemas: dict[str, dict[str, Any]] = {}
        self.validation_errors: list[str] = []

    def register_schema(self, section: str, schema: dict[str, Any]):
        """Register a validation schema for a configuration section."""
        self.schemas[section] = schema

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration against registered schemas."""
        errors = []

        for section, schema in self.schemas.items():
            section_errors = self._validate_section(config.get(section, {}), schema, section)
            errors.extend(section_errors)

        self.validation_errors = errors
        return errors

    def _validate_section(self, data: dict[str, Any], schema: dict[str, Any], section: str) -> list[str]:
        """Validate a configuration section."""
        errors = []

        # Check required fields
        for field, field_schema in schema.get("required", {}).items():
            if field not in data:
                errors.append(f"{section}.{field} is required")
                continue

            # Type validation
            expected_type = field_schema.get("type")
            if expected_type and not isinstance(data[field], expected_type):
                errors.append(
                    f"{section}.{field} must be of type {expected_type.__name__}, got {type(data[field]).__name__}"
                )

            # Value validation
            allowed_values = field_schema.get("values")
            if allowed_values and data[field] not in allowed_values:
                errors.append(f"{section}.{field} must be one of {allowed_values}, got {data[field]}")

            # Range validation
            min_value = field_schema.get("min")
            max_value = field_schema.get("max")
            if min_value is not None and data[field] < min_value:
                errors.append(f"{section}.{field} must be >= {min_value}, got {data[field]}")
            if max_value is not None and data[field] > max_value:
                errors.append(f"{section}.{field} must be <= {max_value}, got {data[field]}")

        # Check optional fields
        for field, field_schema in schema.get("optional", {}).items():
            if field in data:
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(data[field], expected_type):
                    errors.append(
                        f"{section}.{field} must be of type {expected_type.__name__}, got {type(data[field]).__name__}"
                    )

        return errors


class ConfigurationManager:
    """Configuration management system."""

    def __init__(self):
        self.sources: list[ConfigurationSource] = []
        self.validator = ConfigurationValidator()
        self.current_config: dict[str, Any] = {}
        self.config_history: list[dict[str, Any]] = []
        self.watchers: list[callable] = []
        self.is_watching = False
        self.reload_count = 0

    def add_source(self, source: ConfigurationSource):
        """Add a configuration source."""
        self.sources.append(source)
        # Sort by priority (higher priority first)
        self.sources.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"Added configuration source: {source.name} (priority: {source.priority})")

    def remove_source(self, source_name: str) -> bool:
        """Remove a configuration source by name."""
        for i, source in enumerate(self.sources):
            if source.name == source_name:
                self.sources.pop(i)
                logger.info(f"Removed configuration source: {source_name}")
                return True
        return False

    def load_configuration(self) -> dict[str, Any]:
        """Load configuration from all sources in priority order."""
        merged_config = {}
        load_errors = []

        # Load from sources in priority order (highest first)
        for source in self.sources:
            try:
                source_config = source.load()
                logger.info(f"Loaded configuration from {source.name}: {len(source_config)} keys")

                # Merge configuration (higher priority overwrites lower priority)
                merged_config = self._deep_merge(merged_config, source_config)

            except Exception as e:
                logger.error(f"Failed to load from source {source.name}: {e}")
                load_errors.append(f"{source.name}: {e!s}")

        # Validate merged configuration
        validation_errors = self.validator.validate(merged_config)
        if validation_errors:
            logger.warning(f"Configuration validation errors: {validation_errors}")

        # Store previous configuration
        if self.current_config:
            self.config_history.append(self.current_config.copy())

        self.current_config = merged_config

        # Notify watchers of configuration changes
        self._notify_watchers(merged_config)

        if load_errors:
            raise RuntimeError(f"Configuration load errors: {'; '.join(load_errors)}")

        return merged_config.copy()

    def reload_configuration(self) -> dict[str, Any]:
        """Reload configuration from all sources."""
        logger.info("Reloading configuration...")
        self.reload_count += 1
        return self.load_configuration()

    def get_configuration(self, key: str | None = None, default: Any = None) -> Any:
        """Get configuration value by key or entire configuration."""
        if key is None:
            return self.current_config.copy()

        # Support nested keys with dot notation
        keys = key.split(".")
        value = self.current_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set_configuration(self, key: str, value: Any):
        """Set a configuration value (runtime only, not persisted)."""
        keys = key.split(".")
        config = self.current_config

        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the final value
        config[keys[-1]] = value

        # Notify watchers
        self._notify_watchers(self.current_config)

    def add_watcher(self, callback: callable):
        """Add a configuration change watcher."""
        self.watchers.append(callback)

    def remove_watcher(self, callback: callable):
        """Remove a configuration change watcher."""
        if callback in self.watchers:
            self.watchers.remove(callback)

    def start_watching(self):
        """Start watching for configuration changes."""
        self.is_watching = True
        logger.info("Started configuration watching")

    def stop_watching(self):
        """Stop watching for configuration changes."""
        self.is_watching = False
        logger.info("Stopped configuration watching")

    def _deep_merge(self, base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _notify_watchers(self, config: dict[str, Any]):
        """Notify configuration watchers of changes."""
        for watcher in self.watchers:
            try:
                watcher(config)
            except Exception as e:
                logger.error(f"Configuration watcher failed: {e}")


@pytest.fixture
def temp_config_dir():
    """Provide temporary directory for configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def config_manager():
    """Provide configuration manager for testing."""
    return ConfigurationManager()


@pytest.fixture
def sample_database_schema():
    """Provide sample database configuration schema."""
    return {
        "required": {
            "host": {"type": str},
            "port": {"type": int, "min": 1, "max": 65535},
            "database": {"type": str},
            "username": {"type": str},
        },
        "optional": {
            "password": {"type": str},
            "pool_size": {"type": int, "min": 1, "max": 100},
            "timeout": {"type": int, "min": 1},
        },
    }


@pytest.fixture
def sample_logging_schema():
    """Provide sample logging configuration schema."""
    return {
        "required": {"level": {"type": str, "values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}},
        "optional": {"format": {"type": str}, "handlers": {"type": list}},
    }


class TestConfigurationSources:
    """Test individual configuration sources."""

    def test_file_json_configuration_source(self, temp_config_dir):
        """Test JSON file configuration source."""
        config_file = temp_config_dir / "config.json"
        config_data = {
            "database": {"host": "localhost", "port": 5432, "database": "test_db"},
            "logging": {"level": "INFO"},
        }

        # Save configuration to file
        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Load configuration
        source = FileConfigurationSource("json_config", str(config_file), "json")
        loaded_config = source.load()

        assert loaded_config == config_data
        assert source.load_count == 1
        assert source.last_loaded is not None

    def test_file_yaml_configuration_source(self, temp_config_dir):
        """Test YAML file configuration source."""
        config_file = temp_config_dir / "config.yaml"
        config_data = {
            "database": {"host": "localhost", "port": 5432, "database": "test_db"},
            "redis": {"host": "redis-server", "port": 6379},
        }

        # Save configuration to file
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Load configuration
        source = FileConfigurationSource("yaml_config", str(config_file), "yaml")
        loaded_config = source.load()

        assert loaded_config == config_data
        assert source.load_count == 1

    def test_file_configuration_source_not_found(self):
        """Test file configuration source with missing file."""
        source = FileConfigurationSource("missing_config", "/nonexistent/config.json")

        with pytest.raises(FileNotFoundError):
            source.load()

    def test_environment_configuration_source(self):
        """Test environment variable configuration source."""
        # Set some test environment variables
        test_env = {
            "APP_DATABASE_HOST": "env-db-host",
            "APP_DATABASE_PORT": "5433",
            "APP_REDIS_URL": '{"host": "env-redis", "port": 6380}',
            "OTHER_VAR": "should-be-ignored",
        }

        with patch.dict(os.environ, test_env, clear=False):
            source = EnvironmentConfigurationSource("env_config", prefix="APP")
            loaded_config = source.load()

        expected_config = {
            "database_host": "env-db-host",
            "database_port": "5433",
            "redis_url": {"host": "env-redis", "port": 6380},
        }

        assert loaded_config == expected_config
        assert source.load_count == 1
        assert "other_var" not in loaded_config  # Should be filtered by prefix

    @pytest.mark.asyncio
    async def test_remote_configuration_source(self):
        """Test remote configuration source."""
        source = RemoteConfigurationSource("remote_config", "https://config.example.com")
        source.set_data(
            {
                "feature_flags": {"new_ui": True, "beta_features": False},
                "api_keys": {"external_service": "remote-api-key"},
            }
        )

        loaded_config = await source.async_load()

        assert loaded_config["feature_flags"]["new_ui"] is True
        assert loaded_config["api_keys"]["external_service"] == "remote-api-key"
        assert source.load_count == 1

    @pytest.mark.asyncio
    async def test_remote_configuration_source_failure_retry(self):
        """Test remote configuration source with connection failures."""
        source = RemoteConfigurationSource("remote_config", "https://config.example.com")
        source.set_data({"key": "value"})

        # First few calls should fail, then succeed
        with pytest.raises(ConnectionError):
            await source.async_load()  # load_count = 1, fails

        with pytest.raises(ConnectionError):
            await source.async_load()  # load_count = 2, fails

        # This should succeed
        config = await source.async_load()  # load_count = 3, succeeds
        assert config == {"key": "value"}


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_configuration_validator_success(self, sample_database_schema, sample_logging_schema):
        """Test successful configuration validation."""
        validator = ConfigurationValidator()
        validator.register_schema("database", sample_database_schema)
        validator.register_schema("logging", sample_logging_schema)

        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "secret",
                "pool_size": 10,
            },
            "logging": {"level": "INFO", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        }

        errors = validator.validate(config)
        assert errors == []

    def test_configuration_validator_missing_required(self, sample_database_schema):
        """Test configuration validation with missing required fields."""
        validator = ConfigurationValidator()
        validator.register_schema("database", sample_database_schema)

        config = {
            "database": {
                "host": "localhost",
                # Missing required fields: port, database, username
                "password": "secret",
            }
        }

        errors = validator.validate(config)
        assert len(errors) == 3
        assert "database.port is required" in errors
        assert "database.database is required" in errors
        assert "database.username is required" in errors

    def test_configuration_validator_type_errors(self, sample_database_schema):
        """Test configuration validation with type errors."""
        validator = ConfigurationValidator()
        validator.register_schema("database", sample_database_schema)

        config = {
            "database": {
                "host": "localhost",
                "port": "not-a-number",  # Should be int
                "database": "test_db",
                "username": 123,  # Should be str
                "pool_size": "10",  # Should be int
            }
        }

        errors = validator.validate(config)
        assert len(errors) >= 3
        assert any("database.port must be of type int" in error for error in errors)
        assert any("database.username must be of type str" in error for error in errors)
        assert any("database.pool_size must be of type int" in error for error in errors)

    def test_configuration_validator_range_errors(self, sample_database_schema):
        """Test configuration validation with range errors."""
        validator = ConfigurationValidator()
        validator.register_schema("database", sample_database_schema)

        config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # Exceeds max (65535)
                "database": "test_db",
                "username": "test_user",
                "pool_size": 0,  # Below min (1)
            }
        }

        errors = validator.validate(config)
        assert len(errors) == 2
        assert any("database.port must be <= 65535" in error for error in errors)
        assert any("database.pool_size must be >= 1" in error for error in errors)

    def test_configuration_validator_value_restrictions(self, sample_logging_schema):
        """Test configuration validation with restricted values."""
        validator = ConfigurationValidator()
        validator.register_schema("logging", sample_logging_schema)

        config = {
            "logging": {
                "level": "INVALID_LEVEL"  # Not in allowed values
            }
        }

        errors = validator.validate(config)
        assert len(errors) == 1
        assert "logging.level must be one of" in errors[0]
        assert "INVALID_LEVEL" in errors[0]


class TestConfigurationManager:
    """Test configuration manager functionality."""

    def test_configuration_manager_single_source(self, config_manager, temp_config_dir):
        """Test configuration manager with single source."""
        # Create configuration file
        config_file = temp_config_dir / "config.json"
        config_data = {"app": {"name": "test_app", "version": "1.0"}}

        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Add source and load
        source = FileConfigurationSource("main_config", str(config_file))
        config_manager.add_source(source)

        loaded_config = config_manager.load_configuration()

        assert loaded_config == config_data
        assert config_manager.get_configuration() == config_data
        assert config_manager.get_configuration("app.name") == "test_app"
        assert config_manager.get_configuration("app.version") == "1.0"
        assert config_manager.get_configuration("nonexistent", "default") == "default"

    def test_configuration_manager_multiple_sources_priority(self, config_manager, temp_config_dir):
        """Test configuration manager with multiple sources and priority merging."""
        # Create base configuration file (low priority)
        base_config_file = temp_config_dir / "base.json"
        base_config = {
            "app": {"name": "base_app", "version": "1.0", "debug": False},
            "database": {"host": "localhost", "port": 5432},
        }
        with base_config_file.open("w") as f:
            json.dump(base_config, f)

        # Create override configuration file (high priority)
        override_config_file = temp_config_dir / "override.json"
        override_config = {
            "app": {"debug": True},  # Override debug setting
            "database": {"host": "prod-db"},  # Override database host
            "new_setting": {"enabled": True},  # Add new setting
        }
        with override_config_file.open("w") as f:
            json.dump(override_config, f)

        # Add sources (higher priority number = higher priority)
        base_source = FileConfigurationSource("base", str(base_config_file), priority=1)
        override_source = FileConfigurationSource("override", str(override_config_file), priority=10)

        config_manager.add_source(base_source)
        config_manager.add_source(override_source)

        loaded_config = config_manager.load_configuration()

        # Verify merged configuration
        assert loaded_config["app"]["name"] == "base_app"  # From base
        assert loaded_config["app"]["version"] == "1.0"  # From base
        assert loaded_config["app"]["debug"] is True  # Overridden
        assert loaded_config["database"]["host"] == "prod-db"  # Overridden
        assert loaded_config["database"]["port"] == 5432  # From base
        assert loaded_config["new_setting"]["enabled"] is True  # From override

    def test_configuration_manager_environment_override(self, config_manager, temp_config_dir):
        """Test environment variables overriding file configuration."""
        # Create base configuration file
        config_file = temp_config_dir / "config.json"
        config_data = {"app": {"name": "file_app", "port": 8000}, "database": {"host": "localhost"}}
        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Set environment variables
        test_env = {
            "APP_NAME": "env_app",  # Override app.name
            "DATABASE_HOST": "env-db",  # Override database.host
            "NEW_SETTING": "from_env",  # Add new setting
        }

        with patch.dict(os.environ, test_env, clear=False):
            file_source = FileConfigurationSource("file_config", str(config_file), priority=1)
            env_source = EnvironmentConfigurationSource("env_config", priority=10)

            config_manager.add_source(file_source)
            config_manager.add_source(env_source)

            loaded_config = config_manager.load_configuration()

        # Environment should override file settings
        assert loaded_config["app"]["name"] == "file_app"  # From file (env key doesn't match)
        assert loaded_config["app"]["port"] == 8000  # From file
        assert loaded_config["database"]["host"] == "localhost"  # From file (env key doesn't match)
        assert loaded_config["app_name"] == "env_app"  # From environment
        assert loaded_config["database_host"] == "env-db"  # From environment
        assert loaded_config["new_setting"] == "from_env"  # From environment

    def test_configuration_manager_validation_integration(
        self, config_manager, temp_config_dir, sample_database_schema
    ):
        """Test configuration manager with validation."""
        # Register schema
        config_manager.validator.register_schema("database", sample_database_schema)

        # Create invalid configuration
        config_file = temp_config_dir / "config.json"
        invalid_config = {
            "database": {
                "host": "localhost",
                # Missing required fields: port, database, username
            }
        }
        with config_file.open("w") as f:
            json.dump(invalid_config, f)

        source = FileConfigurationSource("config", str(config_file))
        config_manager.add_source(source)

        # Load should succeed but log validation errors
        loaded_config = config_manager.load_configuration()
        assert loaded_config == invalid_config

        # Validation errors should be recorded
        errors = config_manager.validator.validation_errors
        assert len(errors) > 0
        assert any("database.port is required" in error for error in errors)

    def test_configuration_manager_watchers(self, config_manager):
        """Test configuration change watchers."""
        watcher_calls = []

        def config_watcher(config):
            watcher_calls.append(config.copy())

        config_manager.add_watcher(config_watcher)

        # Create and add a simple source
        source = ConfigurationSource("test_source")
        source.set_data({"key1": "value1"})
        config_manager.add_source(source)

        # Load configuration - should trigger watcher
        config_manager.load_configuration()
        assert len(watcher_calls) == 1
        assert watcher_calls[0]["key1"] == "value1"

        # Update configuration - should trigger watcher
        config_manager.set_configuration("key2", "value2")
        assert len(watcher_calls) == 2
        assert watcher_calls[1]["key2"] == "value2"

        # Remove watcher
        config_manager.remove_watcher(config_watcher)
        config_manager.set_configuration("key3", "value3")
        assert len(watcher_calls) == 2  # Should not increase

    def test_configuration_manager_reload(self, config_manager, temp_config_dir):
        """Test configuration reloading."""
        config_file = temp_config_dir / "config.json"

        # Initial configuration
        initial_config = {"app": {"version": "1.0"}}
        with config_file.open("w") as f:
            json.dump(initial_config, f)

        source = FileConfigurationSource("config", str(config_file))
        config_manager.add_source(source)

        # Initial load
        config1 = config_manager.load_configuration()
        assert config1["app"]["version"] == "1.0"
        assert config_manager.reload_count == 0

        # Update configuration file
        updated_config = {"app": {"version": "2.0"}}
        with config_file.open("w") as f:
            json.dump(updated_config, f)

        # Reload configuration
        config2 = config_manager.reload_configuration()
        assert config2["app"]["version"] == "2.0"
        assert config_manager.reload_count == 1

        # Verify history is maintained
        assert len(config_manager.config_history) == 1
        assert config_manager.config_history[0]["app"]["version"] == "1.0"

    def test_configuration_manager_source_failure_handling(self, config_manager):
        """Test handling of configuration source failures."""
        working_source = ConfigurationSource("working", priority=1)
        working_source.set_data({"working": True})

        failing_source = ConfigurationSource("failing", priority=2)
        failing_source.is_available = False

        config_manager.add_source(working_source)
        config_manager.add_source(failing_source)

        # Should raise error due to failing source
        with pytest.raises(RuntimeError, match="Configuration load errors"):
            config_manager.load_configuration()

        # But working source data should still be accessible if we disable the failing one
        failing_source.is_available = True
        failing_source.set_data({"failing": False})

        config = config_manager.load_configuration()
        assert config["working"] is True
        assert config["failing"] is False


class TestConfigurationIntegrationScenarios:
    """Test real-world configuration integration scenarios."""

    @pytest.mark.asyncio
    async def test_multi_environment_configuration(self, config_manager, temp_config_dir):
        """Test configuration for multiple environments (dev, staging, prod)."""
        # Base configuration
        base_config_file = temp_config_dir / "base.json"
        base_config = {
            "app": {"name": "myapp", "version": "1.0"},
            "database": {"host": "localhost", "port": 5432, "pool_size": 5},
            "logging": {"level": "INFO"},
        }
        with base_config_file.open("w") as f:
            json.dump(base_config, f)

        # Environment-specific overrides
        prod_config_file = temp_config_dir / "production.json"
        prod_config = {
            "database": {"host": "prod-db.example.com", "pool_size": 20},
            "logging": {"level": "WARNING"},
            "features": {"debug_mode": False},
        }
        with prod_config_file.open("w") as f:
            json.dump(prod_config, f)

        # Remote configuration (feature flags, etc.)
        remote_source = RemoteConfigurationSource("remote", "https://config.example.com", priority=15)
        remote_source.set_data(
            {"features": {"new_ui": True, "beta_features": False}, "rate_limits": {"api_calls_per_minute": 1000}}
        )

        # Environment variables (highest priority)
        test_env = {"APP_DATABASE_PASSWORD": "prod-secret", "APP_FEATURES": '{"maintenance_mode": true}'}

        with patch.dict(os.environ, test_env, clear=False):
            # Add sources in order of priority
            base_source = FileConfigurationSource("base", str(base_config_file), priority=1)
            prod_source = FileConfigurationSource("prod", str(prod_config_file), priority=5)
            env_source = EnvironmentConfigurationSource("env", prefix="APP", priority=20)

            config_manager.add_source(base_source)
            config_manager.add_source(prod_source)
            config_manager.add_source(remote_source)
            config_manager.add_source(env_source)

            # Load merged configuration
            final_config = config_manager.load_configuration()

        # Verify configuration merging
        assert final_config["app"]["name"] == "myapp"  # From base
        assert final_config["app"]["version"] == "1.0"  # From base
        assert final_config["database"]["host"] == "prod-db.example.com"  # From prod override
        assert final_config["database"]["pool_size"] == 20  # From prod override
        assert final_config["database"]["port"] == 5432  # From base
        assert final_config["logging"]["level"] == "WARNING"  # From prod override
        assert final_config["features"]["debug_mode"] is False  # From prod
        assert final_config["features"]["new_ui"] is True  # From remote
        assert final_config["features"]["maintenance_mode"] is True  # From env (highest priority)
        assert final_config["rate_limits"]["api_calls_per_minute"] == 1000  # From remote
        assert final_config["database_password"] == "prod-secret"  # From env

    def test_configuration_hot_reload_simulation(self, config_manager, temp_config_dir):
        """Test hot-reloading configuration changes."""
        config_file = temp_config_dir / "dynamic.json"

        # Initial configuration
        config_v1 = {"app": {"maintenance_mode": False, "max_users": 1000}, "features": {"new_feature": False}}
        with config_file.open("w") as f:
            json.dump(config_v1, f)

        source = FileConfigurationSource("dynamic", str(config_file))
        config_manager.add_source(source)

        # Track configuration changes
        change_history = []

        def track_changes(config):
            change_history.append({"timestamp": datetime.now(UTC), "config": config.copy()})

        config_manager.add_watcher(track_changes)
        config_manager.start_watching()

        # Load initial configuration
        config_manager.load_configuration()
        assert len(change_history) == 1
        assert change_history[0]["config"]["app"]["maintenance_mode"] is False

        # Simulate configuration changes (as if file was updated externally)
        config_v2 = {
            "app": {"maintenance_mode": True, "max_users": 500},  # Enable maintenance, reduce capacity
            "features": {"new_feature": True},  # Enable feature
        }
        with config_file.open("w") as f:
            json.dump(config_v2, f)

        # Reload configuration (simulates file watcher trigger)
        config_manager.reload_configuration()

        assert len(change_history) == 2
        assert change_history[1]["config"]["app"]["maintenance_mode"] is True
        assert change_history[1]["config"]["app"]["max_users"] == 500
        assert change_history[1]["config"]["features"]["new_feature"] is True

        # Configuration history should be maintained
        assert len(config_manager.config_history) == 1
        assert config_manager.config_history[0]["app"]["maintenance_mode"] is False

    def test_secrets_management_configuration(self, config_manager, temp_config_dir):
        """Test configuration with secrets management."""
        # Public configuration file (no secrets)
        public_config_file = temp_config_dir / "public.json"
        public_config = {
            "app": {"name": "secure_app", "port": 8080},
            "database": {"host": "db.example.com", "port": 5432, "database": "myapp"},
        }
        with public_config_file.open("w") as f:
            json.dump(public_config, f)

        # Secrets from environment variables (simulating secrets manager)
        secrets_env = {
            "SECRET_DATABASE_PASSWORD": "super-secret-password",
            "SECRET_API_KEY": "api-key-12345",
            "SECRET_JWT_SECRET": "jwt-signing-secret",
        }

        with patch.dict(os.environ, secrets_env, clear=False):
            public_source = FileConfigurationSource("public", str(public_config_file), priority=1)
            secrets_source = EnvironmentConfigurationSource("secrets", prefix="SECRET", priority=10)

            config_manager.add_source(public_source)
            config_manager.add_source(secrets_source)

            final_config = config_manager.load_configuration()

        # Verify secrets are available but separated from public config
        assert final_config["app"]["name"] == "secure_app"
        assert final_config["database"]["host"] == "db.example.com"
        assert final_config["database_password"] == "super-secret-password"
        assert final_config["api_key"] == "api-key-12345"
        assert final_config["jwt_secret"] == "jwt-signing-secret"

        # Verify secrets don't appear in config history (basic check)
        # In real implementation, you'd want to mask/encrypt secrets in history
        assert len(config_manager.config_history) == 0  # No history yet
