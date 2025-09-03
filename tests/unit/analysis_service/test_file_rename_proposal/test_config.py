"""Tests for FileRenameProposalConfig."""

import os
from unittest.mock import patch

import pytest

from services.analysis_service.src.file_rename_proposal.config import FileRenameProposalConfig


class TestFileRenameProposalConfig:
    """Test FileRenameProposalConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FileRenameProposalConfig()

        # Check default patterns
        assert "mp3" in config.default_patterns
        assert config.default_patterns["mp3"] == "{artist} - {title}"
        assert config.default_patterns["default"] == "{artist} - {title}"

        # Check filesystem limits
        assert config.max_filename_length == 255
        assert config.max_path_length == 4096

        # Check invalid characters
        assert config.invalid_chars_windows == '<>:"|?*'
        assert config.invalid_chars_unix == "\x00"
        assert config.replacement_char == "_"

        # Check confidence weights
        assert config.confidence_weights["metadata_completeness"] == 0.4
        assert config.confidence_weights["metadata_quality"] == 0.3
        assert config.confidence_weights["pattern_match"] == 0.2
        assert config.confidence_weights["conflicts"] == 0.1

        # Check batch settings
        assert config.batch_size == 100
        assert config.max_batch_size == 1000

        # Check feature flags
        assert config.enable_proposal_generation is True
        assert config.enable_conflict_detection is True
        assert config.enable_unicode_normalization is True

        # Check retention
        assert config.proposal_retention_days == 30

    def test_from_env_with_values(self):
        """Test configuration from environment variables."""
        env_vars = {
            "RENAME_MAX_FILENAME_LENGTH": "128",
            "RENAME_MAX_PATH_LENGTH": "2048",
            "RENAME_BATCH_SIZE": "50",
            "RENAME_PROPOSAL_RETENTION_DAYS": "60",
            "RENAME_ENABLE_PROPOSAL_GENERATION": "false",
            "RENAME_ENABLE_CONFLICT_DETECTION": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = FileRenameProposalConfig.from_env()

            assert config.max_filename_length == 128
            assert config.max_path_length == 2048
            assert config.batch_size == 50
            assert config.proposal_retention_days == 60
            assert config.enable_proposal_generation is False
            assert config.enable_conflict_detection is True

    def test_from_env_without_values(self):
        """Test configuration without environment variables uses defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = FileRenameProposalConfig.from_env()

            assert config.max_filename_length == 255
            assert config.max_path_length == 4096
            assert config.batch_size == 100
            assert config.proposal_retention_days == 30
            assert config.enable_proposal_generation is True

    def test_from_env_with_invalid_values(self):
        """Test configuration handles invalid environment values."""
        env_vars = {
            "RENAME_MAX_FILENAME_LENGTH": "invalid",
            "RENAME_ENABLE_PROPOSAL_GENERATION": "yes",  # Should be "true" or "false"
        }

        with patch.dict(os.environ, env_vars), pytest.raises(ValueError):
            FileRenameProposalConfig.from_env()

    def test_pattern_customization(self):
        """Test that patterns can be customized."""
        config = FileRenameProposalConfig()

        # Modify a pattern
        config.default_patterns["mp3"] = "{artist} - {album} - {title}"
        assert config.default_patterns["mp3"] == "{artist} - {album} - {title}"

    def test_confidence_weights_sum(self):
        """Test that confidence weights are properly balanced."""
        config = FileRenameProposalConfig()

        total_weight = sum(config.confidence_weights.values())
        assert (
            abs(total_weight - 1.0) < 1e-10
        )  # Weights should sum to 1.0 for proper scoring (within floating point precision)
