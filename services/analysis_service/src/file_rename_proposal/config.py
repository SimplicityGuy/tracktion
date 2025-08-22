"""Configuration for File Rename Proposal Service."""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FileRenameProposalConfig:
    """Configuration for the file rename proposal service."""

    # Default naming patterns by file type
    default_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "mp3": "{artist} - {title}",
            "flac": "{artist} - {album} - {track:02d} - {title}",
            "wav": "{artist} - {title} - {bpm}BPM",
            "m4a": "{artist} - {title}",
            "ogg": "{artist} - {title}",
            "oga": "{artist} - {title}",
            "default": "{artist} - {title}",
        }
    )

    # Filesystem limits
    max_filename_length: int = 255
    max_path_length: int = 4096  # Linux default

    # Invalid characters for different OS
    invalid_chars_windows: str = '<>:"|?*'
    invalid_chars_unix: str = "\x00"
    replacement_char: str = "_"

    # Confidence scoring weights
    confidence_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "metadata_completeness": 0.4,
            "metadata_quality": 0.3,
            "pattern_match": 0.2,
            "conflicts": 0.1,
        }
    )

    # Batch processing
    batch_size: int = 100
    max_batch_size: int = 1000

    # Feature flags
    enable_proposal_generation: bool = True
    enable_conflict_detection: bool = True
    enable_unicode_normalization: bool = True

    # Integration with analysis pipeline
    auto_generate_proposals: bool = True

    # Auto-approval threshold for confidence score
    auto_approve_threshold: float = 0.9

    # Database settings
    proposal_retention_days: int = 30

    @classmethod
    def from_env(cls) -> "FileRenameProposalConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if max_filename := os.getenv("RENAME_MAX_FILENAME_LENGTH"):
            config.max_filename_length = int(max_filename)

        if max_path := os.getenv("RENAME_MAX_PATH_LENGTH"):
            config.max_path_length = int(max_path)

        if batch_size := os.getenv("RENAME_BATCH_SIZE"):
            config.batch_size = int(batch_size)

        if retention := os.getenv("RENAME_PROPOSAL_RETENTION_DAYS"):
            config.proposal_retention_days = int(retention)

        # Feature flags
        if enable_gen := os.getenv("RENAME_ENABLE_PROPOSAL_GENERATION"):
            config.enable_proposal_generation = enable_gen.lower() == "true"

        if enable_conflict := os.getenv("RENAME_ENABLE_CONFLICT_DETECTION"):
            config.enable_conflict_detection = enable_conflict.lower() == "true"

        if auto_generate := os.getenv("RENAME_AUTO_GENERATE_PROPOSALS"):
            config.auto_generate_proposals = auto_generate.lower() == "true"

        if auto_approve_threshold := os.getenv("RENAME_AUTO_APPROVE_THRESHOLD"):
            config.auto_approve_threshold = float(auto_approve_threshold)

        return config
