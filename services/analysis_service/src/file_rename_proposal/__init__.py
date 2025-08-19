"""File Rename Proposal Service.

This module provides functionality to generate filename proposals based on
extracted metadata. It does NOT perform actual file renaming operations.
"""

from .config import FileRenameProposalConfig
from .pattern_manager import PatternManager
from .proposal_generator import ProposalGenerator
from .validator import FilesystemValidator

__all__ = [
    "FileRenameProposalConfig",
    "PatternManager",
    "ProposalGenerator",
    "FilesystemValidator",
]
