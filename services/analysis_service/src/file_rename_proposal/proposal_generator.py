"""Core proposal generation logic for file renaming."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from .config import FileRenameProposalConfig
from .pattern_manager import PatternManager
from .validator import FilesystemValidator

logger = logging.getLogger(__name__)


@dataclass
class RenameProposal:
    """Represents a file rename proposal."""

    recording_id: UUID
    original_path: str
    original_filename: str
    proposed_filename: str
    full_proposed_path: str
    confidence_score: float
    status: str = "pending"
    conflicts: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.conflicts is None:
            self.conflicts = []
        if self.warnings is None:
            self.warnings = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class ProposalGenerator:
    """Generates filename proposals based on metadata and patterns."""

    def __init__(
        self,
        config: FileRenameProposalConfig,
        pattern_manager: PatternManager,
        validator: FilesystemValidator,
    ):
        """Initialize the proposal generator.

        Args:
            config: Configuration for the service
            pattern_manager: Manager for naming patterns
            validator: Filesystem validator
        """
        self.config = config
        self.pattern_manager = pattern_manager
        self.validator = validator

    def generate_proposal(
        self,
        recording_id: UUID,
        original_path: str,
        metadata: Dict[str, str],
        file_extension: str,
    ) -> RenameProposal:
        """Generate a rename proposal for a single file.

        Args:
            recording_id: UUID of the recording
            original_path: Original file path
            metadata: Metadata dictionary for the file
            file_extension: File extension (e.g., 'mp3', 'flac')

        Returns:
            RenameProposal object with the proposed filename
        """
        # Extract original filename from path
        original_filename = os.path.basename(original_path)
        original_dir = os.path.dirname(original_path)

        # Generate proposed filename using pattern
        proposed_filename = self.pattern_manager.apply_pattern(metadata, file_extension)

        # Validate and sanitize the proposed filename
        sanitized_filename = self.validator.sanitize_filename(proposed_filename)

        # Add file extension if not present
        if not sanitized_filename.endswith(f".{file_extension}"):
            sanitized_filename = f"{sanitized_filename}.{file_extension}"

        # Create full proposed path
        full_proposed_path = os.path.join(original_dir, sanitized_filename)

        # Calculate confidence score
        confidence_score = self._calculate_confidence(metadata, proposed_filename, sanitized_filename)

        # Create proposal
        proposal = RenameProposal(
            recording_id=recording_id,
            original_path=original_path,
            original_filename=original_filename,
            proposed_filename=sanitized_filename,
            full_proposed_path=full_proposed_path,
            confidence_score=confidence_score,
        )

        # Check for potential issues
        self._check_for_issues(proposal)

        return proposal

    def generate_batch_proposals(self, recordings: List[Tuple[UUID, str, Dict[str, str], str]]) -> List[RenameProposal]:
        """Generate proposals for multiple files.

        Args:
            recordings: List of tuples (recording_id, path, metadata, extension)

        Returns:
            List of RenameProposal objects
        """
        proposals = []
        for recording_id, path, metadata, extension in recordings:
            try:
                proposal = self.generate_proposal(recording_id, path, metadata, extension)
                proposals.append(proposal)
            except Exception as e:
                logger.error(f"Failed to generate proposal for {path}: {e}")
                # Create a failed proposal with error in warnings
                proposal = RenameProposal(
                    recording_id=recording_id,
                    original_path=path,
                    original_filename=os.path.basename(path),
                    proposed_filename="",
                    full_proposed_path="",
                    confidence_score=0.0,
                    status="failed",
                    warnings=[f"Failed to generate proposal: {str(e)}"],
                )
                proposals.append(proposal)

        return proposals

    def _calculate_confidence(self, metadata: Dict[str, str], proposed: str, sanitized: str) -> float:
        """Calculate confidence score for a proposal.

        Args:
            metadata: Metadata dictionary
            proposed: Original proposed filename
            sanitized: Sanitized filename

        Returns:
            Confidence score between 0.0 and 1.0
        """
        weights = self.config.confidence_weights
        score = 0.0

        # Metadata completeness
        required_fields = ["artist", "title"]
        present_fields = sum(1 for field in required_fields if metadata.get(field))
        completeness_score = present_fields / len(required_fields)
        score += completeness_score * weights["metadata_completeness"]

        # Metadata quality (check if values are not empty or default)
        quality_score = 1.0
        for field in required_fields:
            value = metadata.get(field, "")
            if value.lower() in ["unknown", "untitled", ""]:
                quality_score -= 0.5
        quality_score = max(0.0, quality_score)
        score += quality_score * weights["metadata_quality"]

        # Pattern match success (did we have to sanitize much?)
        if proposed == sanitized:
            pattern_score = 1.0
        else:
            # Calculate similarity between proposed and sanitized
            pattern_score = 0.8  # Simplified for now
        score += pattern_score * weights["pattern_match"]

        # No conflicts detected yet (will be updated later)
        score += 1.0 * weights["conflicts"]

        return min(1.0, max(0.0, score))

    def _check_for_issues(self, proposal: RenameProposal) -> None:
        """Check for potential issues with the proposal.

        Args:
            proposal: The proposal to check
        """
        # Ensure warnings list exists
        if proposal.warnings is None:
            proposal.warnings = []

        # Check filename length
        if len(proposal.proposed_filename) > self.config.max_filename_length:
            proposal.warnings.append(f"Filename exceeds maximum length ({self.config.max_filename_length} chars)")

        # Check full path length
        if len(proposal.full_proposed_path) > self.config.max_path_length:
            proposal.warnings.append(f"Full path exceeds maximum length ({self.config.max_path_length} chars)")

        # Check if filename was significantly altered during sanitization
        proposed_base = os.path.splitext(proposal.proposed_filename)[0]
        if proposed_base == self.config.replacement_char * len(proposed_base):
            proposal.warnings.append("Proposed filename consists entirely of replacement characters")
