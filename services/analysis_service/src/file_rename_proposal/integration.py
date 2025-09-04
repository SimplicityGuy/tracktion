"""Integration module for file rename proposal service with analysis pipeline."""

from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import structlog

from shared.core_types.src.models import RenameProposal
from shared.core_types.src.rename_proposal_repository import RenameProposalRepository
from shared.core_types.src.repositories import MetadataRepository, RecordingRepository

from .batch_processor import BatchProcessor
from .confidence_scorer import ConfidenceScorer
from .config import FileRenameProposalConfig
from .conflict_detector import ConflictDetector
from .pattern_manager import PatternManager
from .proposal_generator import ProposalGenerator
from .validator import FilesystemValidator

logger = structlog.get_logger(__name__)


class FileRenameProposalIntegration:
    """Integration layer for file rename proposal service with analysis pipeline."""

    def __init__(
        self,
        proposal_repo: RenameProposalRepository,
        recording_repo: RecordingRepository,
        metadata_repo: MetadataRepository,
        config: FileRenameProposalConfig | None = None,
    ) -> None:
        """Initialize the integration service.

        Args:
            proposal_repo: Repository for rename proposals
            recording_repo: Repository for recordings
            metadata_repo: Repository for metadata
            config: Configuration for file rename proposals
        """
        self.proposal_repo = proposal_repo
        self.recording_repo = recording_repo
        self.metadata_repo = metadata_repo
        self.config = config or FileRenameProposalConfig()

        # Initialize components
        self.pattern_manager = PatternManager(self.config)
        self.validator = FilesystemValidator(self.config)
        self.conflict_detector = ConflictDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.proposal_generator = ProposalGenerator(self.config, self.pattern_manager, self.validator)

        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            proposal_generator=self.proposal_generator,
            conflict_detector=self.conflict_detector,
            confidence_scorer=self.confidence_scorer,
            proposal_repo=self.proposal_repo,
            recording_repo=self.recording_repo,
            metadata_repo=self.metadata_repo,
        )

        self.logger = logger

        # Configuration options
        self.auto_generate_proposals = self.config.auto_generate_proposals
        self.auto_approve_threshold = self.config.auto_approve_threshold

    def process_recording_metadata(
        self, recording_id: UUID, metadata: dict[str, str], correlation_id: str
    ) -> str | None:
        """Process extracted metadata and generate rename proposal if enabled.

        Args:
            recording_id: UUID of the recording
            metadata: Extracted metadata dictionary
            correlation_id: Correlation ID for tracing

        Returns:
            Optional proposal ID if a proposal was generated
        """
        if not self.auto_generate_proposals:
            logger.debug(
                "Auto-generation disabled, skipping proposal generation",
                correlation_id=correlation_id,
                recording_id=str(recording_id),
            )
            return None

        try:
            # Get recording details
            recording = self.recording_repo.get_by_id(recording_id)
            if not recording:
                logger.warning(
                    f"Recording {recording_id} not found, cannot generate proposal",
                    correlation_id=correlation_id,
                )
                return None

            # Validate recording has required fields
            if not recording.file_name or not recording.file_path:
                logger.warning(
                    f"Recording {recording_id} missing file_name or file_path",
                    correlation_id=correlation_id,
                )
                return None

            # Extract file extension
            path_obj = Path(recording.file_name)
            file_extension = path_obj.suffix[1:].lower() if path_obj.suffix else "mp3"

            # Generate proposal
            proposal = self.proposal_generator.generate_proposal(
                recording_id=recording_id,
                original_path=recording.file_path,
                metadata=metadata,
                file_extension=file_extension,
            )

            # Get directory contents for conflict detection
            directory_path = Path(recording.file_path).parent  # Already validated above
            existing_files = set()
            if directory_path.exists():
                try:
                    existing_files = {f.name for f in directory_path.iterdir()}
                except (OSError, PermissionError) as e:
                    logger.warning(
                        f"Could not list directory {directory_path}: {e}",
                        correlation_id=correlation_id,
                    )

            # Get other pending proposals for conflict detection
            other_proposals = [
                {
                    "full_proposed_path": p.full_proposed_path,
                    "recording_id": str(p.recording_id),
                }
                for p in self.proposal_repo.get_pending_proposals()
                if p.recording_id != recording_id
            ]

            # Detect conflicts
            conflicts_result = self.conflict_detector.detect_conflicts(
                proposal.full_proposed_path, existing_files, other_proposals
            )
            conflicts = conflicts_result["conflicts"]
            warnings = conflicts_result["warnings"]

            # Attempt conflict resolution if needed
            resolved_path = proposal.full_proposed_path
            resolved_filename = proposal.proposed_filename
            if conflicts:
                alternative = self.conflict_detector.resolve_conflicts(proposal.full_proposed_path, conflicts)
                if alternative:
                    resolved_path = alternative
                    resolved_filename = Path(alternative).name

                    # Re-check conflicts for the alternative
                    conflicts_result = self.conflict_detector.detect_conflicts(
                        alternative, existing_files, other_proposals
                    )
                    conflicts = conflicts_result["conflicts"]
                    warnings = conflicts_result["warnings"]

            # Calculate confidence score
            confidence, components = self.confidence_scorer.calculate_confidence(
                metadata=cast("dict[str, str | None]", metadata),
                original_filename=recording.file_name,  # Already validated above
                proposed_filename=resolved_filename,
                conflicts=conflicts,
                warnings=warnings,
                pattern_used=proposal.pattern_used,
                source=proposal.metadata_source,
            )

            # Determine status
            if conflicts:
                status = "rejected"  # Auto-reject if conflicts remain
            elif confidence >= self.auto_approve_threshold:
                status = "approved"
            else:
                status = "pending"

            # Create proposal in database
            created_proposal = self.proposal_repo.create(
                recording_id=recording_id,
                original_path=str(Path(recording.file_path).parent),  # Already validated above
                original_filename=recording.file_name,
                proposed_filename=resolved_filename,
                full_proposed_path=resolved_path,
                confidence_score=confidence,
                status=status,
                conflicts=conflicts if conflicts else None,
                warnings=warnings if warnings else None,
                metadata_source=proposal.metadata_source,
                pattern_used=proposal.pattern_used,
            )

            logger.info(
                f"Generated rename proposal {created_proposal.id} for {recording.file_name}",
                correlation_id=correlation_id,
                recording_id=str(recording_id),
                confidence=confidence,
                status=status,
                conflicts_count=len(conflicts) if conflicts else 0,
                warnings_count=len(warnings) if warnings else 0,
            )

            return str(created_proposal.id)

        except Exception as e:
            logger.error(
                f"Failed to generate rename proposal: {e}",
                correlation_id=correlation_id,
                recording_id=str(recording_id),
                exc_info=True,
            )
            return None

    def process_batch_recordings(self, recording_ids: list[UUID], correlation_id: str) -> str | None:
        """Process multiple recordings in batch.

        Args:
            recording_ids: List of recording UUIDs to process
            correlation_id: Correlation ID for tracing

        Returns:
            Optional batch job ID if batch processing was started
        """
        if not self.auto_generate_proposals:
            logger.debug(
                "Auto-generation disabled, skipping batch proposal generation",
                correlation_id=correlation_id,
            )
            return None

        try:
            # Submit batch job
            job = self.batch_processor.submit_batch_job(
                recording_ids=recording_ids,
                job_id=f"analysis_batch_{correlation_id}",
                max_workers=4,
                chunk_size=50,
                auto_approve_threshold=self.auto_approve_threshold,
                enable_conflict_resolution=True,
            )

            # Start processing in background
            # Note: In a real implementation, this should be done asynchronously
            logger.info(
                f"Submitted batch job {job.job_id} for {len(recording_ids)} recordings",
                correlation_id=correlation_id,
                job_id=job.job_id,
            )

            return job.job_id

        except Exception as e:
            logger.error(
                f"Failed to submit batch job: {e}",
                correlation_id=correlation_id,
                exc_info=True,
            )
            return None

    def get_proposal_status(self, recording_id: UUID) -> dict[str, str] | None:
        """Get the status of rename proposals for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            Dictionary with proposal status information or None if no proposals
        """
        try:
            proposals = self.proposal_repo.get_by_recording(recording_id)
            if not proposals:
                return None

            # Get the most recent proposal - created_at is non-nullable in the model
            def get_created_at(proposal: RenameProposal) -> datetime:
                return cast("datetime", proposal.created_at)

            latest_proposal = max(proposals, key=get_created_at)

            return {
                "proposal_id": str(latest_proposal.id),
                "status": str(latest_proposal.status),  # Ensure string type
                "proposed_filename": str(latest_proposal.proposed_filename),  # Ensure string type
                "confidence_score": (
                    str(latest_proposal.confidence_score) if latest_proposal.confidence_score else "0"
                ),
                "created_at": latest_proposal.created_at.isoformat() if latest_proposal.created_at else "",
            }

        except Exception as e:
            logger.error(f"Failed to get proposal status for recording {recording_id}: {e}")
            return None

    def cleanup_old_proposals(self, days: int = 30) -> int:
        """Clean up old proposals.

        Args:
            days: Number of days to keep proposals

        Returns:
            Number of proposals cleaned up
        """
        try:
            result = self.proposal_repo.cleanup_old_proposals(days)
            return int(result)  # Ensure explicit int return type for mypy
        except Exception as e:
            logger.error(f"Failed to cleanup old proposals: {e}")
            return 0
