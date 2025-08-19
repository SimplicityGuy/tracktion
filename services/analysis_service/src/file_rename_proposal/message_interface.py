"""Message-based interface for file rename proposal service."""

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, cast
from uuid import UUID, uuid4

from .batch_processor import BatchProcessor
from .proposal_generator import ProposalGenerator
from .conflict_detector import ConflictDetector
from .confidence_scorer import ConfidenceScorer
from shared.core_types.src.rename_proposal_repository import RenameProposalRepository
from shared.core_types.src.repositories import RecordingRepository

logger = logging.getLogger(__name__)


class MessageTypes:
    """Message type constants."""

    # Request types
    GENERATE_PROPOSAL = "generate_proposal"
    BATCH_PROCESS = "batch_process"
    GET_PROPOSAL = "get_proposal"
    UPDATE_PROPOSAL = "update_proposal"
    GET_BATCH_STATUS = "get_batch_status"
    CANCEL_BATCH = "cancel_batch"
    GET_STATISTICS = "get_statistics"
    CLEANUP_OLD_PROPOSALS = "cleanup_old_proposals"

    # Response types
    PROPOSAL_GENERATED = "proposal_generated"
    BATCH_SUBMITTED = "batch_submitted"
    PROPOSAL_RETRIEVED = "proposal_retrieved"
    PROPOSAL_UPDATED = "proposal_updated"
    BATCH_STATUS = "batch_status"
    BATCH_CANCELLED = "batch_cancelled"
    STATISTICS = "statistics"
    CLEANUP_COMPLETED = "cleanup_completed"
    ERROR = "error"


class RenameProposalMessageInterface:
    """Message-based interface for file rename proposal operations."""

    def __init__(
        self,
        proposal_generator: ProposalGenerator,
        conflict_detector: ConflictDetector,
        confidence_scorer: ConfidenceScorer,
        proposal_repo: RenameProposalRepository,
        recording_repo: RecordingRepository,
        batch_processor: Optional[BatchProcessor] = None,
    ) -> None:
        """Initialize message interface.

        Args:
            proposal_generator: Proposal generation service
            conflict_detector: Conflict detection service
            confidence_scorer: Confidence scoring service
            proposal_repo: Proposal repository
            recording_repo: Recording repository
            batch_processor: Optional batch processor (will create if not provided)
        """
        self.proposal_generator = proposal_generator
        self.conflict_detector = conflict_detector
        self.confidence_scorer = confidence_scorer
        self.proposal_repo = proposal_repo
        self.recording_repo = recording_repo

        if batch_processor:
            self.batch_processor = batch_processor
        else:
            self.batch_processor = BatchProcessor(
                proposal_generator=proposal_generator,
                conflict_detector=conflict_detector,
                confidence_scorer=confidence_scorer,
                proposal_repo=proposal_repo,
                recording_repo=recording_repo,
            )

        self.logger = logger

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return response.

        Args:
            message: Input message dictionary

        Returns:
            Response message dictionary
        """
        try:
            message_type = message.get("type")
            request_id = message.get("request_id", str(uuid4()))

            if not message_type:
                return self._error_response(request_id, "Missing message type", "INVALID_MESSAGE")

            handler = self._get_message_handler(message_type)
            if not handler:
                return self._error_response(request_id, f"Unknown message type: {message_type}", "UNKNOWN_MESSAGE_TYPE")

            response = cast(Dict[str, Any], handler(message, request_id))
            response["request_id"] = request_id
            response["timestamp"] = datetime.utcnow().isoformat()

            return response

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return self._error_response(message.get("request_id", str(uuid4())), str(e), "PROCESSING_ERROR")

    def _get_message_handler(self, message_type: str) -> Optional[Callable]:
        """Get handler function for message type.

        Args:
            message_type: Type of message

        Returns:
            Handler function or None
        """
        handlers = {
            MessageTypes.GENERATE_PROPOSAL: self._handle_generate_proposal,
            MessageTypes.BATCH_PROCESS: self._handle_batch_process,
            MessageTypes.GET_PROPOSAL: self._handle_get_proposal,
            MessageTypes.UPDATE_PROPOSAL: self._handle_update_proposal,
            MessageTypes.GET_BATCH_STATUS: self._handle_get_batch_status,
            MessageTypes.CANCEL_BATCH: self._handle_cancel_batch,
            MessageTypes.GET_STATISTICS: self._handle_get_statistics,
            MessageTypes.CLEANUP_OLD_PROPOSALS: self._handle_cleanup_old_proposals,
        }

        return handlers.get(message_type)

    def _handle_generate_proposal(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle proposal generation request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            recording_id = message.get("recording_id")
            if not recording_id:
                return self._error_response(request_id, "Missing recording_id", "MISSING_PARAMETER")

            # Convert string UUID to UUID object if needed
            if isinstance(recording_id, str):
                recording_id = UUID(recording_id)

            # Get recording
            recording = self.recording_repo.get_by_id(recording_id)
            if not recording:
                return self._error_response(request_id, f"Recording {recording_id} not found", "RECORDING_NOT_FOUND")

            # Generate proposal - need to get metadata separately
            # For now, use empty metadata and derive file extension
            file_extension = (
                os.path.splitext(recording.file_name)[1][1:].lower() if "." in recording.file_name else "mp3"
            )
            proposal = self.proposal_generator.generate_proposal(
                recording_id=recording_id,
                original_path=recording.file_path,
                metadata={},  # TODO: Get actual metadata from MetadataRepository
                file_extension=file_extension,
            )
            if not proposal:
                return self._error_response(request_id, "Failed to generate proposal", "GENERATION_FAILED")

            # Detect conflicts
            conflicts_result = self.conflict_detector.detect_conflicts(
                proposal.full_proposed_path,
                set(),  # Would need directory contents in real scenario
                [],  # Would need other proposals in real scenario
            )

            # Calculate confidence
            confidence, components = self.confidence_scorer.calculate_confidence(
                metadata={},  # TODO: Use actual metadata
                original_filename=recording.file_name,
                proposed_filename=proposal.proposed_filename,
                conflicts=conflicts_result["conflicts"],
                warnings=conflicts_result["warnings"],
                pattern_used=proposal.pattern_used,
                source=proposal.metadata_source,
            )

            # Create proposal in database
            created_proposal = self.proposal_repo.create(
                recording_id=recording_id,
                original_path=os.path.dirname(recording.file_path),
                original_filename=recording.file_name,
                proposed_filename=proposal.proposed_filename,
                full_proposed_path=proposal.full_proposed_path,
                confidence_score=confidence,
                status="pending",
                conflicts=conflicts_result["conflicts"] if conflicts_result["conflicts"] else None,
                warnings=conflicts_result["warnings"] if conflicts_result["warnings"] else None,
                metadata_source=proposal.metadata_source,
                pattern_used=proposal.pattern_used,
            )

            return {
                "type": MessageTypes.PROPOSAL_GENERATED,
                "proposal": {
                    "id": str(created_proposal.id),
                    "recording_id": str(created_proposal.recording_id),
                    "original_filename": created_proposal.original_filename,
                    "proposed_filename": created_proposal.proposed_filename,
                    "full_proposed_path": created_proposal.full_proposed_path,
                    "confidence_score": float(created_proposal.confidence_score)
                    if created_proposal.confidence_score
                    else None,
                    "status": created_proposal.status,
                    "conflicts": created_proposal.conflicts or [],
                    "warnings": created_proposal.warnings or [],
                    "confidence_components": components,
                    "created_at": created_proposal.created_at.isoformat() if created_proposal.created_at else None,
                },
            }

        except Exception as e:
            logger.error(f"Proposal generation error: {e}")
            return self._error_response(request_id, str(e), "GENERATION_ERROR")

    def _handle_batch_process(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle batch processing request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            recording_ids = message.get("recording_ids", [])
            if not recording_ids:
                return self._error_response(request_id, "Missing recording_ids", "MISSING_PARAMETER")

            # Convert string UUIDs to UUID objects
            recording_uuids = []
            for rid in recording_ids:
                if isinstance(rid, str):
                    recording_uuids.append(UUID(rid))
                else:
                    recording_uuids.append(rid)

            # Extract batch options
            options = message.get("options", {})
            job_id = options.get("job_id")
            max_workers = options.get("max_workers", 4)
            chunk_size = options.get("chunk_size", 100)
            auto_approve_threshold = options.get("auto_approve_threshold", 0.9)
            enable_conflict_resolution = options.get("enable_conflict_resolution", True)

            # Submit batch job
            job = self.batch_processor.submit_batch_job(
                recording_ids=recording_uuids,
                job_id=job_id,
                max_workers=max_workers,
                chunk_size=chunk_size,
                auto_approve_threshold=auto_approve_threshold,
                enable_conflict_resolution=enable_conflict_resolution,
            )

            # Start processing if requested
            if options.get("start_immediately", True):
                # Process in background (would need async handling in real implementation)
                try:
                    self.batch_processor.process_batch_job(job.job_id)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")

            return {
                "type": MessageTypes.BATCH_SUBMITTED,
                "job": {
                    "job_id": job.job_id,
                    "status": job.status,
                    "total_recordings": job.total_recordings,
                    "created_at": job.created_at.isoformat(),
                    "options": job.options,
                },
            }

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return self._error_response(request_id, str(e), "BATCH_ERROR")

    def _handle_get_proposal(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle get proposal request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            proposal_id = message.get("proposal_id")
            recording_id = message.get("recording_id")

            if proposal_id:
                # Get by proposal ID
                if isinstance(proposal_id, str):
                    proposal_id = UUID(proposal_id)

                proposal = self.proposal_repo.get(proposal_id)
                if not proposal:
                    return self._error_response(request_id, f"Proposal {proposal_id} not found", "PROPOSAL_NOT_FOUND")

                proposals = [proposal]

            elif recording_id:
                # Get by recording ID
                if isinstance(recording_id, str):
                    recording_id = UUID(recording_id)

                status_filter = message.get("status")
                proposals = self.proposal_repo.get_by_recording(recording_id, status_filter)

            else:
                return self._error_response(request_id, "Must provide proposal_id or recording_id", "MISSING_PARAMETER")

            # Format proposals
            proposal_data = []
            for proposal in proposals:
                proposal_data.append(
                    {
                        "id": str(proposal.id),
                        "recording_id": str(proposal.recording_id),
                        "original_filename": proposal.original_filename,
                        "proposed_filename": proposal.proposed_filename,
                        "full_proposed_path": proposal.full_proposed_path,
                        "confidence_score": float(proposal.confidence_score) if proposal.confidence_score else None,
                        "status": proposal.status,
                        "conflicts": proposal.conflicts or [],
                        "warnings": proposal.warnings or [],
                        "metadata_source": proposal.metadata_source,
                        "pattern_used": proposal.pattern_used,
                        "created_at": proposal.created_at.isoformat() if proposal.created_at else None,
                        "updated_at": proposal.updated_at.isoformat() if proposal.updated_at else None,
                    }
                )

            return {"type": MessageTypes.PROPOSAL_RETRIEVED, "proposals": proposal_data, "count": len(proposal_data)}

        except Exception as e:
            logger.error(f"Get proposal error: {e}")
            return self._error_response(request_id, str(e), "RETRIEVAL_ERROR")

    def _handle_update_proposal(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle update proposal request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            proposal_id = message.get("proposal_id")
            if not proposal_id:
                return self._error_response(request_id, "Missing proposal_id", "MISSING_PARAMETER")

            if isinstance(proposal_id, str):
                proposal_id = UUID(proposal_id)

            # Extract update fields
            updates = message.get("updates", {})
            if not updates:
                return self._error_response(request_id, "No updates provided", "MISSING_PARAMETER")

            # Update proposal
            updated_proposal = self.proposal_repo.update(proposal_id, **updates)
            if not updated_proposal:
                return self._error_response(request_id, f"Proposal {proposal_id} not found", "PROPOSAL_NOT_FOUND")

            return {
                "type": MessageTypes.PROPOSAL_UPDATED,
                "proposal": {
                    "id": str(updated_proposal.id),
                    "recording_id": str(updated_proposal.recording_id),
                    "status": updated_proposal.status,
                    "confidence_score": float(updated_proposal.confidence_score)
                    if updated_proposal.confidence_score
                    else None,
                    "updated_at": updated_proposal.updated_at.isoformat() if updated_proposal.updated_at else None,
                },
            }

        except Exception as e:
            logger.error(f"Update proposal error: {e}")
            return self._error_response(request_id, str(e), "UPDATE_ERROR")

    def _handle_get_batch_status(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle get batch status request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            job_id = message.get("job_id")

            if job_id:
                # Get specific job status
                status = self.batch_processor.get_job_status(job_id)
                if not status:
                    return self._error_response(request_id, f"Job {job_id} not found", "JOB_NOT_FOUND")

                return {"type": MessageTypes.BATCH_STATUS, "job": status}

            else:
                # Get all active jobs
                jobs = self.batch_processor.list_active_jobs()
                return {"type": MessageTypes.BATCH_STATUS, "jobs": jobs, "count": len(jobs)}

        except Exception as e:
            logger.error(f"Get batch status error: {e}")
            return self._error_response(request_id, str(e), "STATUS_ERROR")

    def _handle_cancel_batch(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle cancel batch request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            job_id = message.get("job_id")
            if not job_id:
                return self._error_response(request_id, "Missing job_id", "MISSING_PARAMETER")

            success = self.batch_processor.cancel_job(job_id)
            if not success:
                return self._error_response(request_id, f"Could not cancel job {job_id}", "CANCEL_FAILED")

            return {"type": MessageTypes.BATCH_CANCELLED, "job_id": job_id, "cancelled": True}

        except Exception as e:
            logger.error(f"Cancel batch error: {e}")
            return self._error_response(request_id, str(e), "CANCEL_ERROR")

    def _handle_get_statistics(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle get statistics request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            stats = self.proposal_repo.get_statistics()

            return {"type": MessageTypes.STATISTICS, "statistics": stats}

        except Exception as e:
            logger.error(f"Get statistics error: {e}")
            return self._error_response(request_id, str(e), "STATISTICS_ERROR")

    def _handle_cleanup_old_proposals(self, message: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle cleanup old proposals request.

        Args:
            message: Request message
            request_id: Request identifier

        Returns:
            Response message
        """
        try:
            days = message.get("days", 30)

            cleaned_count = self.proposal_repo.cleanup_old_proposals(days)

            return {"type": MessageTypes.CLEANUP_COMPLETED, "cleaned_count": cleaned_count, "days": days}

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return self._error_response(request_id, str(e), "CLEANUP_ERROR")

    def _error_response(self, request_id: str, message: str, error_code: str) -> Dict[str, Any]:
        """Create error response message.

        Args:
            request_id: Request identifier
            message: Error message
            error_code: Error code

        Returns:
            Error response dictionary
        """
        return {
            "type": MessageTypes.ERROR,
            "request_id": request_id,
            "error": {"code": error_code, "message": message},
            "timestamp": datetime.utcnow().isoformat(),
        }
