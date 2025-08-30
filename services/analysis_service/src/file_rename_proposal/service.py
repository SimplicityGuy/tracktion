"""Service factory for file rename proposal components."""

import logging
from typing import Any, Dict, Optional

from .batch_processor import BatchProcessor
from .confidence_scorer import ConfidenceScorer
from .conflict_detector import ConflictDetector
from .config import FileRenameProposalConfig
from .integration import FileRenameProposalIntegration
from .message_interface import RenameProposalMessageInterface
from .pattern_manager import PatternManager
from .proposal_generator import ProposalGenerator
from .validator import FilesystemValidator
from shared.core_types.src.rename_proposal_repository import RenameProposalRepository
from shared.core_types.src.repositories import RecordingRepository
from shared.core_types.src.database import DatabaseManager
from ..file_rename_executor.executor import FileRenameExecutor

logger = logging.getLogger(__name__)


class FileRenameProposalServiceFactory:
    """Factory for creating file rename proposal service components."""

    def __init__(self, config: Optional[FileRenameProposalConfig] = None) -> None:
        """Initialize the service factory.

        Args:
            config: Optional configuration, will load from environment if not provided
        """
        self.config = config or FileRenameProposalConfig.from_env()
        self._db_manager: Optional[DatabaseManager] = None
        self._proposal_repo: Optional[RenameProposalRepository] = None
        self._recording_repo: Optional[RecordingRepository] = None
        self._rename_executor: Optional[FileRenameExecutor] = None

    @property
    def db_manager(self) -> DatabaseManager:
        """Get or create database manager instance."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager()
        return self._db_manager

    @property
    def proposal_repo(self) -> RenameProposalRepository:
        """Get or create proposal repository instance."""
        if self._proposal_repo is None:
            self._proposal_repo = RenameProposalRepository(self.db_manager)
        return self._proposal_repo

    @property
    def recording_repo(self) -> RecordingRepository:
        """Get or create recording repository instance."""
        if self._recording_repo is None:
            self._recording_repo = RecordingRepository(self.db_manager)
        return self._recording_repo

    @property
    def rename_executor(self) -> FileRenameExecutor:
        """Get or create file rename executor instance."""
        if self._rename_executor is None:
            self._rename_executor = FileRenameExecutor(self.db_manager)
        return self._rename_executor

    def create_pattern_manager(self) -> PatternManager:
        """Create a pattern manager instance."""
        return PatternManager(self.config)

    def create_validator(self) -> FilesystemValidator:
        """Create a filesystem validator instance."""
        return FilesystemValidator(self.config)

    def create_conflict_detector(self) -> ConflictDetector:
        """Create a conflict detector instance."""
        return ConflictDetector()

    def create_confidence_scorer(self) -> ConfidenceScorer:
        """Create a confidence scorer instance."""
        return ConfidenceScorer()

    def create_proposal_generator(self) -> ProposalGenerator:
        """Create a proposal generator instance."""
        pattern_manager = self.create_pattern_manager()
        validator = self.create_validator()
        return ProposalGenerator(self.config, pattern_manager, validator)

    def create_batch_processor(self) -> BatchProcessor:
        """Create a batch processor instance."""
        proposal_generator = self.create_proposal_generator()
        conflict_detector = self.create_conflict_detector()
        confidence_scorer = self.create_confidence_scorer()

        return BatchProcessor(
            proposal_generator=proposal_generator,
            conflict_detector=conflict_detector,
            confidence_scorer=confidence_scorer,
            proposal_repo=self.proposal_repo,
            recording_repo=self.recording_repo,
        )

    def create_message_interface(self) -> RenameProposalMessageInterface:
        """Create a message interface instance."""
        proposal_generator = self.create_proposal_generator()
        conflict_detector = self.create_conflict_detector()
        confidence_scorer = self.create_confidence_scorer()
        batch_processor = self.create_batch_processor()

        return RenameProposalMessageInterface(
            proposal_generator=proposal_generator,
            conflict_detector=conflict_detector,
            confidence_scorer=confidence_scorer,
            proposal_repo=self.proposal_repo,
            recording_repo=self.recording_repo,
            batch_processor=batch_processor,
            rename_executor=self.rename_executor,
        )

    def create_integration(self) -> FileRenameProposalIntegration:
        """Create an integration instance for use with analysis pipeline."""
        return FileRenameProposalIntegration(
            proposal_repo=self.proposal_repo,
            recording_repo=self.recording_repo,
            config=self.config,
        )

    def create_all_components(self) -> dict:
        """Create all components and return as a dictionary.

        Returns:
            Dictionary containing all service components
        """
        return {
            "config": self.config,
            "db_manager": self.db_manager,
            "proposal_repo": self.proposal_repo,
            "recording_repo": self.recording_repo,
            "pattern_manager": self.create_pattern_manager(),
            "validator": self.create_validator(),
            "conflict_detector": self.create_conflict_detector(),
            "confidence_scorer": self.create_confidence_scorer(),
            "proposal_generator": self.create_proposal_generator(),
            "batch_processor": self.create_batch_processor(),
            "message_interface": self.create_message_interface(),
            "integration": self.create_integration(),
            "rename_executor": self.rename_executor,
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on service components.

        Returns:
            Health status dictionary
        """
        health: Dict[str, Any] = {
            "service": "file_rename_proposal",
            "status": "healthy",
            "components": {},
        }

        try:
            # Test database connection
            with self.db_manager.get_db_session() as session:
                session.execute("SELECT 1")  # type: ignore[call-overload]
            health["components"]["database"] = {"status": "connected"}
        except Exception as e:
            health["components"]["database"] = {"status": "error", "message": str(e)}
            health["status"] = "unhealthy"

        try:
            # Test configuration
            _ = self.config.default_patterns
            health["components"]["config"] = {"status": "loaded"}
        except Exception as e:
            health["components"]["config"] = {"status": "error", "message": str(e)}
            health["status"] = "unhealthy"

        try:
            # Test repositories
            _ = self.proposal_repo
            _ = self.recording_repo
            health["components"]["repositories"] = {"status": "initialized"}
        except Exception as e:
            health["components"]["repositories"] = {"status": "error", "message": str(e)}
            health["status"] = "unhealthy"

        return health


# Convenience function for quick service creation
def create_file_rename_proposal_service(
    config: Optional[FileRenameProposalConfig] = None,
) -> FileRenameProposalServiceFactory:
    """Create a file rename proposal service factory.

    Args:
        config: Optional configuration, will load from environment if not provided

    Returns:
        Service factory instance
    """
    return FileRenameProposalServiceFactory(config)
