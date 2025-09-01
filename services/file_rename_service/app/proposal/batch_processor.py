"""Batch processor for rename proposals with parallel processing and progress tracking."""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from .generator import ProposalGenerator
from .models import NamingTemplate, RenameProposal

logger = logging.getLogger(__name__)


class BatchProcessingResult:
    """Result container for batch processing operations."""

    def __init__(
        self,
        successful_proposals: list[RenameProposal],
        failed_files: list[str],
        total_files: int,
        processing_time: float,
        errors: dict[str, str] | None = None,
    ) -> None:
        """Initialize batch processing result.

        Args:
            successful_proposals: List of successfully generated proposals
            failed_files: List of filenames that failed processing
            total_files: Total number of files processed
            processing_time: Total time taken for processing in seconds
            errors: Optional mapping of failed filenames to error messages
        """
        self.successful_proposals = successful_proposals
        self.failed_files = failed_files
        self.total_files = total_files
        self.processing_time = processing_time
        self.errors = errors or {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (len(self.successful_proposals) / self.total_files) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        return 100.0 - self.success_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "successful_count": len(self.successful_proposals),
            "failed_count": len(self.failed_files),
            "total_files": self.total_files,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "processing_time": self.processing_time,
            "successful_proposals": [proposal.model_dump() for proposal in self.successful_proposals],
            "failed_files": self.failed_files,
            "errors": self.errors,
        }


class BatchProcessor:
    """Batch processor for generating rename proposals with parallel processing."""

    def __init__(
        self,
        proposal_generator: ProposalGenerator | None = None,
        max_concurrent_tasks: int = 50,
        chunk_size: int = 10,
    ) -> None:
        """Initialize the batch processor.

        Args:
            proposal_generator: ProposalGenerator instance (will create if None)
            max_concurrent_tasks: Maximum number of concurrent processing tasks
            chunk_size: Number of files to process in each chunk
        """
        self.generator = proposal_generator or ProposalGenerator()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.chunk_size = chunk_size
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def process_batch(
        self,
        filenames: list[str],
        templates: list[NamingTemplate] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchProcessingResult:
        """Process a batch of filenames to generate rename proposals.

        Args:
            filenames: List of filenames to process
            templates: Optional custom naming templates to apply
            progress_callback: Optional callback function called with (current, total, filename)

        Returns:
            BatchProcessingResult containing successful proposals and failed files
        """
        if not filenames:
            return BatchProcessingResult([], [], 0, 0.0)

        start_time = time.time()
        total_files = len(filenames)

        logger.info(f"Starting batch processing of {total_files} files")

        # Track progress and results
        successful_proposals: list[RenameProposal] = []
        failed_files: list[str] = []
        errors: dict[str, str] = {}
        processed_count = 0

        # Process files in chunks to manage memory and provide progress updates
        for chunk_start in range(0, total_files, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_files)
            chunk_filenames = filenames[chunk_start:chunk_end]

            # Process chunk with semaphore-controlled concurrency
            chunk_results = await asyncio.gather(
                *[self._process_file_with_semaphore(filename, templates) for filename in chunk_filenames],
                return_exceptions=True,
            )

            # Process chunk results
            for filename, result in zip(chunk_filenames, chunk_results, strict=False):
                processed_count += 1

                if isinstance(result, Exception):
                    error_msg = str(result)
                    logger.error(f"Error processing '{filename}': {error_msg}")
                    failed_files.append(filename)
                    errors[filename] = error_msg
                elif result is None:
                    error_msg = "Processing returned None"
                    logger.error(f"No result for '{filename}': {error_msg}")
                    failed_files.append(filename)
                    errors[filename] = error_msg
                elif isinstance(result, RenameProposal):
                    successful_proposals.append(result)
                else:
                    error_msg = f"Unexpected result type: {type(result)}"
                    logger.error(f"Error processing '{filename}': {error_msg}")
                    failed_files.append(filename)
                    errors[filename] = error_msg

                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(processed_count, total_files, filename)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

        processing_time = time.time() - start_time

        logger.info(
            f"Batch processing completed: {len(successful_proposals)}/{total_files} successful "
            f"in {processing_time:.2f}s"
        )

        return BatchProcessingResult(
            successful_proposals=successful_proposals,
            failed_files=failed_files,
            total_files=total_files,
            processing_time=processing_time,
            errors=errors,
        )

    async def _process_file_with_semaphore(
        self,
        filename: str,
        templates: list[NamingTemplate] | None = None,
    ) -> RenameProposal | None:
        """Process a single file with semaphore control for concurrency.

        Args:
            filename: Filename to process
            templates: Optional naming templates

        Returns:
            RenameProposal or None if processing failed
        """
        async with self._semaphore:
            return await self.process_file(filename, templates)

    async def process_file(
        self,
        filename: str,
        templates: list[NamingTemplate] | None = None,
    ) -> RenameProposal | None:
        """Process a single filename to generate a rename proposal.

        Args:
            filename: Filename to process
            templates: Optional custom naming templates to apply

        Returns:
            RenameProposal or None if processing failed
        """
        try:
            return await self.generator.generate_proposal(filename, templates=templates)
        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}")
            return None

    def handle_partial_failure(
        self,
        failed_files: list[str],
        successful_proposals: list[RenameProposal],
    ) -> dict[str, Any]:
        """Handle partial batch failure by providing recovery information.

        Args:
            failed_files: List of files that failed processing
            successful_proposals: List of successful proposals

        Returns:
            Dictionary with failure analysis and recovery suggestions
        """
        total_files = len(failed_files) + len(successful_proposals)
        failure_rate = len(failed_files) / total_files * 100 if total_files > 0 else 0

        analysis: dict[str, Any] = {
            "total_files": total_files,
            "successful_count": len(successful_proposals),
            "failed_count": len(failed_files),
            "failure_rate_percent": failure_rate,
            "failed_files": failed_files,
            "recovery_suggestions": [],
        }

        # Add recovery suggestions based on failure patterns
        if failure_rate > 50:
            analysis["recovery_suggestions"].append("High failure rate - consider checking file formats")
        elif failure_rate > 20:
            analysis["recovery_suggestions"].append("Moderate failure rate - review failed files individually")

        if len(failed_files) > 0:
            analysis["recovery_suggestions"].extend(
                [
                    "Retry failed files individually for detailed error analysis",
                    "Check if failed files have unusual formats or characters",
                    "Consider preprocessing files to standardize formats",
                ]
            )

        if len(successful_proposals) > 0:
            analysis["recovery_suggestions"].append(
                f"Process successful proposals ({len(successful_proposals)} files) while investigating failures"
            )

        return analysis

    def track_progress(
        self,
        current: int,
        total: int,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> None:
        """Track and report processing progress.

        Args:
            current: Current number of processed files
            total: Total number of files to process
            callback: Optional callback function to call with progress info
        """
        percentage = 0.0 if total == 0 else (current / total) * 100

        logger.info(f"Progress: {current}/{total} files ({percentage:.1f}%)")

        if callback:
            try:
                callback(current, total, percentage)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    async def estimate_batch_time(
        self,
        file_count: int,
        sample_size: int = 5,
        sample_files: list[str] | None = None,
    ) -> dict[str, float]:
        """Estimate processing time for a batch of files.

        Args:
            file_count: Total number of files to process
            sample_size: Number of files to use for timing estimation
            sample_files: Optional specific files to use as samples

        Returns:
            Dictionary with time estimates in seconds
        """
        if file_count == 0:
            return {"estimated_time": 0.0, "per_file_time": 0.0, "sample_size": 0}

        # Use provided samples or create dummy samples for estimation
        if sample_files:
            samples = sample_files[:sample_size]
        else:
            samples = [f"sample_file_{i}.mp3" for i in range(min(sample_size, file_count))]

        # Time sample processing
        start_time = time.time()
        sample_results = await asyncio.gather(
            *[self.process_file(filename) for filename in samples],
            return_exceptions=True,
        )
        sample_time = time.time() - start_time

        # Calculate estimates
        successful_samples = sum(1 for result in sample_results if not isinstance(result, Exception))
        # Use fallback time if no successful samples
        per_file_time = 1.0 if successful_samples == 0 else sample_time / successful_samples

        # Account for parallel processing efficiency
        concurrency_factor = min(self.max_concurrent_tasks, file_count) / file_count
        estimated_time = per_file_time * file_count * concurrency_factor

        return {
            "estimated_time": estimated_time,
            "per_file_time": per_file_time,
            "sample_size": len(samples),
            "concurrency_factor": concurrency_factor,
        }


class ProgressTracker:
    """Helper class for tracking and reporting batch processing progress."""

    def __init__(
        self,
        total_files: int,
        update_interval: float = 1.0,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize progress tracker.

        Args:
            total_files: Total number of files to process
            update_interval: Minimum time between progress updates in seconds
            callback: Optional callback to receive progress updates
        """
        self.total_files = total_files
        self.update_interval = update_interval
        self.callback = callback
        self.processed_files = 0
        self.start_time = time.time()
        self.last_update_time = 0.0

    def update(self, filename: str) -> None:
        """Update progress with a newly processed file.

        Args:
            filename: Name of the file that was just processed
        """
        self.processed_files += 1
        current_time = time.time()

        # Only update if enough time has passed or we're done
        if current_time - self.last_update_time >= self.update_interval or self.processed_files == self.total_files:
            self._send_update(filename, current_time)
            self.last_update_time = current_time

    def _send_update(self, current_filename: str, current_time: float) -> None:
        """Send progress update to callback.

        Args:
            current_filename: Current file being processed
            current_time: Current timestamp
        """
        if not self.callback:
            return

        elapsed_time = current_time - self.start_time
        percentage = (self.processed_files / self.total_files) * 100 if self.total_files > 0 else 0

        # Estimate remaining time
        if self.processed_files > 0:
            rate = self.processed_files / elapsed_time
            remaining_files = self.total_files - self.processed_files
            estimated_remaining_time = remaining_files / rate if rate > 0 else 0
        else:
            estimated_remaining_time = 0

        progress_info = {
            "processed": self.processed_files,
            "total": self.total_files,
            "percentage": percentage,
            "current_file": current_filename,
            "elapsed_time": elapsed_time,
            "estimated_remaining_time": estimated_remaining_time,
        }

        try:
            self.callback(progress_info)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")
