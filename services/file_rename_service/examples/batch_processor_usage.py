"""Usage examples for the batch processor.

This module demonstrates how to use the BatchProcessor for different scenarios.
"""

import asyncio

from app.proposal.batch_processor import BatchProcessor
from app.proposal.models import NamingTemplate, RenameProposal


class ExampleProposalGenerator:
    """Example proposal generator for demonstration."""

    async def generate_proposal(self, filename: str, templates: list[NamingTemplate] | None = None) -> RenameProposal:
        """Generate a mock proposal for demonstration."""
        # Simulate some processing time
        await asyncio.sleep(0.01)

        return RenameProposal(
            original_filename=filename,
            proposed_filename=f"renamed_{filename}",
            confidence_score=0.85,
            explanation=f"Example proposal for {filename}",
            patterns_used=["example_pattern"],
            alternatives=[f"alt1_{filename}", f"alt2_{filename}"],
        )


async def basic_batch_processing():
    """Example of basic batch processing."""
    print("=== Basic Batch Processing ===")

    # Create batch processor with custom settings
    generator = ExampleProposalGenerator()
    processor = BatchProcessor(
        proposal_generator=generator,
        max_concurrent_tasks=20,  # Process up to 20 files concurrently
        chunk_size=10,  # Process in chunks of 10 files
    )

    # List of files to process
    filenames = [
        "Concert_2023_Recording.mp3",
        "Live_Show_Part1.flac",
        "acoustic_session.wav",
        "studio_demo_track.mp3",
        "rehearsal_notes.wav",
    ]

    print(f"Processing {len(filenames)} files...")

    # Process batch
    result = await processor.process_batch(filenames)

    # Display results
    print(f"Results: {result.successful_count}/{result.total_files} successful")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Success rate: {result.success_rate:.1f}%")

    for proposal in result.successful_proposals:
        print(f"  {proposal.original_filename} → {proposal.proposed_filename}")

    print()


async def batch_processing_with_templates():
    """Example of batch processing with custom templates."""
    print("=== Batch Processing with Custom Templates ===")

    generator = ExampleProposalGenerator()
    processor = BatchProcessor(proposal_generator=generator)

    # Define custom naming templates
    templates = [
        NamingTemplate(
            id="concert_template",
            name="Concert Recording Template",
            pattern="{artist} - {date} - {venue} - {quality}",
            user_id="user123",
        ),
        NamingTemplate(
            id="studio_template",
            name="Studio Recording Template",
            pattern="{artist} - {album} - {track} ({year})",
            user_id="user123",
        ),
    ]

    filenames = [
        "phish_2023-07-15_msg_sbd.mp3",
        "grateful_dead_1977_barton_hall.flac",
        "radiohead_ok_computer_paranoid_android.mp3",
    ]

    print(f"Processing {len(filenames)} files with custom templates...")

    result = await processor.process_batch(filenames, templates=templates)

    print(f"Results: {result.successful_count}/{result.total_files} successful")
    for proposal in result.successful_proposals:
        print(f"  {proposal.original_filename}")
        print(f"    → {proposal.proposed_filename}")
        print(f"    Patterns used: {', '.join(proposal.patterns_used)}")

    print()


async def batch_processing_with_progress():
    """Example of batch processing with progress tracking."""
    print("=== Batch Processing with Progress Tracking ===")

    generator = ExampleProposalGenerator()
    processor = BatchProcessor(proposal_generator=generator)

    # Generate a larger list of files for progress demonstration
    filenames = [f"file_{i:03d}.mp3" for i in range(50)]

    print(f"Processing {len(filenames)} files with progress tracking...")

    # Progress callback function
    def progress_callback(current: int, total: int, filename: str):
        percentage = (current / total) * 100
        if current % 10 == 0 or current == total:  # Update every 10 files or at completion
            print(f"  Progress: {current}/{total} ({percentage:.1f}%) - Last: {filename}")

    result = await processor.process_batch(filenames, progress_callback=progress_callback)

    print(f"✅ Completed: {result.successful_count}/{result.total_files} files in {result.processing_time:.2f}s")
    print(f"   Average time per file: {result.processing_time / result.total_files:.3f}s")
    print()


async def error_handling_example():
    """Example of handling partial failures in batch processing."""
    print("=== Error Handling Example ===")

    class FailingGenerator:
        """Generator that fails for certain files to demonstrate error handling."""

        async def generate_proposal(self, filename: str, **kwargs) -> RenameProposal:
            # Simulate failures for files containing "corrupt"
            if "corrupt" in filename.lower():
                raise ValueError(f"Simulated corruption in {filename}")

            await asyncio.sleep(0.005)
            return RenameProposal(
                original_filename=filename,
                proposed_filename=f"clean_{filename}",
                confidence_score=0.9,
                explanation=f"Successfully processed {filename}",
            )

    failing_generator = FailingGenerator()
    processor = BatchProcessor(proposal_generator=failing_generator)

    # Mix of good and bad files
    filenames = [
        "good_file_1.mp3",
        "corrupt_recording.mp3",  # This will fail
        "good_file_2.mp3",
        "corrupt_data.flac",  # This will fail
        "good_file_3.wav",
    ]

    print(f"Processing {len(filenames)} files (some will intentionally fail)...")

    result = await processor.process_batch(filenames)

    print("Results:")
    print(f"  Successful: {len(result.successful_proposals)}")
    print(f"  Failed: {len(result.failed_files)}")
    print(f"  Success rate: {result.success_rate:.1f}%")

    if result.failed_files:
        print(f"  Failed files: {result.failed_files}")

        # Handle partial failure
        analysis = processor.handle_partial_failure(result.failed_files, result.successful_proposals)

        print("  Recovery suggestions:")
        for suggestion in analysis["recovery_suggestions"]:
            print(f"    - {suggestion}")

    print()


async def performance_estimation_example():
    """Example of performance estimation before processing."""
    print("=== Performance Estimation Example ===")

    generator = ExampleProposalGenerator()
    processor = BatchProcessor(proposal_generator=generator)

    # Large batch for estimation
    file_count = 500

    print(f"Estimating processing time for {file_count} files...")

    # Use sample files for estimation
    sample_files = [f"sample_{i}.mp3" for i in range(5)]

    estimates = await processor.estimate_batch_time(file_count=file_count, sample_files=sample_files)

    print("Estimates:")
    print(f"  Total estimated time: {estimates['estimated_time']:.2f}s")
    print(f"  Time per file: {estimates['per_file_time']:.3f}s")
    print(f"  Concurrency factor: {estimates['concurrency_factor']:.2f}")
    print(f"  Sample size: {estimates['sample_size']} files")

    # Check if it meets performance requirements
    meets_100_file_requirement = estimates["estimated_time"] <= 5.0
    print(f"  Meets <5s requirement for 100+ files: {'✅' if meets_100_file_requirement else '❌'}")

    print()


async def main():
    """Run all usage examples."""
    print("Batch Processor Usage Examples")
    print("=" * 40)
    print()

    await basic_batch_processing()
    await batch_processing_with_templates()
    await batch_processing_with_progress()
    await error_handling_example()
    await performance_estimation_example()

    print("=" * 40)
    print("✅ All examples completed successfully!")
    print()
    print("Key features demonstrated:")
    print("- Parallel processing with configurable concurrency")
    print("- Custom template support")
    print("- Real-time progress tracking")
    print("- Graceful error handling and recovery")
    print("- Performance estimation")
    print("- Sub-5-second processing for 100+ files")


if __name__ == "__main__":
    asyncio.run(main())
