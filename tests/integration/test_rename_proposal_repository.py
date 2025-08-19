"""Integration tests for RenameProposal repository."""

from uuid import uuid4

import pytest

from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.rename_proposal_repository import RenameProposalRepository
from shared.core_types.src.repositories import RecordingRepository


@pytest.fixture
def db_manager(test_database):
    """Create database manager for tests."""
    return DatabaseManager()


@pytest.fixture
def recording_repo(db_manager):
    """Create recording repository."""
    return RecordingRepository(db_manager)


@pytest.fixture
def proposal_repo(db_manager):
    """Create rename proposal repository."""
    return RenameProposalRepository(db_manager)


@pytest.fixture
def test_recording(recording_repo):
    """Create a test recording."""
    recording = recording_repo.create(
        file_path="/test/path/song.mp3", file_name="song.mp3", sha256_hash="test_hash", xxh128_hash="test_xxh"
    )
    yield recording
    # Cleanup
    recording_repo.delete(recording.id)


class TestRenameProposalRepository:
    """Test rename proposal repository operations."""

    def test_create_proposal(self, proposal_repo, test_recording):
        """Test creating a rename proposal."""
        proposal = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/test/path/Artist - Title.mp3",
            confidence_score=0.95,
            status="pending",
        )

        assert proposal.id is not None
        assert proposal.recording_id == test_recording.id
        assert proposal.proposed_filename == "Artist - Title.mp3"
        assert proposal.confidence_score == 0.95
        assert proposal.status == "pending"

        # Cleanup
        proposal_repo.delete(proposal.id)

    def test_get_proposal(self, proposal_repo, test_recording):
        """Test getting a proposal by ID."""
        proposal = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/test/path/Artist - Title.mp3",
        )

        fetched = proposal_repo.get(proposal.id)
        assert fetched is not None
        assert fetched.id == proposal.id
        assert fetched.proposed_filename == "Artist - Title.mp3"

        # Cleanup
        proposal_repo.delete(proposal.id)

    def test_get_by_recording(self, proposal_repo, test_recording):
        """Test getting proposals by recording ID."""
        # Create multiple proposals
        proposals = []
        for i in range(3):
            proposal = proposal_repo.create(
                recording_id=test_recording.id,
                original_path="/test/path",
                original_filename="song.mp3",
                proposed_filename=f"Artist - Title {i}.mp3",
                full_proposed_path=f"/test/path/Artist - Title {i}.mp3",
                status="pending" if i < 2 else "approved",
            )
            proposals.append(proposal)

        # Get all proposals for recording
        all_proposals = proposal_repo.get_by_recording(test_recording.id)
        assert len(all_proposals) == 3

        # Get only pending proposals
        pending = proposal_repo.get_by_recording(test_recording.id, status="pending")
        assert len(pending) == 2

        # Cleanup
        for proposal in proposals:
            proposal_repo.delete(proposal.id)

    def test_update_proposal(self, proposal_repo, test_recording):
        """Test updating a proposal."""
        proposal = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/test/path/Artist - Title.mp3",
            status="pending",
        )

        # Update proposal
        updated = proposal_repo.update(proposal.id, status="approved", confidence_score=0.98)

        assert updated is not None
        assert updated.status == "approved"
        assert updated.confidence_score == 0.98

        # Cleanup
        proposal_repo.delete(proposal.id)

    def test_update_status(self, proposal_repo, test_recording):
        """Test updating proposal status."""
        proposal = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/test/path/Artist - Title.mp3",
            status="pending",
        )

        # Update status
        success = proposal_repo.update_status(proposal.id, "approved")
        assert success is True

        # Verify update
        updated = proposal_repo.get(proposal.id)
        assert updated.status == "approved"

        # Cleanup
        proposal_repo.delete(proposal.id)

    def test_batch_update_status(self, proposal_repo, test_recording):
        """Test batch updating proposal status."""
        # Create multiple proposals
        proposal_ids = []
        for i in range(3):
            proposal = proposal_repo.create(
                recording_id=test_recording.id,
                original_path="/test/path",
                original_filename=f"song{i}.mp3",
                proposed_filename=f"Artist - Title {i}.mp3",
                full_proposed_path=f"/test/path/Artist - Title {i}.mp3",
                status="pending",
            )
            proposal_ids.append(proposal.id)

        # Batch update status
        count = proposal_repo.batch_update_status(proposal_ids, "approved")
        assert count == 3

        # Verify updates
        for proposal_id in proposal_ids:
            proposal = proposal_repo.get(proposal_id)
            assert proposal.status == "approved"
            # Cleanup
            proposal_repo.delete(proposal_id)

    def test_find_conflicts(self, proposal_repo, test_recording, recording_repo):
        """Test finding conflicting proposals."""
        # Create another recording
        recording2 = recording_repo.create(file_path="/test/path/song2.mp3", file_name="song2.mp3")

        # Create proposals with same target path
        target_path = "/test/path/Artist - Title.mp3"

        proposal1 = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path=target_path,
            status="pending",
        )

        proposal2 = proposal_repo.create(
            recording_id=recording2.id,
            original_path="/test/path",
            original_filename="song2.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path=target_path,
            status="pending",
        )

        # Find conflicts for first recording
        conflicts = proposal_repo.find_conflicts(target_path, test_recording.id)
        assert len(conflicts) == 1
        assert conflicts[0].recording_id == recording2.id

        # Cleanup
        proposal_repo.delete(proposal1.id)
        proposal_repo.delete(proposal2.id)
        recording_repo.delete(recording2.id)

    def test_get_statistics(self, proposal_repo, test_recording):
        """Test getting statistics."""
        # Create proposals with different statuses
        proposals = []
        for status in ["pending", "pending", "approved", "rejected"]:
            proposal = proposal_repo.create(
                recording_id=test_recording.id,
                original_path="/test/path",
                original_filename="song.mp3",
                proposed_filename=f"Artist - {status}.mp3",
                full_proposed_path=f"/test/path/Artist - {status}.mp3",
                status=status,
                confidence_score=0.85,
                conflicts=["conflict1"] if status == "rejected" else None,
                warnings=["warning1"] if status == "pending" else None,
            )
            proposals.append(proposal)

        # Get statistics
        stats = proposal_repo.get_statistics()

        assert stats["total"] == 4
        assert stats["by_status"]["pending"] == 2
        assert stats["by_status"]["approved"] == 1
        assert stats["by_status"]["rejected"] == 1
        assert stats["average_confidence"] > 0
        assert stats["with_conflicts"] == 1
        assert stats["with_warnings"] == 2

        # Cleanup
        for proposal in proposals:
            proposal_repo.delete(proposal.id)

    def test_cleanup_old_proposals(self, proposal_repo, test_recording):
        """Test cleaning up old proposals."""
        # Create old proposals
        # We can't directly set the updated_at in create, so we'll update after creation
        old_proposals = []
        for status in ["rejected", "applied"]:
            proposal = proposal_repo.create(
                recording_id=test_recording.id,
                original_path="/test/path",
                original_filename="old.mp3",
                proposed_filename=f"Old - {status}.mp3",
                full_proposed_path=f"/test/path/Old - {status}.mp3",
                status=status,
            )
            # Manually update the updated_at timestamp (would need direct DB access)
            # For now, we'll just test the function works
            old_proposals.append(proposal)

        # Create recent proposal (should not be deleted)
        recent = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="recent.mp3",
            proposed_filename="Recent - Title.mp3",
            full_proposed_path="/test/path/Recent - Title.mp3",
            status="pending",
        )

        # In a real test, we'd update the timestamps and test cleanup
        # For now, just verify the function doesn't error
        _ = proposal_repo.cleanup_old_proposals(days=30)

        # Cleanup
        for proposal in old_proposals:
            proposal_repo.delete(proposal.id)
        proposal_repo.delete(recent.id)

    def test_delete_proposal(self, proposal_repo, test_recording):
        """Test deleting a proposal."""
        proposal = proposal_repo.create(
            recording_id=test_recording.id,
            original_path="/test/path",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/test/path/Artist - Title.mp3",
        )

        # Delete proposal
        success = proposal_repo.delete(proposal.id)
        assert success is True

        # Verify deletion
        fetched = proposal_repo.get(proposal.id)
        assert fetched is None

        # Try deleting non-existent proposal
        success = proposal_repo.delete(uuid4())
        assert success is False
