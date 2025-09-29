"""
Tests for CueFileRepository.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from services.tracklist_service.src.models.cue_file import CueFileDB
from services.tracklist_service.src.repository.cue_file_repository import CueFileRepository
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def mock_session():
    """Create mock async database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def repository(mock_session):
    """Create CueFileRepository with mock session."""
    return CueFileRepository(mock_session)


@pytest.fixture
def sample_cue_file():
    """Create sample CUE file for testing."""
    return CueFileDB(
        id=uuid4(),
        tracklist_id=uuid4(),
        format="standard",
        version=1,
        file_path="/data/cue_files/2024/01/test.cue",
        checksum="abc123def456",
        file_size=2048,
        generation_time_ms=150.5,
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class TestCueFileRepository:
    """Test CueFileRepository operations."""

    @pytest.mark.asyncio
    async def test_create_cue_file_new(self, repository, mock_session, sample_cue_file):
        """Test creating a new CUE file."""
        # Setup: No existing files
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.create_cue_file(sample_cue_file)

        # Verify
        assert result == sample_cue_file
        assert sample_cue_file.version == 1
        assert sample_cue_file.is_active is True
        mock_session.add.assert_called_once_with(sample_cue_file)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_cue_file)

    @pytest.mark.asyncio
    async def test_create_cue_file_with_existing_versions(self, repository, mock_session, sample_cue_file):
        """Test creating a new version of existing CUE file."""
        # Setup: Existing files
        existing_file = CueFileDB(
            id=uuid4(),
            tracklist_id=sample_cue_file.tracklist_id,
            format=sample_cue_file.format,
            version=2,
            file_path="/data/cue_files/2024/01/test.v2.cue",
            checksum="old123",
            file_size=1024,
            is_active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Mock the get_cue_files_by_tracklist_and_format call
        repository.get_cue_files_by_tracklist_and_format = AsyncMock(return_value=[existing_file])

        # Execute
        result = await repository.create_cue_file(sample_cue_file)

        # Verify
        assert result == sample_cue_file
        assert sample_cue_file.version == 3  # Next version after 2
        assert sample_cue_file.is_active is True
        assert existing_file.is_active is False  # Previous version deactivated
        mock_session.add.assert_called_once_with(sample_cue_file)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_cue_file_database_error(self, repository, mock_session, sample_cue_file):
        """Test create with database error."""
        # Setup: Database error
        mock_session.commit.side_effect = SQLAlchemyError("Database error")

        # Mock the get_cue_files_by_tracklist_and_format call
        repository.get_cue_files_by_tracklist_and_format = AsyncMock(return_value=[])

        # Execute and verify
        with pytest.raises(SQLAlchemyError):
            await repository.create_cue_file(sample_cue_file)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cue_file_by_id_found(self, repository, mock_session, sample_cue_file):
        """Test getting CUE file by ID when it exists."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_cue_file
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_cue_file_by_id(sample_cue_file.id)

        # Verify
        assert result == sample_cue_file
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cue_file_by_id_not_found(self, repository, mock_session):
        """Test getting CUE file by ID when it doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_cue_file_by_id(uuid4())

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cue_files_by_tracklist(self, repository, mock_session):
        """Test getting all CUE files for a tracklist."""
        # Setup
        tracklist_id = uuid4()
        cue_files = [
            CueFileDB(
                id=uuid4(),
                tracklist_id=tracklist_id,
                format=f"format_{i}",
                version=1,
                file_path=f"/data/cue_files/test_{i}.cue",
                checksum=f"checksum_{i}",
                file_size=1024 * i,
                is_active=True,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            for i in range(3)
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = cue_files
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_cue_files_by_tracklist(tracklist_id)

        # Verify
        assert len(result) == 3
        assert result == cue_files

    @pytest.mark.asyncio
    async def test_get_active_cue_file(self, repository, mock_session, sample_cue_file):
        """Test getting active CUE file for tracklist and format."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_cue_file
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_active_cue_file(sample_cue_file.tracklist_id, sample_cue_file.format)

        # Verify
        assert result == sample_cue_file

    @pytest.mark.asyncio
    async def test_update_cue_file(self, repository, mock_session, sample_cue_file):
        """Test updating CUE file."""
        # Setup
        sample_cue_file.file_size = 4096
        sample_cue_file.checksum = "updated123"

        # Execute
        result = await repository.update_cue_file(sample_cue_file)

        # Verify
        assert result == sample_cue_file
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_cue_file)

    @pytest.mark.asyncio
    async def test_soft_delete_cue_file(self, repository, mock_session, sample_cue_file):
        """Test soft deleting CUE file."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_cue_file
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.soft_delete_cue_file(sample_cue_file.id)

        # Verify
        assert result is True
        assert sample_cue_file.is_active is False
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_soft_delete_cue_file_not_found(self, repository, mock_session):
        """Test soft deleting non-existent CUE file."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.soft_delete_cue_file(uuid4())

        # Verify
        assert result is False
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_hard_delete_cue_file(self, repository, mock_session, sample_cue_file):
        """Test hard deleting CUE file."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_cue_file
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.hard_delete_cue_file(sample_cue_file.id)

        # Verify
        assert result is True
        mock_session.delete.assert_called_once_with(sample_cue_file)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_cue_files_with_filters(self, repository, mock_session):
        """Test listing CUE files with filters and pagination."""
        # Setup
        cue_files = [
            CueFileDB(
                id=uuid4(),
                tracklist_id=uuid4(),
                format="standard",
                version=1,
                file_path=f"/data/cue_files/test_{i}.cue",
                checksum=f"checksum_{i}",
                file_size=1024 * i,
                is_active=True,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = cue_files[:2]  # Pagination limit
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.list_cue_files(tracklist_id=uuid4(), cue_format="standard", limit=2, offset=0)

        # Verify
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_count_cue_files(self, repository, mock_session):
        """Test counting CUE files with filters."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.count_cue_files(tracklist_id=uuid4())

        # Verify
        assert result == 42

    @pytest.mark.asyncio
    async def test_count_cue_files_none_result(self, repository, mock_session):
        """Test counting CUE files when result is None."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.count_cue_files()

        # Verify
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_file_versions(self, repository, mock_session):
        """Test getting all versions of a CUE file."""
        # Setup
        base_file = CueFileDB(
            id=uuid4(),
            tracklist_id=uuid4(),
            format="standard",
            version=3,
            file_path="/data/cue_files/test.cue",
            checksum="latest",
            file_size=2048,
            is_active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        versions = [
            CueFileDB(
                id=uuid4(),
                tracklist_id=base_file.tracklist_id,
                format=base_file.format,
                version=i,
                file_path=f"/data/cue_files/test.v{i}.cue",
                checksum=f"checksum_{i}",
                file_size=1024 * i,
                is_active=(i == 3),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            for i in range(1, 4)
        ]

        # Mock get_cue_file_by_id
        repository.get_cue_file_by_id = AsyncMock(return_value=base_file)

        # Mock execute for versions query
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = versions
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_file_versions(base_file.id)

        # Verify
        assert len(result) == 3
        assert all(v.tracklist_id == base_file.tracklist_id for v in result)
        assert all(v.format == base_file.format for v in result)

    @pytest.mark.asyncio
    async def test_get_file_versions_not_found(self, repository, mock_session):
        """Test getting versions when base file not found."""
        # Setup
        repository.get_cue_file_by_id = AsyncMock(return_value=None)

        # Execute
        result = await repository.get_file_versions(uuid4())

        # Verify
        assert result == []
        mock_session.execute.assert_not_called()
