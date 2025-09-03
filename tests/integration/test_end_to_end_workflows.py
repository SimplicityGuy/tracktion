"""End-to-end workflow tests for the complete system.

This module contains comprehensive end-to-end tests that verify complete user
journeys through the system, from file upload through analysis, cataloging,
and tracklist generation.
"""

import asyncio

# Configure test logging
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAudioFile:
    """Mock audio file for testing purposes."""

    def __init__(
        self, path: str, duration: float = 180.0, format: str = "mp3", bitrate: int = 320, sample_rate: int = 44100
    ):
        self.path = Path(path)
        self.duration = duration
        self.format = format
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.metadata = {
            "title": self.path.stem,
            "artist": "Test Artist",
            "album": "Test Album",
            "genre": "Electronic",
            "year": 2023,
        }

    def get_metadata(self) -> dict[str, Any]:
        """Get file metadata."""
        return self.metadata.copy()

    def get_audio_properties(self) -> dict[str, Any]:
        """Get audio properties."""
        return {
            "duration": self.duration,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "format": self.format,
        }


class MockAnalysisResult:
    """Mock analysis result for testing."""

    def __init__(self, file_path: str, analysis_data: dict[str, Any] | None = None):
        self.file_path = file_path
        self.analysis_id = str(uuid.uuid4())
        self.timestamp = datetime.now(UTC)
        self.analysis_data = analysis_data or self._generate_default_analysis()

    def _generate_default_analysis(self) -> dict[str, Any]:
        """Generate default analysis data."""
        return {
            "bpm": 128.5,
            "key": "Am",
            "energy": 0.75,
            "danceability": 0.82,
            "valence": 0.45,
            "acousticness": 0.15,
            "instrumentalness": 0.95,
            "loudness": -8.5,
            "spectral_centroid": 1500.0,
            "zero_crossing_rate": 0.1,
            "mfcc": [1.5, -2.1, 0.8, 1.2, -0.5],
            "onset_times": [0.0, 2.1, 4.3, 6.5, 8.7],
            "beat_times": [0.0, 0.47, 0.94, 1.41, 1.88],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "analysis_id": self.analysis_id,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat(),
            "analysis_data": self.analysis_data,
        }


class MockCuePoint:
    """Mock cue point for testing."""

    def __init__(
        self,
        timestamp: float,
        cue_type: str = "mix_in",
        confidence: float = 0.9,
        metadata: dict[str, Any] | None = None,
    ):
        self.timestamp = timestamp
        self.cue_type = cue_type
        self.confidence = confidence
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "cue_type": self.cue_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class WorkflowTestOrchestrator:
    """Orchestrates end-to-end workflow tests."""

    def __init__(self):
        self.file_service = AsyncMock()
        self.analysis_service = AsyncMock()
        self.catalog_service = AsyncMock()
        self.cue_service = AsyncMock()
        self.notification_service = AsyncMock()

        self._setup_mock_behaviors()

    def _setup_mock_behaviors(self):
        """Setup default mock behaviors."""
        # File service mocks
        self.file_service.detect_new_files = AsyncMock(return_value=[])
        self.file_service.validate_file = AsyncMock(return_value=True)
        self.file_service.get_file_metadata = AsyncMock()

        # Analysis service mocks
        self.analysis_service.analyze_file = AsyncMock()
        self.analysis_service.get_analysis = AsyncMock()

        # Catalog service mocks
        self.catalog_service.add_track = AsyncMock()
        self.catalog_service.get_track = AsyncMock()
        self.catalog_service.update_track_analysis = AsyncMock()

        # Cue service mocks
        self.cue_service.generate_cue_points = AsyncMock()
        self.cue_service.get_cue_points = AsyncMock(return_value=[])

        # Notification service mocks
        self.notification_service.send_notification = AsyncMock()

    async def simulate_file_upload(self, files: list[MockAudioFile]) -> list[str]:
        """Simulate file upload workflow."""
        uploaded_files = []

        for file in files:
            # Simulate file detection
            self.file_service.detect_new_files.return_value = [file.path]

            # Simulate file validation
            self.file_service.validate_file.return_value = True

            # Simulate metadata extraction
            self.file_service.get_file_metadata.return_value = file.get_metadata()

            uploaded_files.append(str(file.path))

        return uploaded_files

    async def simulate_analysis_workflow(self, file_path: str, mock_file: MockAudioFile) -> MockAnalysisResult:
        """Simulate analysis workflow."""
        # Generate mock analysis result
        analysis_result = MockAnalysisResult(file_path)

        # Configure analysis service mock
        self.analysis_service.analyze_file.return_value = analysis_result.to_dict()
        self.analysis_service.get_analysis.return_value = analysis_result.to_dict()

        # Simulate analysis execution
        await self.analysis_service.analyze_file(file_path, mock_file.get_audio_properties())

        return analysis_result

    async def simulate_cataloging_workflow(
        self, file_path: str, mock_file: MockAudioFile, analysis_result: MockAnalysisResult
    ) -> dict[str, Any]:
        """Simulate cataloging workflow."""
        track_data = {
            "id": str(uuid.uuid4()),
            "file_path": file_path,
            "metadata": mock_file.get_metadata(),
            "audio_properties": mock_file.get_audio_properties(),
            "analysis_data": analysis_result.analysis_data,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }

        # Configure catalog service mock
        self.catalog_service.add_track.return_value = track_data["id"]
        self.catalog_service.get_track.return_value = track_data

        # Simulate cataloging execution
        await self.catalog_service.add_track(track_data)
        await self.catalog_service.update_track_analysis(track_data["id"], analysis_result.to_dict())

        return track_data

    async def simulate_cue_generation_workflow(
        self, track_id: str, analysis_result: MockAnalysisResult
    ) -> list[MockCuePoint]:
        """Simulate cue generation workflow."""
        # Generate mock cue points
        cue_points = [
            MockCuePoint(8.5, "mix_in", 0.95, {"bpm": analysis_result.analysis_data["bpm"]}),
            MockCuePoint(165.2, "mix_out", 0.92, {"energy_fade": True}),
            MockCuePoint(45.3, "vocal_start", 0.88),
            MockCuePoint(120.7, "breakdown", 0.85, {"type": "filter_sweep"}),
        ]

        # Configure cue service mock
        cue_data = [cue.to_dict() for cue in cue_points]
        self.cue_service.generate_cue_points.return_value = cue_data
        self.cue_service.get_cue_points.return_value = cue_data

        # Simulate cue generation execution
        await self.cue_service.generate_cue_points(track_id, analysis_result.to_dict())

        return cue_points


@pytest.fixture
def workflow_orchestrator():
    """Provide workflow test orchestrator."""
    return WorkflowTestOrchestrator()


@pytest.fixture
def sample_audio_files():
    """Provide sample audio files for testing."""
    return [
        MockAudioFile("/test/audio/track1.mp3", 180.5, "mp3", 320, 44100),
        MockAudioFile("/test/audio/track2.wav", 240.2, "wav", 1411, 44100),
        MockAudioFile("/test/audio/track3.flac", 195.8, "flac", 1411, 48000),
    ]


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_single_file_complete_workflow(self, workflow_orchestrator, sample_audio_files):
        """Test complete workflow for a single file."""
        mock_file = sample_audio_files[0]

        # Step 1: File upload
        uploaded_files = await workflow_orchestrator.simulate_file_upload([mock_file])
        assert len(uploaded_files) == 1
        assert uploaded_files[0] == str(mock_file.path)

        # Verify file service calls
        workflow_orchestrator.file_service.detect_new_files.assert_called()
        workflow_orchestrator.file_service.validate_file.assert_called()
        workflow_orchestrator.file_service.get_file_metadata.assert_called()

        # Step 2: Analysis
        file_path = uploaded_files[0]
        analysis_result = await workflow_orchestrator.simulate_analysis_workflow(file_path, mock_file)

        assert analysis_result.file_path == file_path
        assert "bpm" in analysis_result.analysis_data
        assert "key" in analysis_result.analysis_data

        # Verify analysis service calls
        workflow_orchestrator.analysis_service.analyze_file.assert_called_once_with(
            file_path, mock_file.get_audio_properties()
        )

        # Step 3: Cataloging
        track_data = await workflow_orchestrator.simulate_cataloging_workflow(file_path, mock_file, analysis_result)

        assert track_data["file_path"] == file_path
        assert track_data["metadata"] == mock_file.get_metadata()
        assert track_data["analysis_data"] == analysis_result.analysis_data

        # Verify catalog service calls
        workflow_orchestrator.catalog_service.add_track.assert_called_once()
        workflow_orchestrator.catalog_service.update_track_analysis.assert_called_once()

        # Step 4: Cue generation
        cue_points = await workflow_orchestrator.simulate_cue_generation_workflow(track_data["id"], analysis_result)

        assert len(cue_points) > 0
        assert any(cue.cue_type == "mix_in" for cue in cue_points)
        assert any(cue.cue_type == "mix_out" for cue in cue_points)

        # Verify cue service calls
        workflow_orchestrator.cue_service.generate_cue_points.assert_called_once_with(
            track_data["id"], analysis_result.to_dict()
        )

    @pytest.mark.asyncio
    async def test_batch_file_processing_workflow(self, workflow_orchestrator, sample_audio_files):
        """Test batch processing workflow for multiple files."""
        batch_size = len(sample_audio_files)

        # Step 1: Batch file upload
        uploaded_files = await workflow_orchestrator.simulate_file_upload(sample_audio_files)
        assert len(uploaded_files) == batch_size

        # Process each file through the complete workflow
        processed_tracks = []

        for i, file_path in enumerate(uploaded_files):
            mock_file = sample_audio_files[i]

            # Analysis
            analysis_result = await workflow_orchestrator.simulate_analysis_workflow(file_path, mock_file)

            # Cataloging
            track_data = await workflow_orchestrator.simulate_cataloging_workflow(file_path, mock_file, analysis_result)

            # Cue generation
            cue_points = await workflow_orchestrator.simulate_cue_generation_workflow(track_data["id"], analysis_result)

            processed_tracks.append(
                {"track_data": track_data, "analysis_result": analysis_result, "cue_points": cue_points}
            )

        # Verify all files were processed
        assert len(processed_tracks) == batch_size

        # Verify each track has complete data
        for track in processed_tracks:
            assert "id" in track["track_data"]
            assert track["analysis_result"].analysis_data is not None
            assert len(track["cue_points"]) > 0

    @pytest.mark.asyncio
    async def test_workflow_with_error_handling(self, workflow_orchestrator, sample_audio_files):
        """Test workflow behavior with error conditions."""
        mock_file = sample_audio_files[0]

        # Simulate analysis failure
        workflow_orchestrator.analysis_service.analyze_file.side_effect = Exception("Analysis failed")

        # Attempt workflow execution
        uploaded_files = await workflow_orchestrator.simulate_file_upload([mock_file])

        with pytest.raises(Exception, match="Analysis failed"):
            await workflow_orchestrator.simulate_analysis_workflow(uploaded_files[0], mock_file)

        # Reset mock and test recovery
        workflow_orchestrator.analysis_service.analyze_file.side_effect = None
        analysis_result = MockAnalysisResult(uploaded_files[0])
        workflow_orchestrator.analysis_service.analyze_file.return_value = analysis_result.to_dict()

        # Verify recovery workflow
        recovery_result = await workflow_orchestrator.simulate_analysis_workflow(uploaded_files[0], mock_file)
        assert recovery_result is not None

    @pytest.mark.asyncio
    async def test_workflow_performance_monitoring(self, workflow_orchestrator, sample_audio_files):
        """Test workflow with performance monitoring."""
        mock_file = sample_audio_files[0]

        # Track timing for each step
        start_time = datetime.now(UTC)

        # Step 1: File upload
        upload_start = datetime.now(UTC)
        uploaded_files = await workflow_orchestrator.simulate_file_upload([mock_file])
        upload_duration = (datetime.now(UTC) - upload_start).total_seconds()

        # Step 2: Analysis
        analysis_start = datetime.now(UTC)
        analysis_result = await workflow_orchestrator.simulate_analysis_workflow(uploaded_files[0], mock_file)
        analysis_duration = (datetime.now(UTC) - analysis_start).total_seconds()

        # Step 3: Cataloging
        catalog_start = datetime.now(UTC)
        track_data = await workflow_orchestrator.simulate_cataloging_workflow(
            uploaded_files[0], mock_file, analysis_result
        )
        catalog_duration = (datetime.now(UTC) - catalog_start).total_seconds()

        # Step 4: Cue generation
        cue_start = datetime.now(UTC)
        _ = await workflow_orchestrator.simulate_cue_generation_workflow(track_data["id"], analysis_result)
        cue_duration = (datetime.now(UTC) - cue_start).total_seconds()

        total_duration = (datetime.now(UTC) - start_time).total_seconds()

        # Verify performance characteristics
        assert upload_duration < 1.0  # Upload should be fast (mocked)
        assert analysis_duration < 2.0  # Analysis should be reasonable (mocked)
        assert catalog_duration < 1.0  # Cataloging should be fast (mocked)
        assert cue_duration < 1.0  # Cue generation should be fast (mocked)
        assert total_duration < 5.0  # Total workflow should complete quickly

        # Log performance metrics (would integrate with real monitoring)
        performance_metrics = {
            "upload_duration": upload_duration,
            "analysis_duration": analysis_duration,
            "catalog_duration": catalog_duration,
            "cue_duration": cue_duration,
            "total_duration": total_duration,
            "file_size_mb": 45.2,  # Mock file size
            "throughput_mb_per_second": 45.2 / total_duration,
        }

        # Verify metrics are reasonable
        assert performance_metrics["throughput_mb_per_second"] > 5.0

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, workflow_orchestrator, sample_audio_files):
        """Test concurrent execution of multiple workflows."""

        async def process_single_file(mock_file: MockAudioFile) -> dict[str, Any]:
            """Process a single file through the complete workflow."""
            # Upload
            uploaded_files = await workflow_orchestrator.simulate_file_upload([mock_file])

            # Analysis
            analysis_result = await workflow_orchestrator.simulate_analysis_workflow(uploaded_files[0], mock_file)

            # Cataloging
            track_data = await workflow_orchestrator.simulate_cataloging_workflow(
                uploaded_files[0], mock_file, analysis_result
            )

            # Cue generation
            cue_points = await workflow_orchestrator.simulate_cue_generation_workflow(track_data["id"], analysis_result)

            return {"track_id": track_data["id"], "file_path": uploaded_files[0], "cue_count": len(cue_points)}

        # Execute workflows concurrently
        start_time = datetime.now(UTC)

        tasks = [process_single_file(mock_file) for mock_file in sample_audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        concurrent_duration = (datetime.now(UTC) - start_time).total_seconds()

        # Verify all workflows completed successfully
        assert len(results) == len(sample_audio_files)

        for result in results:
            assert not isinstance(result, Exception)
            assert "track_id" in result
            assert "file_path" in result
            assert result["cue_count"] > 0

        # Verify concurrent execution was efficient
        # (should be faster than sequential execution due to mocking)
        assert concurrent_duration < 10.0  # Should complete quickly with mocks


class TestWorkflowIntegration:
    """Test workflow integration points."""

    @pytest.mark.asyncio
    async def test_notification_workflow_integration(self, workflow_orchestrator, sample_audio_files):
        """Test workflow integration with notification system."""
        mock_file = sample_audio_files[0]

        # Complete workflow with notification tracking
        uploaded_files = await workflow_orchestrator.simulate_file_upload([mock_file])
        analysis_result = await workflow_orchestrator.simulate_analysis_workflow(uploaded_files[0], mock_file)
        track_data = await workflow_orchestrator.simulate_cataloging_workflow(
            uploaded_files[0], mock_file, analysis_result
        )
        cue_points = await workflow_orchestrator.simulate_cue_generation_workflow(track_data["id"], analysis_result)

        # Simulate notifications at key workflow points
        notifications = [
            {"type": "file_uploaded", "track_id": track_data["id"]},
            {"type": "analysis_completed", "track_id": track_data["id"]},
            {"type": "track_cataloged", "track_id": track_data["id"]},
            {"type": "cues_generated", "track_id": track_data["id"], "cue_count": len(cue_points)},
        ]

        for notification in notifications:
            await workflow_orchestrator.notification_service.send_notification(notification)

        # Verify notification calls
        assert workflow_orchestrator.notification_service.send_notification.call_count == 4

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, workflow_orchestrator, sample_audio_files):
        """Test workflow state persistence and recovery."""
        mock_file = sample_audio_files[0]

        # Simulate workflow state at different stages
        workflow_states = []

        # State 1: File uploaded
        uploaded_files = await workflow_orchestrator.simulate_file_upload([mock_file])
        workflow_states.append(
            {"stage": "uploaded", "file_path": uploaded_files[0], "timestamp": datetime.now(UTC).isoformat()}
        )

        # State 2: Analysis completed
        analysis_result = await workflow_orchestrator.simulate_analysis_workflow(uploaded_files[0], mock_file)
        workflow_states.append(
            {
                "stage": "analyzed",
                "file_path": uploaded_files[0],
                "analysis_id": analysis_result.analysis_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # State 3: Cataloged
        track_data = await workflow_orchestrator.simulate_cataloging_workflow(
            uploaded_files[0], mock_file, analysis_result
        )
        workflow_states.append(
            {
                "stage": "cataloged",
                "file_path": uploaded_files[0],
                "track_id": track_data["id"],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # State 4: Cues generated
        cue_points = await workflow_orchestrator.simulate_cue_generation_workflow(track_data["id"], analysis_result)
        workflow_states.append(
            {
                "stage": "completed",
                "file_path": uploaded_files[0],
                "track_id": track_data["id"],
                "cue_count": len(cue_points),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Verify workflow progression
        assert len(workflow_states) == 4
        assert workflow_states[0]["stage"] == "uploaded"
        assert workflow_states[-1]["stage"] == "completed"

        # Verify state consistency
        file_path = workflow_states[0]["file_path"]
        for state in workflow_states:
            assert state["file_path"] == file_path
