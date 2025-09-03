"""
Unit tests for BPM data storage in StorageHandler.

Tests storage of BPM detection and temporal analysis results.
"""

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from services.analysis_service.src.storage_handler import StorageError, StorageHandler


class TestStorageHandlerBPM:
    """Test suite for BPM data storage functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock environment variables
        with (
            patch.dict(
                "os.environ",
                {
                    "NEO4J_URI": "bolt://localhost:7687",
                    "NEO4J_USER": "neo4j",
                    "NEO4J_PASSWORD": "password",
                },
            ),
            patch("services.analysis_service.src.storage_handler.get_db_session"),
            patch("services.analysis_service.src.storage_handler.RecordingRepository"),
            patch("services.analysis_service.src.storage_handler.MetadataRepository"),
            patch("services.analysis_service.src.storage_handler.Neo4jRepository"),
        ):
            # Mock repository initialization
            self.handler = StorageHandler()

        # Mock repositories
        self.handler.recording_repo = MagicMock()
        self.handler.metadata_repo = MagicMock()
        self.handler.neo4j_repo = MagicMock()

        # Test data
        self.recording_id = uuid4()
        self.correlation_id = "test-correlation-123"

    def test_store_bpm_data_success(self):
        """Test successful storage of BPM data."""
        bpm_data = {
            "bpm": 128.5,
            "confidence": 0.95,
            "algorithm": "RhythmExtractor2013",
        }

        temporal_data = {
            "start_bpm": 126.0,
            "end_bpm": 130.0,
            "average_bpm": 128.5,
            "stability_score": 0.88,
            "temporal_bpm": [126.0, 127.5, 128.0, 129.0, 130.0],
        }

        # Mock successful storage
        with patch.object(self.handler, "store_metadata", return_value=True) as mock_store:
            result = self.handler.store_bpm_data(
                self.recording_id,
                bpm_data,
                temporal_data,
                self.correlation_id,
            )

            assert result is True

            # Verify metadata was prepared correctly
            mock_store.assert_called_once()
            stored_metadata = mock_store.call_args[0][1]

            assert stored_metadata["bpm_average"] == "128.5"
            assert stored_metadata["bpm_confidence"] == "0.95"
            assert stored_metadata["bpm_algorithm"] == "RhythmExtractor2013"
            assert stored_metadata["bpm_start"] == "126.0"
            assert stored_metadata["bpm_end"] == "130.0"
            assert stored_metadata["bpm_stability"] == "0.88"
            assert stored_metadata["bpm_status"] == "completed"

            # Verify temporal data is JSON serialized
            temporal_json = json.loads(stored_metadata["bpm_temporal"])
            assert temporal_json == [126.0, 127.5, 128.0, 129.0, 130.0]

    def test_store_bpm_data_without_temporal(self):
        """Test storing BPM data without temporal analysis."""
        bpm_data = {
            "bpm": 120.0,
            "confidence": 0.85,
            "algorithm": "PercivalBpmEstimator",
        }

        with patch.object(self.handler, "store_metadata", return_value=True) as mock_store:
            result = self.handler.store_bpm_data(
                self.recording_id,
                bpm_data,
                None,  # No temporal data
                self.correlation_id,
            )

            assert result is True

            stored_metadata = mock_store.call_args[0][1]
            assert stored_metadata["bpm_average"] == "120.0"
            assert "bpm_start" not in stored_metadata
            assert "bpm_end" not in stored_metadata
            assert "bpm_stability" not in stored_metadata
            assert "bpm_temporal" not in stored_metadata

    def test_store_bpm_data_with_error(self):
        """Test storing failed BPM detection."""
        bpm_data = {
            "error": "Audio file corrupted",
        }

        with patch.object(self.handler, "store_metadata", return_value=True) as mock_store:
            result = self.handler.store_bpm_data(
                self.recording_id,
                bpm_data,
                None,
                self.correlation_id,
            )

            assert result is True

            stored_metadata = mock_store.call_args[0][1]
            assert stored_metadata["bpm_error"] == "Audio file corrupted"
            assert stored_metadata["bpm_status"] == "failed"
            assert "bpm_average" not in stored_metadata

    def test_store_bpm_data_storage_failure(self):
        """Test handling of storage failures."""
        bpm_data = {"bpm": 128.0, "confidence": 0.9}

        # Mock storage failure
        with (
            patch.object(self.handler, "store_metadata", side_effect=StorageError("Database error")),
            pytest.raises(StorageError, match="Failed to store BPM data"),
        ):
            self.handler.store_bpm_data(
                self.recording_id,
                bpm_data,
                None,
                self.correlation_id,
            )

    def test_create_bpm_relationships(self):
        """Test creation of BPM relationships in Neo4j."""
        metadata = {
            "bpm_average": "128.0",
            "bpm_confidence": "0.95",
            "bpm_stability": "0.85",
        }

        # Mock Neo4j methods
        self.handler.neo4j_repo.create_or_get_bpm_range.return_value = "bpm_range_123"
        self.handler.neo4j_repo.create_or_get_tempo_type.return_value = "tempo_type_456"

        self.handler._create_bpm_relationships(
            self.recording_id,
            metadata,
            self.correlation_id,
        )

        # Verify BPM range was created
        self.handler.neo4j_repo.create_or_get_bpm_range.assert_called_once_with("fast")

        # Verify relationships were created
        assert self.handler.neo4j_repo.create_relationship.call_count == 2

        # Check BPM range relationship
        bpm_range_call = self.handler.neo4j_repo.create_relationship.call_args_list[0]
        assert bpm_range_call[1]["from_id"] == self.recording_id
        assert bpm_range_call[1]["to_id"] == "bpm_range_123"
        assert bpm_range_call[1]["relationship_type"] == "HAS_BPM_RANGE"
        assert bpm_range_call[1]["properties"]["bpm"] == 128.0

        # Check tempo type relationship
        tempo_call = self.handler.neo4j_repo.create_relationship.call_args_list[1]
        assert tempo_call[1]["from_id"] == self.recording_id
        assert tempo_call[1]["to_id"] == "tempo_type_456"
        assert tempo_call[1]["relationship_type"] == "HAS_TEMPO_TYPE"
        assert tempo_call[1]["properties"]["stability_score"] == 0.85

    def test_create_bpm_relationships_no_neo4j(self):
        """Test BPM relationships when Neo4j is not available."""
        self.handler.neo4j_repo = None
        metadata = {"bpm_average": "128.0"}

        # Should not raise an error
        self.handler._create_bpm_relationships(
            self.recording_id,
            metadata,
            self.correlation_id,
        )

    def test_create_bpm_relationships_invalid_data(self):
        """Test BPM relationships with invalid data."""
        metadata = {
            "bpm_average": "not_a_number",
            "bpm_stability": "also_not_a_number",
        }

        # Should handle gracefully without raising
        self.handler._create_bpm_relationships(
            self.recording_id,
            metadata,
            self.correlation_id,
        )

        # No relationships should be created
        self.handler.neo4j_repo.create_relationship.assert_not_called()

    def test_get_bpm_range(self):
        """Test BPM range categorization."""
        test_cases = [
            (40.0, "very_slow"),  # Largo
            (65.0, "slow"),  # Adagio
            (90.0, "moderate"),  # Andante
            (115.0, "moderate_fast"),  # Allegretto
            (128.0, "fast"),  # Allegro
            (150.0, "very_fast"),  # Vivace
            (175.0, "extremely_fast"),  # Presto
            (220.0, "ultra_fast"),  # Prestissimo
        ]

        for bpm, expected_range in test_cases:
            result = self.handler._get_bpm_range(bpm)
            assert result == expected_range, f"BPM {bpm} should be in range {expected_range}"

    def test_get_bpm_statistics(self):
        """Test getting BPM statistics."""
        # Mock metadata items - setup different returns for each recording
        mock_metadata_1 = [
            MagicMock(key="bpm_average", value="128.0"),
            MagicMock(key="bpm_confidence", value="0.95"),
        ]
        mock_metadata_2 = [
            MagicMock(key="bpm_average", value="100.0"),  # moderate range
            MagicMock(key="bpm_confidence", value="0.88"),
        ]

        mock_metadata_3 = [
            MagicMock(key="bpm_average", value="invalid"),  # Invalid value
        ]

        self.handler.metadata_repo.get_by_recording.side_effect = [
            mock_metadata_1,
            mock_metadata_2,
            mock_metadata_3,
        ]

        recording_ids = [uuid4(), uuid4(), uuid4()]
        stats = self.handler.get_bpm_statistics(recording_ids)

        assert stats["total_analyzed"] == 3  # 2 successful + 1 failed
        assert stats["successful"] == 2
        assert stats["failed"] == 1
        assert stats["average_bpm"] == 114.0  # (128 + 100) / 2
        assert stats["average_confidence"] == 0.915  # (0.95 + 0.88) / 2
        assert stats["bpm_ranges"]["fast"] == 1  # 128 BPM
        assert stats["bpm_ranges"]["moderate"] == 1  # 100 BPM

    def test_get_bpm_statistics_no_data(self):
        """Test getting statistics with no BPM data."""
        self.handler.metadata_repo.get_by_recording.return_value = []

        stats = self.handler.get_bpm_statistics([uuid4()])

        assert stats["total_analyzed"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["average_bpm"] is None
        assert stats["average_confidence"] is None
        assert stats["bpm_ranges"] == {}

    def test_get_bpm_statistics_no_metadata_repo(self):
        """Test getting statistics when metadata repo is not available."""
        self.handler.metadata_repo = None

        stats = self.handler.get_bpm_statistics()

        assert stats["total_analyzed"] == 0
        assert stats["average_bpm"] is None

    def test_get_bpm_statistics_exception(self):
        """Test handling of exceptions in statistics calculation."""
        self.handler.metadata_repo.get_by_recording.side_effect = Exception("Database error")

        stats = self.handler.get_bpm_statistics([uuid4()])

        assert "error" in stats
        assert "Database error" in stats["error"]

    def test_store_bpm_data_with_partial_temporal(self):
        """Test storing BPM data with partial temporal data."""
        bpm_data = {"bpm": 140.0, "confidence": 0.75}

        temporal_data = {
            "start_bpm": 138.0,
            # No end_bpm
            "stability_score": 0.92,
            # No temporal_bpm array
        }

        with patch.object(self.handler, "store_metadata", return_value=True) as mock_store:
            result = self.handler.store_bpm_data(
                self.recording_id,
                bpm_data,
                temporal_data,
                self.correlation_id,
            )

            assert result is True

            stored_metadata = mock_store.call_args[0][1]
            assert stored_metadata["bpm_start"] == "138.0"
            assert "bpm_end" not in stored_metadata  # Missing
            assert stored_metadata["bpm_stability"] == "0.92"
            assert "bpm_temporal" not in stored_metadata  # Missing

    def test_create_bpm_relationships_with_variable_tempo(self):
        """Test BPM relationships for variable tempo tracks."""
        metadata = {
            "bpm_average": "95.0",
            "bpm_stability": "0.3",  # Low stability = variable tempo
        }

        self.handler.neo4j_repo.create_or_get_tempo_type.return_value = "tempo_type_var"

        self.handler._create_bpm_relationships(
            self.recording_id,
            metadata,
            self.correlation_id,
        )

        # Verify variable tempo type was created
        self.handler.neo4j_repo.create_or_get_tempo_type.assert_called_once_with("variable")


class TestStorageHandlerBPMIntegration:
    """Integration tests for BPM storage with real database operations."""

    @pytest.mark.integration
    @pytest.mark.requires_db
    def test_full_bpm_storage_flow(self):
        """Test complete BPM storage flow with real databases."""
        # This would test with actual PostgreSQL and Neo4j connections
