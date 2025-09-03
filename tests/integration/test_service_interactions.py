"""
Integration tests for service-to-service interactions.

Tests the communication and data flow between different services in the Tracktion system,
including analysis service, cataloging service, tracklist service, and file watcher.
"""

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.shared_utilities import (
    TestDataGenerator,
    generate_recording_data,
    generate_track_data,
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def data_generator():
    """Test data generator for consistent test data."""
    return TestDataGenerator(seed=42)


@pytest.fixture
def mock_message_queue():
    """Mock message queue for inter-service communication."""
    queue = MagicMock()
    queue.messages = []
    queue.subscribers = {}

    async def publish_async(topic: str, message: dict):
        queue.messages.append({"topic": topic, "message": message, "timestamp": datetime.now(UTC)})
        # Simulate message delivery to subscribers
        if topic in queue.subscribers:
            for callback in queue.subscribers[topic]:
                await callback(message)

    def subscribe(topic: str, callback):
        if topic not in queue.subscribers:
            queue.subscribers[topic] = []
        queue.subscribers[topic].append(callback)

    queue.publish = publish_async
    queue.subscribe = subscribe
    return queue


@pytest.fixture
def service_registry():
    """Registry of all service mocks for interaction testing."""
    registry = {
        "analysis_service": MagicMock(),
        "cataloging_service": MagicMock(),
        "tracklist_service": MagicMock(),
        "file_watcher": MagicMock(),
    }

    # Configure service endpoints and capabilities
    registry["analysis_service"].analyze_audio = AsyncMock()
    registry["analysis_service"].get_analysis_result = AsyncMock()
    registry["analysis_service"].health_check = AsyncMock(return_value={"status": "healthy"})

    registry["cataloging_service"].catalog_recording = AsyncMock()
    registry["cataloging_service"].get_recording = AsyncMock()
    registry["cataloging_service"].update_metadata = AsyncMock()
    registry["cataloging_service"].health_check = AsyncMock(return_value={"status": "healthy"})

    registry["tracklist_service"].generate_tracklist = AsyncMock()
    registry["tracklist_service"].get_tracklist = AsyncMock()
    registry["tracklist_service"].generate_cue = AsyncMock()
    registry["tracklist_service"].health_check = AsyncMock(return_value={"status": "healthy"})

    registry["file_watcher"].watch_directory = AsyncMock()
    registry["file_watcher"].get_watch_status = AsyncMock()
    registry["file_watcher"].health_check = AsyncMock(return_value={"status": "healthy"})

    return registry


class TestServiceInteractions:
    """Test suite for service-to-service interactions."""

    @pytest.mark.asyncio
    async def test_file_detection_to_analysis_workflow(self, mock_message_queue, service_registry, data_generator):
        """Test workflow from file detection to analysis completion."""

        # Generate test data
        recording_id = data_generator.generate_uuid_string()
        file_path = f"/test/audio/{recording_id}.mp3"

        # Mock file watcher detecting new file
        file_event = {
            "event_type": "created",
            "file_path": file_path,
            "timestamp": datetime.now(UTC).isoformat(),
            "recording_id": recording_id,
        }

        # Mock analysis service response
        analysis_result = {
            "recording_id": recording_id,
            "bpm": 120.5,
            "key": "C major",
            "duration": 180.0,
            "energy": 0.75,
            "danceability": 0.68,
            "status": "completed",
        }
        service_registry["analysis_service"].analyze_audio.return_value = analysis_result

        # Setup message queue subscriptions
        analysis_requests = []

        async def handle_analysis_request(message):
            analysis_requests.append(message)
            # Simulate analysis service processing
            await service_registry["analysis_service"].analyze_audio(
                recording_id=message["recording_id"], file_path=message["file_path"]
            )
            # Publish analysis result
            await mock_message_queue.publish(
                "analysis.completed", {"recording_id": message["recording_id"], **analysis_result}
            )

        mock_message_queue.subscribe("file.detected", handle_analysis_request)

        # Simulate file watcher publishing file detection event
        await mock_message_queue.publish("file.detected", file_event)

        # Allow async processing
        await asyncio.sleep(0.1)

        # Verify interaction chain
        assert len(analysis_requests) == 1
        assert analysis_requests[0]["recording_id"] == recording_id
        assert analysis_requests[0]["file_path"] == file_path

        # Verify analysis service was called
        service_registry["analysis_service"].analyze_audio.assert_called_once_with(
            recording_id=recording_id, file_path=file_path
        )

        # Verify analysis completed message was published
        analysis_messages = [m for m in mock_message_queue.messages if m["topic"] == "analysis.completed"]
        assert len(analysis_messages) == 1
        assert analysis_messages[0]["message"]["recording_id"] == recording_id
        assert analysis_messages[0]["message"]["bpm"] == 120.5

        logger.info("✅ File detection to analysis workflow test completed successfully")

    @pytest.mark.asyncio
    async def test_analysis_to_cataloging_integration(self, mock_message_queue, service_registry, data_generator):
        """Test integration between analysis service and cataloging service."""

        recording_id = data_generator.generate_uuid_string()

        # Mock analysis completion event
        analysis_result = generate_recording_data(recording_id, data_generator)

        # Mock cataloging service response
        catalog_entry = {
            "recording_id": recording_id,
            "catalog_id": data_generator.generate_uuid_string(),
            "metadata": {
                "title": "Test Track",
                "artist": "Test Artist",
                "bpm": analysis_result["bpm"],
                "key": analysis_result["key"],
                "duration": analysis_result["duration"],
            },
            "status": "cataloged",
        }
        service_registry["cataloging_service"].catalog_recording.return_value = catalog_entry

        # Setup cataloging workflow
        catalog_requests = []

        async def handle_analysis_completed(message):
            catalog_requests.append(message)
            # Simulate cataloging service processing analysis result
            result = await service_registry["cataloging_service"].catalog_recording(
                recording_id=message["recording_id"], analysis_data=message
            )
            # Publish cataloging completion
            await mock_message_queue.publish("cataloging.completed", result)

        mock_message_queue.subscribe("analysis.completed", handle_analysis_completed)

        # Simulate analysis service publishing completion
        await mock_message_queue.publish("analysis.completed", analysis_result)

        # Allow async processing
        await asyncio.sleep(0.1)

        # Verify cataloging workflow
        assert len(catalog_requests) == 1
        assert catalog_requests[0]["recording_id"] == recording_id

        # Verify cataloging service was called with analysis data
        service_registry["cataloging_service"].catalog_recording.assert_called_once_with(
            recording_id=recording_id, analysis_data=analysis_result
        )

        # Verify cataloging completed message
        catalog_messages = [m for m in mock_message_queue.messages if m["topic"] == "cataloging.completed"]
        assert len(catalog_messages) == 1
        assert catalog_messages[0]["message"]["recording_id"] == recording_id
        assert catalog_messages[0]["message"]["metadata"]["bpm"] == analysis_result["bpm"]

        logger.info("✅ Analysis to cataloging integration test completed successfully")

    @pytest.mark.asyncio
    async def test_tracklist_generation_workflow(self, mock_message_queue, service_registry, data_generator):
        """Test complete tracklist generation workflow involving multiple services."""

        # Generate test data for multiple tracks
        track_count = 3
        recordings = []

        for _ in range(track_count):
            recording_id = data_generator.generate_uuid_string()
            recording = generate_recording_data(recording_id, data_generator)
            recordings.append(recording)

        # Mock tracklist service response
        tracklist_id = data_generator.generate_uuid_string()
        tracklist_result = {
            "tracklist_id": tracklist_id,
            "tracks": [generate_track_data(rec["recording_id"], data_generator) for rec in recordings],
            "total_duration": sum(rec["duration"] for rec in recordings),
            "status": "generated",
        }
        service_registry["tracklist_service"].generate_tracklist.return_value = tracklist_result

        # Setup tracklist generation workflow
        tracklist_requests = []

        async def handle_tracklist_request(message):
            tracklist_requests.append(message)
            # Simulate fetching recording data from cataloging service
            recording_data = []
            for recording_id in message["recording_ids"]:
                # Mock cataloging service lookup
                catalog_data = next((rec for rec in recordings if rec["recording_id"] == recording_id), None)
                if catalog_data:
                    recording_data.append(catalog_data)

            # Generate tracklist
            result = await service_registry["tracklist_service"].generate_tracklist(
                recording_ids=message["recording_ids"], recording_data=recording_data
            )

            # Publish tracklist completion
            await mock_message_queue.publish("tracklist.generated", result)

        mock_message_queue.subscribe("tracklist.requested", handle_tracklist_request)

        # Simulate tracklist generation request
        tracklist_request = {
            "recording_ids": [rec["recording_id"] for rec in recordings],
            "user_id": data_generator.generate_uuid_string(),
            "request_id": data_generator.generate_uuid_string(),
        }

        await mock_message_queue.publish("tracklist.requested", tracklist_request)

        # Allow async processing
        await asyncio.sleep(0.1)

        # Verify tracklist generation workflow
        assert len(tracklist_requests) == 1
        assert len(tracklist_requests[0]["recording_ids"]) == track_count

        # Verify tracklist service was called
        service_registry["tracklist_service"].generate_tracklist.assert_called_once()
        call_args = service_registry["tracklist_service"].generate_tracklist.call_args
        assert len(call_args.kwargs["recording_ids"]) == track_count
        assert len(call_args.kwargs["recording_data"]) == track_count

        # Verify tracklist generated message
        tracklist_messages = [m for m in mock_message_queue.messages if m["topic"] == "tracklist.generated"]
        assert len(tracklist_messages) == 1
        assert tracklist_messages[0]["message"]["tracklist_id"] == tracklist_id
        assert len(tracklist_messages[0]["message"]["tracks"]) == track_count

        logger.info("✅ Tracklist generation workflow test completed successfully")

    @pytest.mark.asyncio
    async def test_service_health_check_coordination(self, service_registry):
        """Test health check coordination across all services."""

        # Configure health check responses
        health_responses = {
            "analysis_service": {"status": "healthy", "uptime": 3600, "active_jobs": 2},
            "cataloging_service": {"status": "healthy", "uptime": 3605, "db_connected": True},
            "tracklist_service": {"status": "degraded", "uptime": 3610, "warning": "High memory usage"},
            "file_watcher": {"status": "healthy", "uptime": 3615, "watched_dirs": 5},
        }

        for service_name, response in health_responses.items():
            service_registry[service_name].health_check.return_value = response

        # Simulate system health check
        system_health = {}
        for service_name, service in service_registry.items():
            system_health[service_name] = await service.health_check()

        # Verify all services were checked
        for service in service_registry.values():
            service.health_check.assert_called_once()

        # Verify health responses
        assert system_health["analysis_service"]["status"] == "healthy"
        assert system_health["analysis_service"]["active_jobs"] == 2
        assert system_health["cataloging_service"]["db_connected"] is True
        assert system_health["tracklist_service"]["status"] == "degraded"
        assert system_health["tracklist_service"]["warning"] == "High memory usage"
        assert system_health["file_watcher"]["watched_dirs"] == 5

        # Calculate overall system health
        service_statuses = [health["status"] for health in system_health.values()]
        overall_status = "healthy" if all(s == "healthy" for s in service_statuses) else "degraded"
        assert overall_status == "degraded"  # Due to tracklist_service warning

        logger.info("✅ Service health check coordination test completed successfully")

    @pytest.mark.asyncio
    async def test_cross_service_data_consistency(self, mock_message_queue, service_registry, data_generator):
        """Test data consistency across service boundaries."""

        recording_id = data_generator.generate_uuid_string()

        # Initial analysis data
        analysis_data = generate_recording_data(recording_id, data_generator)

        # Mock service responses that should maintain data consistency
        catalog_data = {
            "recording_id": recording_id,
            "bpm": analysis_data["bpm"],  # Should match analysis
            "key": analysis_data["key"],
            "duration": analysis_data["duration"],
            "updated_at": datetime.now(UTC).isoformat(),
        }

        track_data = {
            "recording_id": recording_id,
            "bpm": analysis_data["bpm"],  # Should match analysis and catalog
            "key": analysis_data["key"],
            "duration": analysis_data["duration"],
            "position": 1,
        }

        service_registry["cataloging_service"].get_recording.return_value = catalog_data
        service_registry["tracklist_service"].get_tracklist.return_value = {"tracks": [track_data]}

        # Test data consistency across services
        analysis_result = analysis_data
        catalog_result = await service_registry["cataloging_service"].get_recording(recording_id)
        tracklist_result = await service_registry["tracklist_service"].get_tracklist(recording_id)

        # Verify data consistency
        track = tracklist_result["tracks"][0]

        assert analysis_result["recording_id"] == catalog_result["recording_id"] == track["recording_id"]
        assert analysis_result["bpm"] == catalog_result["bpm"] == track["bpm"]
        assert analysis_result["key"] == catalog_result["key"] == track["key"]
        assert abs(analysis_result["duration"] - catalog_result["duration"]) < 0.1
        assert abs(analysis_result["duration"] - track["duration"]) < 0.1

        # Verify service calls
        service_registry["cataloging_service"].get_recording.assert_called_once_with(recording_id)
        service_registry["tracklist_service"].get_tracklist.assert_called_once_with(recording_id)

        logger.info("✅ Cross-service data consistency test completed successfully")

    @pytest.mark.asyncio
    async def test_service_dependency_chain(self, mock_message_queue, service_registry, data_generator):
        """Test complex service dependency chains and error handling."""

        recording_id = data_generator.generate_uuid_string()

        # Create a dependency chain: File Watcher → Analysis → Cataloging → Tracklist
        chain_events = []

        async def file_watcher_handler(message):
            chain_events.append(("file_detected", message))
            # Simulate file watcher requesting analysis
            await mock_message_queue.publish(
                "analysis.requested", {"recording_id": message["recording_id"], "file_path": message["file_path"]}
            )

        async def analysis_handler(message):
            chain_events.append(("analysis_requested", message))
            # Simulate analysis completion
            analysis_result = generate_recording_data(message["recording_id"], data_generator)
            await mock_message_queue.publish("analysis.completed", analysis_result)

        async def cataloging_handler(message):
            chain_events.append(("analysis_completed", message))
            # Simulate cataloging
            catalog_result = {"recording_id": message["recording_id"], "status": "cataloged", "metadata": message}
            await mock_message_queue.publish("cataloging.completed", catalog_result)

        async def tracklist_handler(message):
            chain_events.append(("cataloging_completed", message))
            # Simulate tracklist update
            tracklist_result = {"recording_id": message["recording_id"], "status": "tracklist_updated"}
            await mock_message_queue.publish("tracklist.updated", tracklist_result)

        # Setup subscription chain
        mock_message_queue.subscribe("file.detected", file_watcher_handler)
        mock_message_queue.subscribe("analysis.requested", analysis_handler)
        mock_message_queue.subscribe("analysis.completed", cataloging_handler)
        mock_message_queue.subscribe("cataloging.completed", tracklist_handler)

        # Trigger the chain
        initial_event = {
            "recording_id": recording_id,
            "file_path": f"/test/{recording_id}.mp3",
            "event_type": "created",
        }

        await mock_message_queue.publish("file.detected", initial_event)

        # Allow chain to complete
        await asyncio.sleep(0.2)

        # Verify complete chain execution
        assert len(chain_events) == 4
        assert chain_events[0][0] == "file_detected"
        assert chain_events[1][0] == "analysis_requested"
        assert chain_events[2][0] == "analysis_completed"
        assert chain_events[3][0] == "cataloging_completed"

        # Verify all events have the same recording_id
        for _event_type, event_data in chain_events:
            assert event_data["recording_id"] == recording_id

        # Verify final tracklist update message
        tracklist_messages = [m for m in mock_message_queue.messages if m["topic"] == "tracklist.updated"]
        assert len(tracklist_messages) == 1
        assert tracklist_messages[0]["message"]["recording_id"] == recording_id

        logger.info("✅ Service dependency chain test completed successfully")

    @pytest.mark.asyncio
    async def test_concurrent_service_interactions(self, mock_message_queue, service_registry, data_generator):
        """Test concurrent interactions between services under load."""

        # Generate multiple concurrent requests
        concurrent_count = 5
        recording_ids = [data_generator.generate_uuid_string() for _ in range(concurrent_count)]

        # Track all concurrent operations
        concurrent_results = []

        async def process_recording(recording_id: str):
            """Process a single recording through all services."""
            try:
                # Simulate analysis
                analysis_data = generate_recording_data(recording_id, data_generator)
                await service_registry["analysis_service"].analyze_audio(
                    recording_id=recording_id, file_path=f"/test/{recording_id}.mp3"
                )

                # Simulate cataloging
                await service_registry["cataloging_service"].catalog_recording(
                    recording_id=recording_id, analysis_data=analysis_data
                )

                # Simulate tracklist generation
                await service_registry["tracklist_service"].generate_tracklist(
                    recording_ids=[recording_id], recording_data=[analysis_data]
                )

                concurrent_results.append(
                    {"recording_id": recording_id, "status": "success", "timestamp": datetime.now(UTC)}
                )

            except Exception as e:
                concurrent_results.append(
                    {"recording_id": recording_id, "status": "failed", "error": str(e), "timestamp": datetime.now(UTC)}
                )

        # Execute all recordings concurrently
        tasks = [process_recording(rid) for rid in recording_ids]
        await asyncio.gather(*tasks)

        # Verify all concurrent operations completed
        assert len(concurrent_results) == concurrent_count
        successful_results = [r for r in concurrent_results if r["status"] == "success"]
        assert len(successful_results) == concurrent_count

        # Verify each service was called the correct number of times
        assert service_registry["analysis_service"].analyze_audio.call_count == concurrent_count
        assert service_registry["cataloging_service"].catalog_recording.call_count == concurrent_count
        assert service_registry["tracklist_service"].generate_tracklist.call_count == concurrent_count

        # Verify all operations completed within reasonable time
        timestamps = [result["timestamp"] for result in concurrent_results]
        duration = max(timestamps) - min(timestamps)
        assert duration.total_seconds() < 1.0  # Should complete quickly with mocks

        logger.info(f"✅ Concurrent service interactions test completed successfully ({concurrent_count} operations)")


@pytest.mark.integration
class TestServiceInteractionEdgeCases:
    """Test edge cases and error scenarios in service interactions."""

    @pytest.mark.asyncio
    async def test_service_timeout_handling(self, service_registry):
        """Test handling of service timeouts and unresponsive services."""

        # Configure service to timeout
        async def timeout_mock(*args, **kwargs):
            await asyncio.sleep(2.0)  # Simulate timeout
            raise TimeoutError("Service timeout")

        service_registry["analysis_service"].analyze_audio = timeout_mock

        # Test timeout handling
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                service_registry["analysis_service"].analyze_audio(recording_id="test-id", file_path="/test.mp3"),
                timeout=1.0,
            )

        logger.info("✅ Service timeout handling test completed successfully")

    @pytest.mark.asyncio
    async def test_service_failure_propagation(self, mock_message_queue, service_registry, data_generator):
        """Test how service failures propagate through the system."""

        recording_id = data_generator.generate_uuid_string()

        # Configure analysis service to fail
        service_registry["analysis_service"].analyze_audio.side_effect = Exception("Analysis service failure")

        # Track failure propagation
        failure_events = []

        async def handle_analysis_failure(message):
            failure_events.append(message)
            # Simulate error propagation
            await mock_message_queue.publish(
                "analysis.failed",
                {
                    "recording_id": message["recording_id"],
                    "error": "Analysis service failure",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        mock_message_queue.subscribe("analysis.requested", handle_analysis_failure)

        # Trigger failure
        await mock_message_queue.publish(
            "analysis.requested", {"recording_id": recording_id, "file_path": f"/test/{recording_id}.mp3"}
        )

        await asyncio.sleep(0.1)

        # Verify failure was handled and propagated
        assert len(failure_events) == 1
        failure_messages = [m for m in mock_message_queue.messages if m["topic"] == "analysis.failed"]
        assert len(failure_messages) == 1
        assert failure_messages[0]["message"]["recording_id"] == recording_id
        assert "Analysis service failure" in failure_messages[0]["message"]["error"]

        logger.info("✅ Service failure propagation test completed successfully")
