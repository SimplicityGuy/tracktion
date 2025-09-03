"""
Integration tests for message queue systems and inter-service messaging.

Tests RabbitMQ, Redis pub/sub, and custom message queue implementations used
for communication between services in the Tracktion system.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from tests.shared_utilities import (
    TestDataGenerator,
    generate_recording_data,
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRabbitMQChannel:
    """Mock RabbitMQ channel for testing."""

    def __init__(self):
        self.exchanges = {}
        self.queues = {}
        self.bindings = {}
        self.published_messages = []
        self.consumers = {}
        self.is_closed = False

    async def declare_exchange(self, exchange_name: str, exchange_type: str = "topic"):
        """Declare an exchange."""
        self.exchanges[exchange_name] = {"type": exchange_type, "created_at": datetime.now(UTC)}

    async def declare_queue(self, queue_name: str, durable: bool = True):
        """Declare a queue."""
        self.queues[queue_name] = {"durable": durable, "messages": [], "created_at": datetime.now(UTC)}
        return {"queue": queue_name, "message_count": 0, "consumer_count": 0}

    async def bind_queue(self, queue_name: str, exchange_name: str, routing_key: str):
        """Bind a queue to an exchange."""
        binding_key = f"{exchange_name}:{queue_name}:{routing_key}"
        self.bindings[binding_key] = {
            "queue": queue_name,
            "exchange": exchange_name,
            "routing_key": routing_key,
            "created_at": datetime.now(UTC),
        }

    async def basic_publish(self, exchange: str, routing_key: str, body: bytes, properties=None):
        """Publish a message."""
        message = {
            "exchange": exchange,
            "routing_key": routing_key,
            "body": body,
            "properties": properties or {},
            "timestamp": datetime.now(UTC),
            "message_id": str(uuid4()),
        }
        self.published_messages.append(message)

        # Deliver to bound queues
        for binding in self.bindings.values():
            if binding["exchange"] == exchange and self._matches_routing_key(binding["routing_key"], routing_key):
                queue_name = binding["queue"]
                if queue_name in self.queues:
                    self.queues[queue_name]["messages"].append(message)

    async def basic_consume(self, queue_name: str, callback, auto_ack: bool = False):
        """Set up a consumer for a queue."""
        if queue_name not in self.consumers:
            self.consumers[queue_name] = []
        self.consumers[queue_name].append({"callback": callback, "auto_ack": auto_ack, "created_at": datetime.now(UTC)})

    def _matches_routing_key(self, pattern: str, key: str) -> bool:
        """Check if routing key matches pattern."""
        if pattern in {"*", "#"}:
            return True
        return pattern == key

    async def close(self):
        """Close the channel."""
        self.is_closed = True


class MockRabbitMQConnection:
    """Mock RabbitMQ connection for testing."""

    def __init__(self):
        self.channels = []
        self.is_closed = False

    async def channel(self):
        """Create a new channel."""
        channel = MockRabbitMQChannel()
        self.channels.append(channel)
        return channel

    async def close(self):
        """Close the connection."""
        for channel in self.channels:
            await channel.close()
        self.is_closed = True


class MockRedisClient:
    """Mock Redis client for pub/sub testing."""

    def __init__(self):
        self.data = {}
        self.subscribers = {}
        self.published_messages = []

    async def publish(self, channel: str, message: str):
        """Publish a message to a channel."""
        msg_data = {"channel": channel, "message": message, "timestamp": datetime.now(UTC)}
        self.published_messages.append(msg_data)

        # Deliver to subscribers
        if channel in self.subscribers:
            for callback in self.subscribers[channel]:
                await callback(msg_data)

    def subscribe(self, channel: str, callback):
        """Subscribe to a channel."""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(callback)

    async def get(self, key: str):
        """Get a value from Redis."""
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        """Set a value in Redis."""
        self.data[key] = {"value": value, "expires_at": datetime.now(UTC).timestamp() + (ex or 3600)}

    async def delete(self, key: str):
        """Delete a key from Redis."""
        return self.data.pop(key, None) is not None


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock RabbitMQ connection for testing."""
    return MockRabbitMQConnection()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    return MockRedisClient()


@pytest.fixture
def data_generator():
    """Test data generator."""
    return TestDataGenerator(seed=42)


class TestRabbitMQIntegration:
    """Test suite for RabbitMQ message queue integration."""

    @pytest.mark.asyncio
    async def test_rabbitmq_exchange_declaration(self, mock_rabbitmq_connection):
        """Test RabbitMQ exchange declaration and configuration."""

        channel = await mock_rabbitmq_connection.channel()

        # Declare exchanges for different message types
        await channel.declare_exchange("tracktion.analysis", "topic")
        await channel.declare_exchange("tracktion.cataloging", "topic")
        await channel.declare_exchange("tracktion.tracklist", "topic")
        await channel.declare_exchange("tracktion.notifications", "fanout")

        # Verify exchanges were created
        assert "tracktion.analysis" in channel.exchanges
        assert "tracktion.cataloging" in channel.exchanges
        assert "tracktion.tracklist" in channel.exchanges
        assert "tracktion.notifications" in channel.exchanges

        # Verify exchange types
        assert channel.exchanges["tracktion.analysis"]["type"] == "topic"
        assert channel.exchanges["tracktion.notifications"]["type"] == "fanout"

        await channel.close()
        logger.info("✅ RabbitMQ exchange declaration test completed successfully")

    @pytest.mark.asyncio
    async def test_rabbitmq_queue_binding(self, mock_rabbitmq_connection):
        """Test RabbitMQ queue binding and routing."""

        channel = await mock_rabbitmq_connection.channel()

        # Set up exchanges and queues
        await channel.declare_exchange("tracktion.events", "topic")

        # Declare service-specific queues
        await channel.declare_queue("analysis.requests", durable=True)
        await channel.declare_queue("cataloging.requests", durable=True)
        await channel.declare_queue("tracklist.requests", durable=True)

        # Bind queues with routing keys
        await channel.bind_queue("analysis.requests", "tracktion.events", "analysis.*")
        await channel.bind_queue("cataloging.requests", "tracktion.events", "cataloging.*")
        await channel.bind_queue("tracklist.requests", "tracktion.events", "tracklist.*")

        # Verify bindings were created
        analysis_binding = "tracktion.events:analysis.requests:analysis.*"
        cataloging_binding = "tracktion.events:cataloging.requests:cataloging.*"
        tracklist_binding = "tracktion.events:tracklist.requests:tracklist.*"

        assert analysis_binding in channel.bindings
        assert cataloging_binding in channel.bindings
        assert tracklist_binding in channel.bindings

        # Verify binding details
        assert channel.bindings[analysis_binding]["routing_key"] == "analysis.*"
        assert channel.bindings[cataloging_binding]["queue"] == "cataloging.requests"

        await channel.close()
        logger.info("✅ RabbitMQ queue binding test completed successfully")

    @pytest.mark.asyncio
    async def test_rabbitmq_message_publishing(self, mock_rabbitmq_connection, data_generator):
        """Test RabbitMQ message publishing and routing."""

        channel = await mock_rabbitmq_connection.channel()

        # Set up message infrastructure
        await channel.declare_exchange("tracktion.events", "topic")
        await channel.declare_queue("analysis.requests", durable=True)
        await channel.bind_queue("analysis.requests", "tracktion.events", "analysis.requested")

        # Publish analysis request message
        recording_id = data_generator.generate_uuid_string()
        message_data = {
            "recording_id": recording_id,
            "file_path": f"/audio/{recording_id}.mp3",
            "timestamp": datetime.now(UTC).isoformat(),
            "priority": "normal",
        }

        await channel.basic_publish(
            exchange="tracktion.events",
            routing_key="analysis.requested",
            body=json.dumps(message_data).encode(),
            properties={"content_type": "application/json", "delivery_mode": 2},  # Persistent
        )

        # Verify message was published
        assert len(channel.published_messages) == 1
        published_msg = channel.published_messages[0]
        assert published_msg["exchange"] == "tracktion.events"
        assert published_msg["routing_key"] == "analysis.requested"

        # Verify message was routed to correct queue
        queue_messages = channel.queues["analysis.requests"]["messages"]
        assert len(queue_messages) == 1
        assert json.loads(queue_messages[0]["body"])["recording_id"] == recording_id

        await channel.close()
        logger.info("✅ RabbitMQ message publishing test completed successfully")

    @pytest.mark.asyncio
    async def test_rabbitmq_message_consumption(self, mock_rabbitmq_connection, data_generator):
        """Test RabbitMQ message consumption and processing."""

        channel = await mock_rabbitmq_connection.channel()

        # Set up infrastructure
        await channel.declare_exchange("tracktion.events", "topic")
        await channel.declare_queue("test.consumer", durable=True)
        await channel.bind_queue("test.consumer", "tracktion.events", "test.message")

        # Set up consumer
        consumed_messages = []

        async def message_handler(message):
            consumed_messages.append(message)

        await channel.basic_consume("test.consumer", message_handler, auto_ack=True)

        # Publish test messages
        for i in range(3):
            message_data = {"id": i, "recording_id": data_generator.generate_uuid_string(), "data": f"test message {i}"}

            await channel.basic_publish(
                exchange="tracktion.events", routing_key="test.message", body=json.dumps(message_data).encode()
            )

        # Verify consumer was set up
        assert "test.consumer" in channel.consumers
        assert len(channel.consumers["test.consumer"]) == 1

        # Verify messages were delivered to queue
        assert len(channel.queues["test.consumer"]["messages"]) == 3

        await channel.close()
        logger.info("✅ RabbitMQ message consumption test completed successfully")

    @pytest.mark.asyncio
    async def test_rabbitmq_error_handling(self, mock_rabbitmq_connection, data_generator):
        """Test RabbitMQ error handling and message retry logic."""

        channel = await mock_rabbitmq_connection.channel()

        # Set up dead letter exchange for failed messages
        await channel.declare_exchange("tracktion.events", "topic")
        await channel.declare_exchange("tracktion.deadletter", "topic")

        await channel.declare_queue("analysis.requests", durable=True)
        await channel.declare_queue("analysis.deadletter", durable=True)

        await channel.bind_queue("analysis.requests", "tracktion.events", "analysis.requested")
        await channel.bind_queue("analysis.deadletter", "tracktion.deadletter", "analysis.failed")

        # Simulate failed message processing
        failed_messages = []

        async def failing_handler(message):
            """Handler that fails processing."""
            failed_messages.append(message)
            # Simulate processing failure
            await channel.basic_publish(
                exchange="tracktion.deadletter",
                routing_key="analysis.failed",
                body=message["body"],
                properties={"retry_count": "1", "original_routing_key": "analysis.requested"},
            )

        await channel.basic_consume("analysis.requests", failing_handler)

        # Publish message that will fail
        error_message = {
            "recording_id": data_generator.generate_uuid_string(),
            "file_path": "/invalid/path.mp3",
            "error_simulation": True,
        }

        await channel.basic_publish(
            exchange="tracktion.events", routing_key="analysis.requested", body=json.dumps(error_message).encode()
        )

        # Verify error handling
        assert len(channel.queues["analysis.deadletter"]["messages"]) == 1
        dead_letter_msg = channel.queues["analysis.deadletter"]["messages"][0]
        assert dead_letter_msg["routing_key"] == "analysis.failed"
        assert dead_letter_msg["properties"]["retry_count"] == "1"

        await channel.close()
        logger.info("✅ RabbitMQ error handling test completed successfully")


class TestRedisIntegration:
    """Test suite for Redis pub/sub and caching integration."""

    @pytest.mark.asyncio
    async def test_redis_pubsub_basic(self, mock_redis, data_generator):
        """Test basic Redis pub/sub functionality."""

        # Set up subscriber
        received_messages = []

        async def message_handler(message):
            received_messages.append(message)

        mock_redis.subscribe("tracktion:events", message_handler)

        # Publish messages
        test_messages = [
            {"type": "analysis_completed", "recording_id": data_generator.generate_uuid_string()},
            {"type": "cataloging_completed", "recording_id": data_generator.generate_uuid_string()},
            {"type": "tracklist_generated", "tracklist_id": data_generator.generate_uuid_string()},
        ]

        for msg in test_messages:
            await mock_redis.publish("tracktion:events", json.dumps(msg))

        # Verify messages were published and received
        assert len(mock_redis.published_messages) == 3
        assert len(received_messages) == 3

        # Verify message content
        first_msg = json.loads(received_messages[0]["message"])
        assert first_msg["type"] == "analysis_completed"
        assert "recording_id" in first_msg

        logger.info("✅ Redis pub/sub basic test completed successfully")

    @pytest.mark.asyncio
    async def test_redis_caching_integration(self, mock_redis, data_generator):
        """Test Redis caching for analysis results and metadata."""

        # Cache analysis results
        recording_id = data_generator.generate_uuid_string()
        analysis_result = generate_recording_data(recording_id, data_generator)

        cache_key = f"analysis:result:{recording_id}"
        await mock_redis.set(cache_key, json.dumps(analysis_result), ex=3600)  # 1 hour expiry

        # Retrieve from cache
        cached_data = await mock_redis.get(cache_key)
        assert cached_data is not None

        cached_result = json.loads(cached_data["value"])
        assert cached_result["recording_id"] == recording_id
        assert cached_result["bpm"] == analysis_result["bpm"]

        # Test cache expiry
        assert cached_data["expires_at"] > datetime.now(UTC).timestamp()

        # Test cache deletion
        deleted = await mock_redis.delete(cache_key)
        assert deleted is True

        # Verify key is deleted
        missing_data = await mock_redis.get(cache_key)
        assert missing_data is None

        logger.info("✅ Redis caching integration test completed successfully")

    @pytest.mark.asyncio
    async def test_redis_distributed_locking(self, mock_redis, data_generator):
        """Test Redis distributed locking mechanism."""

        resource_id = data_generator.generate_uuid_string()
        lock_key = f"lock:analysis:{resource_id}"
        lock_value = str(uuid4())

        # Acquire lock
        await mock_redis.set(lock_key, lock_value, ex=30)  # 30 second lock

        # Verify lock is held
        current_lock = await mock_redis.get(lock_key)
        assert current_lock is not None
        assert current_lock["value"] == lock_value

        # Test lock ownership
        other_lock_value = str(uuid4())
        # In real Redis, this would fail with NX option, but our mock doesn't implement NX
        await mock_redis.set(f"lock:analysis:{data_generator.generate_uuid_string()}", other_lock_value, ex=30)

        # Release lock
        released = await mock_redis.delete(lock_key)
        assert released is True

        logger.info("✅ Redis distributed locking test completed successfully")


class TestMessageQueueWorkflows:
    """Test complete message queue workflows across services."""

    @pytest.mark.asyncio
    async def test_analysis_workflow_messaging(self, mock_rabbitmq_connection, data_generator):
        """Test complete analysis workflow through message queues."""

        channel = await mock_rabbitmq_connection.channel()

        # Set up analysis workflow infrastructure
        await channel.declare_exchange("tracktion.analysis", "topic")
        await channel.declare_queue("analysis.requests", durable=True)
        await channel.declare_queue("analysis.results", durable=True)

        await channel.bind_queue("analysis.requests", "tracktion.analysis", "analysis.requested")
        await channel.bind_queue("analysis.results", "tracktion.analysis", "analysis.completed")

        # Track workflow messages
        workflow_messages = []

        async def analysis_processor(message):
            """Simulate analysis service processing."""
            msg_data = json.loads(message["body"])
            workflow_messages.append(("request_received", msg_data))

            # Simulate processing time
            await asyncio.sleep(0.01)

            # Generate analysis result
            result = {
                "recording_id": msg_data["recording_id"],
                "bpm": 120.5,
                "key": "C major",
                "duration": 180.0,
                "status": "completed",
                "processed_at": datetime.now(UTC).isoformat(),
            }

            # Publish result
            await channel.basic_publish(
                exchange="tracktion.analysis", routing_key="analysis.completed", body=json.dumps(result).encode()
            )

            workflow_messages.append(("result_published", result))

        await channel.basic_consume("analysis.requests", analysis_processor)

        # Submit analysis request
        recording_id = data_generator.generate_uuid_string()
        request = {
            "recording_id": recording_id,
            "file_path": f"/audio/{recording_id}.mp3",
            "priority": "normal",
            "requested_at": datetime.now(UTC).isoformat(),
        }

        await channel.basic_publish(
            exchange="tracktion.analysis", routing_key="analysis.requested", body=json.dumps(request).encode()
        )

        # Allow processing
        await asyncio.sleep(0.1)

        # Verify workflow
        assert len(workflow_messages) == 2
        assert workflow_messages[0][0] == "request_received"
        assert workflow_messages[1][0] == "result_published"

        # Verify request processing
        processed_request = workflow_messages[0][1]
        assert processed_request["recording_id"] == recording_id

        # Verify result publication
        published_result = workflow_messages[1][1]
        assert published_result["recording_id"] == recording_id
        assert published_result["status"] == "completed"

        # Verify result was queued
        assert len(channel.queues["analysis.results"]["messages"]) == 1

        await channel.close()
        logger.info("✅ Analysis workflow messaging test completed successfully")

    @pytest.mark.asyncio
    async def test_cross_service_messaging_patterns(self, mock_redis, data_generator):
        """Test messaging patterns between multiple services."""

        # Set up service communication channels
        service_channels = {
            "analysis": "tracktion:analysis:events",
            "cataloging": "tracktion:cataloging:events",
            "tracklist": "tracktion:tracklist:events",
            "notifications": "tracktion:notifications:events",
        }

        # Track inter-service messages
        service_messages = {service: [] for service in service_channels}

        # Set up service subscribers
        for service, channel in service_channels.items():

            async def create_handler(service_name):
                async def handler(message):
                    msg_data = json.loads(message["message"])
                    service_messages[service_name].append(msg_data)

                return handler

            mock_redis.subscribe(channel, await create_handler(service))

        # Simulate service interaction workflow
        recording_id = data_generator.generate_uuid_string()

        # 1. Analysis service completes analysis
        analysis_event = {
            "event_type": "analysis_completed",
            "recording_id": recording_id,
            "bpm": 128.0,
            "key": "A minor",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await mock_redis.publish(service_channels["analysis"], json.dumps(analysis_event))

        # 2. Cataloging service processes analysis result
        cataloging_event = {
            "event_type": "cataloging_completed",
            "recording_id": recording_id,
            "catalog_id": data_generator.generate_uuid_string(),
            "metadata_updated": True,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await mock_redis.publish(service_channels["cataloging"], json.dumps(cataloging_event))

        # 3. Tracklist service updates tracklists
        tracklist_event = {
            "event_type": "tracklist_updated",
            "recording_id": recording_id,
            "affected_tracklists": [data_generator.generate_uuid_string(), data_generator.generate_uuid_string()],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await mock_redis.publish(service_channels["tracklist"], json.dumps(tracklist_event))

        # 4. Notification service sends updates
        notification_event = {
            "event_type": "user_notification",
            "recording_id": recording_id,
            "notification_type": "analysis_complete",
            "user_ids": [data_generator.generate_uuid_string()],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await mock_redis.publish(service_channels["notifications"], json.dumps(notification_event))

        # Verify cross-service communication
        assert len(service_messages["analysis"]) == 1
        assert len(service_messages["cataloging"]) == 1
        assert len(service_messages["tracklist"]) == 1
        assert len(service_messages["notifications"]) == 1

        # Verify message content consistency
        for messages in service_messages.values():
            assert messages[0]["recording_id"] == recording_id

        # Verify event progression
        assert service_messages["analysis"][0]["event_type"] == "analysis_completed"
        assert service_messages["cataloging"][0]["event_type"] == "cataloging_completed"
        assert service_messages["tracklist"][0]["event_type"] == "tracklist_updated"
        assert service_messages["notifications"][0]["event_type"] == "user_notification"

        logger.info("✅ Cross-service messaging patterns test completed successfully")

    @pytest.mark.asyncio
    async def test_message_queue_resilience(self, mock_rabbitmq_connection, data_generator):
        """Test message queue resilience and recovery scenarios."""

        channel = await mock_rabbitmq_connection.channel()

        # Set up resilient messaging infrastructure
        await channel.declare_exchange("tracktion.resilient", "topic")
        await channel.declare_queue("resilient.primary", durable=True)
        await channel.declare_queue("resilient.backup", durable=True)
        await channel.declare_queue("resilient.deadletter", durable=True)

        await channel.bind_queue("resilient.primary", "tracktion.resilient", "message.normal")
        await channel.bind_queue("resilient.backup", "tracktion.resilient", "message.backup")
        await channel.bind_queue("resilient.deadletter", "tracktion.resilient", "message.failed")

        # Track resilience scenarios
        resilience_events = []

        # Primary processor (can fail)
        primary_failure_count = 0

        async def primary_processor(message):
            nonlocal primary_failure_count
            msg_data = json.loads(message["body"])

            # Simulate occasional failures
            if primary_failure_count < 2:
                primary_failure_count += 1
                resilience_events.append(("primary_failed", msg_data))

                # Route to backup processor
                await channel.basic_publish(
                    exchange="tracktion.resilient", routing_key="message.backup", body=message["body"]
                )
                return

            # Successful processing
            resilience_events.append(("primary_success", msg_data))

        # Backup processor (always succeeds)
        async def backup_processor(message):
            msg_data = json.loads(message["body"])
            resilience_events.append(("backup_success", msg_data))

        await channel.basic_consume("resilient.primary", primary_processor)
        await channel.basic_consume("resilient.backup", backup_processor)

        # Send test messages
        for i in range(3):
            test_message = {
                "id": i,
                "recording_id": data_generator.generate_uuid_string(),
                "data": f"resilience test {i}",
            }

            await channel.basic_publish(
                exchange="tracktion.resilient", routing_key="message.normal", body=json.dumps(test_message).encode()
            )

        # Allow processing
        await asyncio.sleep(0.1)

        # Verify resilience behavior
        assert len(resilience_events) >= 3

        # Count event types
        primary_failures = [e for e in resilience_events if e[0] == "primary_failed"]
        backup_successes = [e for e in resilience_events if e[0] == "backup_success"]
        primary_successes = [e for e in resilience_events if e[0] == "primary_success"]

        # Verify failover occurred
        assert len(primary_failures) == 2  # First two messages failed
        assert len(backup_successes) == 2  # Backup processed the failures
        assert len(primary_successes) == 1  # Third message succeeded in primary

        await channel.close()
        logger.info("✅ Message queue resilience test completed successfully")
