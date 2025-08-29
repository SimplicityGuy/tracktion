"""Integration tests for async API endpoints."""

import asyncio
import json
from typing import Any

import httpx
import pytest
import websockets
from fastapi import status

# Base URL for testing (should be configured via environment)
BASE_URL = "http://localhost:8000/v1"
WS_URL = "ws://localhost:8000/v1/ws"


class TestAsyncAPIIntegration:
    """Test async API integration."""

    @pytest.fixture
    async def async_client(self):
        """Create async HTTP client."""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        """Test health check endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling multiple concurrent requests."""
        # Create concurrent health check requests
        tasks = []
        for _ in range(50):
            task = async_client.get("/health")
            tasks.append(task)

        # Execute concurrently
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client):
        """Test rate limiting behavior."""
        # Exceed rate limit with rapid requests
        responses = []
        for _ in range(30):  # More than burst size
            response = await async_client.get("/health")
            responses.append(response)

        # Some requests should be rate limited
        rate_limited = [r for r in responses if r.status_code == status.HTTP_429_TOO_MANY_REQUESTS]
        assert len(rate_limited) > 0

        # Check rate limit headers
        for response in responses:
            if "X-RateLimit-Limit" in response.headers:
                assert int(response.headers["X-RateLimit-Limit"]) > 0
            if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_request_timeout(self, async_client):
        """Test request timeout handling."""
        # Test endpoint with delay parameter
        try:
            response = await async_client.get(
                "/analysis/slow",
                params={"delay": 35},  # Longer than default timeout
                timeout=40.0,
            )
            # Should timeout or return timeout error
            assert response.status_code in [status.HTTP_408_REQUEST_TIMEOUT, status.HTTP_504_GATEWAY_TIMEOUT]
        except httpx.ReadTimeout:
            # Expected timeout
            pass

    @pytest.mark.asyncio
    async def test_streaming_response(self, async_client):
        """Test streaming response endpoint."""
        chunks = []
        async with async_client.stream("GET", "/streaming/ndjson") as response:
            assert response.status_code == status.HTTP_200_OK
            async for chunk in response.aiter_text():
                if chunk.strip():
                    chunks.append(json.loads(chunk))

        assert len(chunks) > 0
        for chunk in chunks:
            assert "data" in chunk or "event" in chunk

    @pytest.mark.asyncio
    async def test_server_sent_events(self, async_client):
        """Test Server-Sent Events endpoint."""
        events = []
        async with async_client.stream("GET", "/streaming/events") as response:
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/event-stream"

            # Read a few events
            event_count = 0
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(line[5:].strip())
                    event_count += 1
                    if event_count >= 3:
                        break

        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_batch_processing(self, async_client):
        """Test batch processing endpoint."""
        batch_data = {
            "items": [
                {"id": 1, "data": "item1"},
                {"id": 2, "data": "item2"},
                {"id": 3, "data": "item3"},
            ]
        }

        response = await async_client.post("/analysis/batch", json=batch_data)

        assert response.status_code == status.HTTP_200_OK
        results = response.json()
        assert "results" in results
        assert len(results["results"]) == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, async_client):
        """Test error handling and propagation."""
        # Test 404
        response = await async_client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "X-Request-ID" in response.headers

        # Test validation error
        response = await async_client.post("/analysis/validate", json={"invalid": "data"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        error_detail = response.json()
        assert "detail" in error_detail


class TestWebSocketIntegration:
    """Test WebSocket integration."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test basic WebSocket connection."""
        async with websockets.connect(WS_URL) as websocket:
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))

            # Receive pong
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "pong"

    @pytest.mark.asyncio
    async def test_websocket_subscribe(self):
        """Test WebSocket subscription."""
        async with websockets.connect(WS_URL) as websocket:
            # Subscribe to progress updates
            await websocket.send(json.dumps({"type": "subscribe", "channel": "progress"}))

            # Receive subscription confirmation
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "subscribed"
            assert data["channel"] == "progress"

    @pytest.mark.asyncio
    async def test_websocket_broadcast(self):
        """Test WebSocket broadcast functionality."""
        clients = []

        # Connect multiple clients
        for _i in range(3):
            ws = await websockets.connect(WS_URL)
            clients.append(ws)

            # Subscribe to broadcast channel
            await ws.send(json.dumps({"type": "subscribe", "channel": "broadcast"}))

            # Wait for subscription confirmation
            await ws.recv()

        # Send broadcast from first client
        await clients[0].send(json.dumps({"type": "broadcast", "data": "test message"}))

        # All clients should receive the broadcast
        for client in clients:
            response = await asyncio.wait_for(client.recv(), timeout=2.0)
            data = json.loads(response)
            assert data["type"] == "broadcast"
            assert data["data"] == "test message"

        # Clean up
        for client in clients:
            await client.close()

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling."""
        async with websockets.connect(WS_URL) as websocket:
            # Send invalid message
            await websocket.send("invalid json")

            # Should receive error response
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "error"
            assert "message" in data

    @pytest.mark.asyncio
    async def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat/keepalive."""
        async with websockets.connect(WS_URL) as websocket:
            # Send initial ping
            await websocket.send(json.dumps({"type": "ping"}))
            await websocket.recv()  # pong

            # Wait for heartbeat
            await asyncio.sleep(2)

            # Connection should still be alive
            await websocket.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            data = json.loads(response)
            assert data["type"] == "pong"


class TestConcurrentLoad:
    """Test concurrent load handling."""

    @pytest.mark.asyncio
    async def test_high_concurrent_load(self):
        """Test handling high concurrent load."""

        async def make_request(client: httpx.AsyncClient, endpoint: str) -> dict[str, Any]:
            """Make a single request."""
            try:
                response = await client.get(endpoint)
                return {
                    "status": response.status_code,
                    "headers": dict(response.headers),
                    "success": response.status_code == status.HTTP_200_OK,
                }
            except Exception as e:
                return {"status": 0, "error": str(e), "success": False}

        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # Create many concurrent requests
            tasks = []
            endpoints = ["/health", "/metadata", "/recordings"]

            for i in range(100):
                endpoint = endpoints[i % len(endpoints)]
                task = make_request(client, endpoint)
                tasks.append(task)

            # Execute all requests
            results = await asyncio.gather(*tasks)

            # Analyze results
            successful = [r for r in results if r["success"]]
            rate_limited = [r for r in results if r["status"] == status.HTTP_429_TOO_MANY_REQUESTS]
            errors = [r for r in results if "error" in r]

            # Most requests should succeed or be rate limited
            assert len(successful) + len(rate_limited) >= 90

            # Rate limiting should kick in
            assert len(rate_limited) > 0

            print(f"Results: {len(successful)} successful, {len(rate_limited)} rate limited, {len(errors)} errors")

    @pytest.mark.asyncio
    async def test_connection_pool_limits(self):
        """Test connection pool and concurrency limits."""
        # Create many simultaneous connections
        connections = []

        try:
            for _i in range(15):  # More than per-IP limit
                ws = await websockets.connect(WS_URL)
                connections.append(ws)
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
            # Expected when connection limit reached
            pass

        # Should have some connections but not all
        assert 1 <= len(connections) <= 10  # Per-IP limit is 10

        # Clean up
        for conn in connections:
            await conn.close()


class TestAPIDocumentation:
    """Test API documentation and OpenAPI schema."""

    @pytest.fixture
    async def async_client(self):
        """Create async HTTP client."""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            yield client

    @pytest.mark.asyncio
    async def test_openapi_schema(self, async_client):
        """Test OpenAPI schema availability."""
        response = await async_client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Analysis Service API"
        assert "paths" in schema

    @pytest.mark.asyncio
    async def test_swagger_ui(self, async_client):
        """Test Swagger UI availability."""
        response = await async_client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_redoc_ui(self, async_client):
        """Test ReDoc UI availability."""
        response = await async_client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]


# Benchmark test (optional, requires pytest-benchmark)
@pytest.mark.benchmark
class TestPerformanceBenchmark:
    """Performance benchmark tests."""

    @pytest.mark.asyncio
    async def test_request_latency(self, benchmark):
        """Benchmark request latency."""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:

            async def make_request():
                response = await client.get("/health")
                return response.status_code

            result = await benchmark(make_request)
            assert result == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_concurrent_throughput(self, benchmark):
        """Benchmark concurrent request throughput."""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:

            async def make_concurrent_requests():
                tasks = [client.get("/health") for _ in range(10)]
                responses = await asyncio.gather(*tasks)
                return [r.status_code for r in responses]

            results = await benchmark(make_concurrent_requests)
            assert all(r == status.HTTP_200_OK for r in results)
