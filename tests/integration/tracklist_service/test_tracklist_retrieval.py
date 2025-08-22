"""
Integration tests for tracklist retrieval workflow.

Tests the complete flow from URL to parsed tracklist data.
"""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aio_pika import IncomingMessage

from services.tracklist_service.src.cache.redis_cache import RedisCache
from services.tracklist_service.src.messaging.tracklist_handler import TracklistMessageHandler
from services.tracklist_service.src.models.tracklist_models import (
    Track,
    Tracklist,
    TracklistRequest,
)
from services.tracklist_service.src.scraper.tracklist_scraper import TracklistScraper


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """
    <html>
    <head>
        <meta property="og:title" content="Test DJ @ Test Event">
    </head>
    <body>
        <div class="tlHead">
            <h1 class="marL10">Test DJ</h1>
            <h2>Test Event 2024</h2>
        </div>
        <div class="tracklist">
            <div class="tlpTog">
                <span class="cue">00:00</span>
                <span class="artist">Artist 1</span>
                <span class="track">Track 1</span>
                <span class="label">Label 1</span>
            </div>
            <div class="tlpTog">
                <span class="cue">05:30</span>
                <span class="artist">Artist 2</span>
                <span class="track">Track 2</span>
            </div>
        </div>
    </body>
    </html>
    """


class TestTracklistRetrievalIntegration:
    """Integration tests for tracklist retrieval."""

    @pytest.mark.asyncio
    @patch("services.tracklist_service.src.scraper.tracklist_scraper.TracklistScraper._make_request")
    async def test_end_to_end_retrieval(self, mock_request, sample_html):
        """Test complete retrieval flow from URL to tracklist."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_request.return_value = mock_response

        # Create scraper and test retrieval
        scraper = TracklistScraper()
        url = "https://www.1001tracklists.com/tracklist/test"

        tracklist = scraper.scrape_tracklist(url)

        # Verify basic structure
        assert tracklist.url == url
        assert tracklist.dj_name == "Test DJ"
        assert tracklist.event_name == "Test Event 2024"
        assert len(tracklist.tracks) == 2

        # Verify track details
        assert tracklist.tracks[0].artist == "Artist 1"
        assert tracklist.tracks[0].title == "Track 1"
        assert tracklist.tracks[0].label == "Label 1"
        assert tracklist.tracks[0].timestamp.formatted_time == "00:00"

        assert tracklist.tracks[1].artist == "Artist 2"
        assert tracklist.tracks[1].title == "Track 2"
        assert tracklist.tracks[1].timestamp.formatted_time == "05:30"

    @pytest.mark.asyncio
    @patch("services.tracklist_service.src.cache.redis_cache.RedisCache.get")
    @patch("services.tracklist_service.src.cache.redis_cache.RedisCache.set")
    async def test_caching_workflow(self, mock_set, mock_get):
        """Test caching workflow for tracklist data."""
        # Setup
        cache = RedisCache()
        tracklist = Tracklist(
            url="https://www.1001tracklists.com/test",
            dj_name="Test DJ",
            tracks=[
                Track(number=1, artist="A1", title="T1"),
                Track(number=2, artist="A2", title="T2"),
            ],
        )

        # Test cache miss
        mock_get.return_value = None
        cached = await cache.get("tracklist:test")
        assert cached is None

        # Test cache set
        mock_set.return_value = True
        success = await cache.set(
            "tracklist:test",
            tracklist.model_dump_json(),
            ttl=3600,
        )
        assert success is True

        # Test cache hit
        mock_get.return_value = tracklist.model_dump_json()
        cached = await cache.get("tracklist:test")
        assert cached is not None

        # Verify cached data
        cached_tracklist = Tracklist.model_validate_json(cached)
        assert cached_tracklist.dj_name == "Test DJ"
        assert len(cached_tracklist.tracks) == 2

    @pytest.mark.asyncio
    @patch("services.tracklist_service.src.messaging.tracklist_handler.TracklistMessageHandler._make_request")
    async def test_message_queue_processing(self, mock_request):
        """Test message queue processing for async retrieval."""
        handler = TracklistMessageHandler()

        # Create test message
        request = TracklistRequest(
            url="https://www.1001tracklists.com/test",
            correlation_id=uuid4(),
        )

        message_body = {
            "request": request.model_dump(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Mock message
        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()
        mock_message.redelivered_count = 0
        mock_message.ack = AsyncMock()

        # Mock cache operations
        with patch.object(handler._cache, "get", AsyncMock(return_value=None)):
            with patch.object(handler._cache, "set", AsyncMock(return_value=True)):
                # Mock scraper
                mock_tracklist = Tracklist(
                    url=request.url,
                    dj_name="Test DJ",
                    tracks=[],
                )
                with patch.object(handler._scraper, "scrape_tracklist", return_value=mock_tracklist):
                    # Process message
                    await handler.process_tracklist_request(mock_message)

        # Verify message was acknowledged
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling throughout the retrieval workflow."""
        from services.tracklist_service.src.resilience.error_handler import (
            CircuitBreaker,
            ParseError,
            TracklistNotFoundError,
            retry,
        )

        # Test custom exceptions
        with pytest.raises(TracklistNotFoundError) as exc_info:
            raise TracklistNotFoundError("https://example.com/404")
        assert "not found" in str(exc_info.value).lower()

        with pytest.raises(ParseError) as exc_info:
            raise ParseError("Failed to parse tracks", element="div.track")
        assert exc_info.value.details.get("element") == "div.track"

        # Test retry decorator
        attempt_count = 0

        @retry(max_attempts=3, delay=0.01, backoff=1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        def failing_function():
            raise RuntimeError("Always fails")

        # First two calls should fail normally
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_function)

        # Third call should trigger circuit breaker
        from services.tracklist_service.src.resilience.error_handler import ScrapingError

        with pytest.raises(ScrapingError) as exc_info:
            breaker.call(failing_function)
        assert "Circuit breaker is open" in str(exc_info.value)


class TestPerformance:
    """Performance tests for tracklist retrieval."""

    @pytest.mark.asyncio
    async def test_parsing_performance(self):
        """Test parsing performance for large tracklists."""
        # Generate large HTML with many tracks
        tracks_html = ""
        for i in range(100):
            tracks_html += f"""
            <div class="tlpTog">
                <span class="cue">{i:02d}:00</span>
                <span class="artist">Artist {i}</span>
                <span class="track">Track {i}</span>
                <span class="label">Label {i}</span>
            </div>
            """

        html = f"""
        <html>
        <body>
            <div class="tlHead">
                <h1 class="marL10">Performance Test DJ</h1>
            </div>
            <div class="tracklist">
                {tracks_html}
            </div>
        </body>
        </html>
        """

        # Mock request
        with patch(
            "services.tracklist_service.src.scraper.tracklist_scraper.TracklistScraper._make_request"
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.text = html
            mock_request.return_value = mock_response

            # Measure parsing time
            import time

            scraper = TracklistScraper()

            start = time.time()
            tracklist = scraper.scrape_tracklist("https://www.1001tracklists.com/test")
            elapsed = time.time() - start

            # Verify results
            assert len(tracklist.tracks) == 100
            assert elapsed < 5.0  # Should parse within 5 seconds

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance under load."""
        cache = RedisCache()

        # Mock Redis client
        with patch.object(cache, "client") as mock_client:
            mock_client.get.return_value = None
            mock_client.setex.return_value = True

            # Test multiple cache operations
            tasks = []
            for i in range(100):
                key = f"tracklist:test_{i}"
                value = json.dumps({"id": i})
                tasks.append(cache.set(key, value, ttl=60))

            # Execute in parallel
            results = await asyncio.gather(*tasks)

            # Verify all succeeded
            assert all(results)
            assert mock_client.setex.call_count == 100


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_malformed_html(self):
        """Test handling of malformed HTML."""
        malformed_html = """
        <div class="tlHead"
            <h1>Broken DJ
        <div class="tracklist">
            <span class="cue">00:00
            <span class="artist">Artist
        """

        with patch(
            "services.tracklist_service.src.scraper.tracklist_scraper.TracklistScraper._make_request"
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.text = malformed_html
            mock_request.return_value = mock_response

            scraper = TracklistScraper()

            # Should still extract what it can
            tracklist = scraper.scrape_tracklist("https://www.1001tracklists.com/test")
            assert tracklist.dj_name  # Should have some DJ name
            # Tracks might be empty or partial

    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_html = """
        <html>
        <body>
            <div class="tlHead">
                <h1 class="marL10">Amélie Léns</h1>
            </div>
            <div class="tracklist">
                <div class="tlpTog">
                    <span class="artist">Kölsch</span>
                    <span class="track">Für Dich (夜明け Remix)</span>
                </div>
            </div>
        </body>
        </html>
        """

        with patch(
            "services.tracklist_service.src.scraper.tracklist_scraper.TracklistScraper._make_request"
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.text = unicode_html
            mock_request.return_value = mock_response

            scraper = TracklistScraper()
            tracklist = scraper.scrape_tracklist("https://www.1001tracklists.com/test")

            assert "Amélie" in tracklist.dj_name
            assert "Kölsch" in tracklist.tracks[0].artist
            assert "夜明け" in tracklist.tracks[0].title

    @pytest.mark.asyncio
    async def test_empty_tracklist(self):
        """Test handling of empty tracklist."""
        empty_html = """
        <html>
        <body>
            <div class="tlHead">
                <h1 class="marL10">DJ No Tracks</h1>
            </div>
            <div class="tracklist">
                <!-- No tracks here -->
            </div>
        </body>
        </html>
        """

        with patch(
            "services.tracklist_service.src.scraper.tracklist_scraper.TracklistScraper._make_request"
        ) as mock_request:
            mock_response = MagicMock()
            mock_response.text = empty_html
            mock_request.return_value = mock_response

            scraper = TracklistScraper()
            tracklist = scraper.scrape_tracklist("https://www.1001tracklists.com/test")

            assert tracklist.dj_name == "DJ No Tracks"
            assert len(tracklist.tracks) == 0

    @pytest.mark.asyncio
    async def test_network_timeout(self):
        """Test handling of network timeouts."""
        from services.tracklist_service.src.resilience.error_handler import async_retry

        call_count = 0

        @async_retry(max_attempts=3, delay=0.01, exceptions=(TimeoutError,))
        async def timeout_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Connection timeout")
            return "success"

        result = await timeout_function()
        assert result == "success"
        assert call_count == 3
