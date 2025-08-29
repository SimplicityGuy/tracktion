"""
Async base scraper class for 1001tracklists.com web scraping.

Provides async session management, rate limiting, circuit breaker, and anti-detection features.
"""

import asyncio
import random
import time
from typing import Any, Dict, Optional

import httpx
import structlog
from bs4 import BeautifulSoup

from shared.utils.async_http_client import (
    AsyncHTTPClient,
    HTTPClientConfig,
    RetryHandler,
    get_global_http_factory,
)

from ..config import get_config

logger = structlog.get_logger(__name__)


class AsyncScraperBase:
    """Async base class for web scraping with anti-detection, circuit breaker, and rate limiting."""

    def __init__(self, service_name: str = "1001tracklists") -> None:
        """Initialize the async base scraper.

        Args:
            service_name: Name of the service for circuit breaker identification
        """
        self.config = get_config().scraping
        self.service_name = service_name
        self.last_request_time = 0.0

        # Configure HTTP client with scraper-specific settings
        http_config = HTTPClientConfig(
            timeout=self.config.request_timeout,
            max_keepalive_connections=10,
            max_connections=20,
            user_agent=random.choice(self.config.user_agents),
            retry_attempts=3,
            retry_delay=1.0,
            retry_max_delay=10.0,
            circuit_breaker_fail_max=5,
            circuit_breaker_reset_timeout=60,
        )

        self.factory = get_global_http_factory(http_config)
        self.retry_handler = RetryHandler(http_config)
        self.http_client = AsyncHTTPClient(self.factory, self.retry_handler)
        self._current_user_agent = http_config.user_agent

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            # Add some randomness to avoid detection
            sleep_time += random.uniform(0, 0.5)
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

    async def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """Make an async HTTP request with retry logic, circuit breaker, and rate limiting.

        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: POST data
            headers: Additional headers

        Returns:
            Response object

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        await self._apply_rate_limit()

        # Prepare headers with current user agent
        request_headers = {
            "User-Agent": self._current_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        if headers:
            request_headers.update(headers)

        try:
            response = await self.http_client.request_with_circuit_breaker(
                service_name=self.service_name,
                method=method,
                url=url,
                params=params,
                data=data,
                headers=request_headers,
            )
            return response
        except Exception as e:
            logger.error(
                "Request failed",
                url=url,
                method=method,
                error=str(e),
                service=self.service_name,
            )
            raise

    def parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content into BeautifulSoup object.

        Args:
            html_content: Raw HTML string

        Returns:
            BeautifulSoup object for parsing
        """
        return BeautifulSoup(html_content, "lxml")

    async def get_page(self, url: str, params: Optional[Dict[str, Any]] = None) -> BeautifulSoup:
        """Get a page and return parsed HTML.

        Args:
            url: URL to fetch
            params: Query parameters

        Returns:
            Parsed HTML as BeautifulSoup object

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await self._make_request(url, params=params)
        return self.parse_html(response.text)

    async def batch_get_pages(
        self,
        urls: list[str],
        max_concurrent: int = 5,
    ) -> list[BeautifulSoup]:
        """Fetch multiple pages concurrently with rate limiting.

        Args:
            urls: List of URLs to fetch
            max_concurrent: Maximum concurrent requests

        Returns:
            List of parsed HTML pages

        Raises:
            httpx.HTTPError: If any request fails
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _fetch_one(url: str) -> BeautifulSoup:
            async with semaphore:
                return await self.get_page(url)

        tasks = [_fetch_one(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    def rotate_user_agent(self) -> None:
        """Rotate to a new random user agent."""
        self._current_user_agent = random.choice(self.config.user_agents)
        logger.debug("Rotated user agent", new_agent=self._current_user_agent)

    async def close(self) -> None:
        """Close the HTTP client sessions."""
        # The global factory will handle cleanup
        pass
