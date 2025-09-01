"""
Base scraper class for 1001tracklists.com web scraping.

Provides session management, rate limiting, and anti-detection features.
"""

import random
import time
from typing import Any

import requests  # type: ignore[import-untyped]  # types-requests not installed in this environment
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_config


class ScraperBase:
    """Base class for web scraping with anti-detection and rate limiting."""

    def __init__(self) -> None:
        """Initialize the base scraper."""
        self.config = get_config().scraping
        self.session = self._create_session()
        self.last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        """Create a session with proper headers and configuration."""
        session = requests.Session()

        # Set a random user agent
        session.headers.update(
            {
                "User-Agent": random.choice(self.config.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        return session

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            # Add some randomness to avoid detection
            sleep_time += random.uniform(0, 0.5)
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> requests.Response:
        """Make an HTTP request with retry logic and rate limiting.

        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: POST data

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails after retries
        """
        self._apply_rate_limit()

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            data=data,
            timeout=self.config.request_timeout,
        )

        response.raise_for_status()
        return response

    def parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content into BeautifulSoup object.

        Args:
            html_content: Raw HTML string

        Returns:
            BeautifulSoup object for parsing
        """
        return BeautifulSoup(html_content, "lxml")

    def get_page(self, url: str, params: dict[str, Any] | None = None) -> BeautifulSoup:
        """Get a page and return parsed HTML.

        Args:
            url: URL to fetch
            params: Query parameters

        Returns:
            Parsed HTML as BeautifulSoup object

        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request(url, params=params)
        return self.parse_html(response.text)

    def rotate_user_agent(self) -> None:
        """Rotate to a new random user agent."""
        new_agent = random.choice(self.config.user_agents)
        self.session.headers.update({"User-Agent": new_agent})

    def reset_session(self) -> None:
        """Reset the session with new headers."""
        self.session.close()
        self.session = self._create_session()
