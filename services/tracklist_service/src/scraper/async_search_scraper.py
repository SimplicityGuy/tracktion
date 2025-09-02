"""
Async search scraper for 1001tracklists.com.

Implements async search functionality for DJs, events, and tracks with circuit breaker support.
"""

import asyncio
import re
import time
from urllib.parse import urlencode, urljoin

import structlog
from bs4 import BeautifulSoup, NavigableString, PageElement, Tag

from services.tracklist_service.src.models.search import (
    PaginationInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchType,
)

from .async_base_scraper import AsyncScraperBase

logger = structlog.get_logger(__name__)


class AsyncSearchScraper(AsyncScraperBase):
    """Async scraper for searching 1001tracklists.com."""

    def __init__(self) -> None:
        """Initialize the async search scraper."""
        super().__init__(service_name="1001tracklists_search")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute an async search request.

        Args:
            request: Search request with query and filters

        Returns:
            Search response with results and pagination

        Raises:
            httpx.HTTPError: If scraping fails
        """
        start_time = time.time()

        try:
            # Build search URL based on search type
            search_url = self._build_search_url(request)

            # Get the search results page
            soup = await self.get_page(search_url)

            # Parse search results
            results = self._parse_search_results(soup, request.search_type)

            # Parse pagination info
            pagination = self._parse_pagination(soup, request.page, request.limit)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Build response
            response = SearchResponse(
                results=results,
                pagination=pagination,
                query_info={
                    "query": request.query,
                    "type": request.search_type.value,
                    "start_date": str(request.start_date) if request.start_date else None,
                    "end_date": str(request.end_date) if request.end_date else None,
                },
                cache_hit=False,
                response_time_ms=response_time_ms,
                correlation_id=request.correlation_id,
            )

            logger.info(
                "Search completed",
                query=request.query,
                search_type=request.search_type.value,
                results_count=len(results),
                response_time_ms=response_time_ms,
            )

            return response

        except Exception as e:
            logger.error(
                "Search failed",
                query=request.query,
                search_type=request.search_type.value,
                error=str(e),
                correlation_id=request.correlation_id,
            )
            raise

    async def batch_search(
        self,
        requests: list[SearchRequest],
        max_concurrent: int = 3,
    ) -> list[SearchResponse]:
        """Execute multiple search requests concurrently.

        Args:
            requests: List of search requests
            max_concurrent: Maximum concurrent searches

        Returns:
            List of search responses

        Raises:
            httpx.HTTPError: If any search fails
        """

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _search_one(req: SearchRequest) -> SearchResponse:
            async with semaphore:
                return await self.search(req)

        tasks = [_search_one(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            "Batch search completed",
            total_requests=len(requests),
            successful=len(results),
        )

        return results

    def _build_search_url(self, request: SearchRequest) -> str:
        """Build the search URL with parameters.

        Args:
            request: Search request

        Returns:
            Complete search URL
        """
        base_url = "https://www.1001tracklists.com"

        # Map search type to URL path
        if request.search_type == SearchType.DJ:
            path = "/search/result.php"
            params = {"search_selection": "2"}  # DJ search
        elif request.search_type == SearchType.EVENT:
            path = "/search/result.php"
            params = {"search_selection": "3"}  # Event search
        elif request.search_type == SearchType.TRACK:
            path = "/search/result.php"
            params = {"search_selection": "1"}  # Track search
        else:
            path = "/search/result.php"
            params = {}

        # Add search query
        params["main_search"] = request.query

        # Add date filters if provided
        if request.start_date:
            params["date_start"] = request.start_date.strftime("%Y-%m-%d")
        if request.end_date:
            params["date_end"] = request.end_date.strftime("%Y-%m-%d")

        # Add pagination
        if request.page > 1:
            params["page"] = str(request.page)

        # Build complete URL
        url = urljoin(base_url, path)
        if params:
            url = f"{url}?{urlencode(params)}"

        return url

    def _parse_search_results(self, soup: BeautifulSoup, search_type: SearchType) -> list[SearchResult]:
        """Parse search results from HTML.

        Args:
            soup: Parsed HTML
            search_type: Type of search

        Returns:
            List of search results
        """
        results = []

        # Find result containers based on search type
        if search_type == SearchType.DJ:
            containers = soup.find_all("div", class_="bTitle")
        elif search_type == SearchType.EVENT:
            containers = soup.find_all("div", class_="eventBlock")
        elif search_type == SearchType.TRACK:
            containers = soup.find_all("div", class_="trackItem")
        else:
            containers = soup.find_all("div", class_=["bTitle", "eventBlock", "trackItem"])

        for container in containers:
            result = self._parse_single_result(container, search_type)
            if result:
                results.append(result)

        return results

    def _parse_single_result(
        self, container: Tag | PageElement | NavigableString, search_type: SearchType
    ) -> SearchResult | None:
        """Parse a single search result.

        Args:
            container: HTML container for the result
            search_type: Type of search

        Returns:
            Parsed search result or None if parsing fails
        """
        try:
            # Ensure container is a Tag
            if not isinstance(container, Tag):
                return None

            # Extract common fields
            title = ""
            url = ""
            description = ""
            date = None

            # Parse based on search type
            if search_type == SearchType.DJ:
                title_elem = container.find("a")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get("href", "") if isinstance(title_elem, Tag) else ""
                    url = href if isinstance(href, str) else ""

            elif search_type == SearchType.EVENT:
                title_elem = container.find("h4")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                link_elem = container.find("a")
                if link_elem:
                    href = link_elem.get("href", "") if isinstance(link_elem, Tag) else ""
                    url = href if isinstance(href, str) else ""

            elif search_type == SearchType.TRACK:
                title_elem = container.find("span", class_="trackName")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                artist_elem = container.find("span", class_="artistName")
                if artist_elem:
                    description = artist_elem.get_text(strip=True)

            # Build result
            if title and url:
                return SearchResult(
                    dj_name=description or title or "Unknown DJ",  # Use description or title as DJ name
                    event_name=None,
                    date=date,
                    venue=None,
                    set_type=None,
                    url=url if url.startswith("http") else f"https://www.1001tracklists.com{url}",
                    duration=None,
                    track_count=None,
                    genre=None,
                    description=title if description else None,
                    source_url=url if url.startswith("http") else f"https://www.1001tracklists.com{url}",
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to parse search result: {e}")
            return None

    def _parse_pagination(self, soup: BeautifulSoup, current_page: int, limit: int) -> PaginationInfo:
        """Parse pagination information from HTML.

        Args:
            soup: Parsed HTML
            current_page: Current page number
            limit: Results per page

        Returns:
            Pagination information
        """
        total_results = 0
        total_pages = 1

        # Try to find pagination elements
        pagination_elem = soup.find("div", class_="pagination")
        if pagination_elem and isinstance(pagination_elem, Tag):
            # Look for total results count
            results_text = pagination_elem.get_text()

            match = re.search(r"(\d+)\s+results?", results_text, re.IGNORECASE)
            if match:
                total_results = int(match.group(1))

            # Calculate total pages
            if total_results > 0:
                total_pages = (total_results + limit - 1) // limit

            # Find last page link
            last_link = pagination_elem.find_all("a")
            if last_link:
                for link in reversed(last_link):
                    href_attr = link.get("href", "") if isinstance(link, Tag) else ""
                    href = href_attr if isinstance(href_attr, str) else ""
                    if "page=" in href:
                        match = re.search(r"page=(\d+)", href)
                        if match:
                            total_pages = int(match.group(1))
                            break

        return PaginationInfo(
            page=current_page,
            total_pages=max(total_pages, 1),
            total_items=total_results,
            limit=limit,
            has_next=current_page < total_pages,
            has_previous=current_page > 1,
        )
