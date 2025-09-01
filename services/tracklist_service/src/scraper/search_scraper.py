"""
Search scraper for 1001tracklists.com.

Implements search functionality for DJs, events, and tracks.
"""

import contextlib
import logging
from urllib.parse import urlencode, urljoin
from uuid import uuid4

from bs4 import BeautifulSoup, Tag

from src.models.search_models import PaginationInfo, SearchRequest, SearchResponse, SearchResult, SearchType

from .base_scraper import ScraperBase

logger = logging.getLogger(__name__)


class SearchScraper(ScraperBase):
    """Scraper for searching 1001tracklists.com."""

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search request.

        Args:
            request: Search request with query and filters

        Returns:
            Search response with results and pagination

        Raises:
            requests.RequestException: If scraping fails
        """
        try:
            # Build search URL based on search type
            search_url = self._build_search_url(request)

            # Get the search results page
            soup = self.get_page(search_url)

            # Parse search results
            results = self._parse_search_results(soup, request.search_type)

            # Parse pagination info
            pagination = self._parse_pagination(soup, request.page, request.limit)

            # Build response
            response = SearchResponse(
                results=results,
                pagination=pagination,
                query_info={
                    "query": request.query,
                    "type": request.search_type.value,
                    "start_date": (str(request.start_date) if request.start_date else None),
                    "end_date": str(request.end_date) if request.end_date else None,
                },
                cache_hit=False,
                response_time_ms=0.0,  # Will be set by caller
                correlation_id=request.correlation_id,
            )

            logger.info(
                f"Search completed: query='{request.query}', type={request.search_type.value}, results={len(results)}"
            )

            return response

        except Exception as e:
            logger.error(f"Search failed for query '{request.query}': {e}")
            raise

    def _build_search_url(self, request: SearchRequest) -> str:
        """Build the search URL with parameters.

        Args:
            request: Search request

        Returns:
            Complete search URL
        """
        # Map search types to 1001tracklists endpoints
        search_paths = {
            SearchType.DJ: "/dj",
            SearchType.EVENT: "/event",
            SearchType.TRACK: "/track",
        }

        base_path = search_paths.get(request.search_type, "/search")

        # Build query parameters
        params = {
            "q": request.query,
            "page": request.page,
        }

        # Add date filters if provided
        if request.start_date:
            params["start_date"] = request.start_date.isoformat()
        if request.end_date:
            params["end_date"] = request.end_date.isoformat()

        # Construct full URL
        url = urljoin(self.config.base_url, base_path)
        if params:
            url += "?" + urlencode(params)

        return str(url)

    def _parse_search_results(self, soup: BeautifulSoup, search_type: SearchType) -> list[SearchResult]:
        """Parse search results from HTML.

        Args:
            soup: Parsed HTML page
            search_type: Type of search being performed

        Returns:
            List of search results
        """
        results = []

        # Find result containers (adjust selectors based on actual HTML structure)
        # These selectors are placeholders - need to be updated based on actual site
        result_containers = soup.find_all("div", class_="tlItem")

        for container in result_containers:
            if isinstance(container, Tag):  # Type guard to ensure it's a Tag
                result = self._parse_single_result(container, search_type)
                if result:
                    results.append(result)

        return results

    def _parse_single_result(self, container: Tag, search_type: SearchType) -> SearchResult | None:
        """Parse a single search result.

        Args:
            container: HTML element containing the result
            search_type: Type of search being performed

        Returns:
            Parsed search result or None if parsing fails
        """
        try:
            # Extract common fields
            # Note: These selectors need to be adjusted based on actual HTML
            title_elem = container.find("a", class_="tlLink")
            if not title_elem:
                return None

            href = title_elem.get("href", "") if isinstance(title_elem, Tag) else ""
            url = urljoin(self.config.base_url, href if isinstance(href, str) else "")

            # Extract DJ name
            dj_elem = container.find("span", class_="djName")
            dj_name = dj_elem.text.strip() if dj_elem else "Unknown DJ"

            # Extract event name
            event_elem = container.find("span", class_="eventName")
            event_name = event_elem.text.strip() if event_elem else None

            # Extract date
            date_elem = container.find("span", class_="tlDate")
            result_date = None
            if date_elem:
                with contextlib.suppress(Exception):
                    # Parse date from text (format may vary)
                    _ = date_elem.text.strip()  # date_text - parsing placeholder
                    # This is a placeholder - actual date parsing logic needed
                    # result_date = parse_date(date_text)

            # Extract venue
            venue_elem = container.find("span", class_="venue")
            venue = venue_elem.text.strip() if venue_elem else None

            # Extract set type
            set_type_elem = container.find("span", class_="setType")
            set_type = set_type_elem.text.strip() if set_type_elem else None

            # Extract track count
            track_count_elem = container.find("span", class_="trackCount")
            track_count = None
            if track_count_elem:
                try:
                    # Extract number from text like "25 tracks"
                    count_text = track_count_elem.text.strip()
                    track_count = int("".join(filter(str.isdigit, count_text)))
                except (ValueError, TypeError):
                    pass

            # Extract genre
            genre_elem = container.find("span", class_="genre")
            genre = genre_elem.text.strip() if genre_elem else None

            # Create search result
            return SearchResult(
                dj_name=dj_name,
                event_name=event_name,
                date=result_date,
                venue=venue,
                set_type=set_type,
                url=url,
                duration=None,  # Added missing field
                track_count=track_count,
                genre=genre,
                description=None,  # Added missing field
                source_url=url,
            )

        except Exception as e:
            logger.warning(f"Failed to parse search result: {e}")
            return None

    def _parse_pagination(self, soup: BeautifulSoup, current_page: int, limit: int) -> PaginationInfo:
        """Parse pagination information from the page.

        Args:
            soup: Parsed HTML page
            current_page: Current page number
            limit: Items per page limit

        Returns:
            Pagination information
        """
        # Find pagination container (adjust selector based on actual HTML)
        pagination_elem = soup.find("div", class_="pagination")

        total_pages = 1
        total_items = 0

        if pagination_elem and isinstance(pagination_elem, Tag):
            # Try to find total pages
            last_page_elem = pagination_elem.find("a", class_="last")
            if last_page_elem and isinstance(last_page_elem, Tag):
                with contextlib.suppress(ValueError, TypeError):
                    # Extract page number from href or text
                    total_pages = int(last_page_elem.text.strip())

            # Try to find total items count
            count_elem = soup.find("span", class_="resultCount")
            if count_elem:
                try:
                    # Extract number from text like "Found 150 results"
                    count_text = count_elem.text.strip()
                    total_items = int("".join(filter(str.isdigit, count_text)))
                except (ValueError, TypeError):
                    pass

        # If we couldn't find total items, estimate from current page
        if total_items == 0:
            result_count = len(soup.find_all("div", class_="tlItem"))
            total_items = result_count

            # If we're not on the last page, estimate total
            if result_count == limit:
                total_items = total_pages * limit

        return PaginationInfo(
            page=current_page,
            limit=limit,
            total_pages=total_pages,
            total_items=total_items,
            has_next=current_page < total_pages,
            has_previous=current_page > 1,
        )

    def get_dj_tracklists(self, dj_slug: str, page: int = 1, limit: int = 20) -> SearchResponse:
        """Get tracklists for a specific DJ.

        Args:
            dj_slug: DJ identifier/slug
            page: Page number
            limit: Items per page

        Returns:
            Search response with DJ's tracklists
        """
        # Build URL for DJ's tracklist page
        url = urljoin(self.config.base_url, f"/dj/{dj_slug}/index")
        params = {"page": page}

        # Get the page
        soup = self.get_page(url, params)

        # Parse results
        results = self._parse_search_results(soup, SearchType.DJ)
        pagination = self._parse_pagination(soup, page, limit)

        return SearchResponse(
            results=results,
            pagination=pagination,
            query_info={
                "dj_slug": dj_slug,
                "type": "dj_tracklists",
            },
            cache_hit=False,
            response_time_ms=0.0,
            correlation_id=uuid4(),
        )

    def get_event_tracklists(self, event_slug: str, page: int = 1, limit: int = 20) -> SearchResponse:
        """Get tracklists for a specific event.

        Args:
            event_slug: Event identifier/slug
            page: Page number
            limit: Items per page

        Returns:
            Search response with event's tracklists
        """
        # Build URL for event's tracklist page
        url = urljoin(self.config.base_url, f"/event/{event_slug}/index")
        params = {"page": page}

        # Get the page
        soup = self.get_page(url, params)

        # Parse results
        results = self._parse_search_results(soup, SearchType.EVENT)
        pagination = self._parse_pagination(soup, page, limit)

        return SearchResponse(
            results=results,
            pagination=pagination,
            query_info={
                "event_slug": event_slug,
                "type": "event_tracklists",
            },
            cache_hit=False,
            response_time_ms=0.0,
            correlation_id=uuid4(),
        )
