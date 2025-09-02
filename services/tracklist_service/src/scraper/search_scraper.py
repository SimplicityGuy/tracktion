"""
Search scraper for 1001tracklists.com.

Implements search functionality for DJs, events, and tracks.
"""

import contextlib
import logging
from datetime import UTC, datetime
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

        # Find result containers based on actual 1001tracklists.com structure
        # Each search result is in a generic container with cursor=pointer
        result_containers = soup.find_all("generic", attrs={"cursor": "pointer"})

        # Filter out advertisement containers
        result_containers = [
            container for container in result_containers if not self._is_advertisement_container(container)
        ]

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
            # Extract title and URL from link element
            title_elem = container.find("link")
            if not title_elem:
                logger.warning("No title link found in search result container")
                return None

            # Get the tracklist title and URL
            title = title_elem.get_text(strip=True) if hasattr(title_elem, "get_text") else "Unknown Title"
            href = title_elem.get("href", "") if isinstance(title_elem, Tag) else ""
            url = urljoin(self.config.base_url, href if isinstance(href, str) else "")

            # Extract metadata from the generic container
            metadata_container = container.find("generic", recursive=True)
            if not metadata_container:
                logger.warning("No metadata container found in search result")
                metadata_container = container

            # Extract DJ/creator name from creator section
            creator_elem = metadata_container.find("generic", string="creator")
            dj_name = "Unknown DJ"
            if creator_elem and creator_elem.parent:
                creator_link = creator_elem.parent.find("link")
                if creator_link:
                    dj_name = creator_link.get_text(strip=True)

            # Extract event name from title (for live sets)
            event_name = None
            if "@" in title:
                # For live sets like "Hardwell @ Mysteryland"
                parts = title.split("@")
                if len(parts) > 1:
                    event_name = parts[1].strip()
                    # Remove venue details after comma
                    if "," in event_name:
                        event_name = event_name.split(",")[0].strip()

            # Extract date from tracklist date section
            date_elem = metadata_container.find("generic", string="tracklist date")
            result_date = None
            if date_elem and date_elem.parent:
                date_text_elem = date_elem.parent.find("text")
                if date_text_elem:
                    date_text = date_text_elem.strip()
                    result_date = self._parse_date(date_text)

            # Extract venue from event name or title
            venue = None
            if event_name and "," in title:
                # Extract venue from title like "Hardwell @ Mysteryland, Netherlands"
                venue_parts = title.split(",")
                if len(venue_parts) > 1:
                    venue = venue_parts[-1].strip()

            # Extract set type (assume live set if event_name exists, otherwise radio show)
            set_type = "Live Set" if event_name else "Radio Show"

            # Extract track count from IDed tracks section
            track_count = None
            tracks_elem = metadata_container.find("generic", string="IDed tracks / total tracks")
            if tracks_elem and tracks_elem.parent:
                tracks_text_elem = tracks_elem.parent.find("text")
                if tracks_text_elem:
                    tracks_text = tracks_text_elem.strip()
                    # Extract from formats like "34/39" or "all/18"
                    if "/" in tracks_text:
                        try:
                            total_part = tracks_text.split("/")[1].strip()
                            track_count = int(total_part)
                        except (ValueError, IndexError):
                            pass

            # Extract genre from musicstyle(s) section
            genre = None
            genre_elem = metadata_container.find("generic", string="musicstyle(s)")
            if genre_elem and genre_elem.parent:
                genre_text_elem = genre_elem.parent.find("text")
                if genre_text_elem:
                    genre = genre_text_elem.strip()

            # Extract duration from play time section
            duration = None
            duration_elem = metadata_container.find("generic", string="play time")
            if duration_elem and duration_elem.parent:
                duration_text_elem = duration_elem.parent.find("text")
                if duration_text_elem:
                    duration_text = duration_text_elem.strip()
                    duration = self._parse_duration(duration_text)

            # Create search result
            return SearchResult(
                dj_name=dj_name,
                event_name=event_name,
                date=result_date,
                venue=venue,
                set_type=set_type,
                url=url,
                duration=duration,
                track_count=track_count,
                genre=genre,
                description=title,  # Use title as description
                source_url=url,
            )

        except Exception as e:
            logger.warning(f"Failed to parse search result: {e}")
            logger.debug(f"Container HTML: {container}")
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
            # Filter out ads from count
            actual_results = [
                container
                for container in soup.find_all("generic", attrs={"cursor": "pointer"})
                if not self._is_advertisement_container(container)
            ]
            total_items = len(actual_results)

            # If we're not on the last page, estimate total
            if total_items >= limit:
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

    def _is_advertisement_container(self, container: Tag) -> bool:
        """Check if a container is an advertisement.

        Args:
            container: HTML container element

        Returns:
            True if container is an advertisement
        """
        # Check for advertisement text or class indicators
        if container.find(text="ADVERTISEMENT"):
            return True

        # Check for common ad-related patterns
        ad_indicators = ["advertisement", "promo", "sponsor", "ad-"]
        container_text = container.get_text(separator=" ", strip=True).lower()

        return any(indicator in container_text for indicator in ad_indicators)

    def _parse_date(self, date_text: str) -> datetime | None:
        """Parse date string into datetime object.

        Args:
            date_text: Date string from the website

        Returns:
            Parsed datetime object or None if parsing fails
        """
        if not date_text:
            return None

        try:
            # Handle common date formats used on 1001tracklists
            # Format: 2025-08-23
            if len(date_text) == 10 and "-" in date_text:
                return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=UTC)

            # Format: 2025-08-23T00:00:00
            if "T" in date_text:
                return datetime.fromisoformat(date_text.replace("T", " ").split(".")[0])

            logger.warning(f"Unable to parse date format: {date_text}")
            return None

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse date '{date_text}': {e}")
            return None

    def _parse_duration(self, duration_text: str) -> int | None:
        """Parse duration string into total minutes.

        Args:
            duration_text: Duration string like "1h 23m" or "59m"

        Returns:
            Total duration in minutes or None if parsing fails
        """
        if not duration_text:
            return None

        try:
            total_minutes = 0

            # Handle formats like "1h 23m", "1h", "59m"
            if "h" in duration_text:
                parts = duration_text.split("h")
                hours = int(parts[0].strip())
                total_minutes += hours * 60

                # Check for minutes after hours
                if len(parts) > 1 and "m" in parts[1]:
                    minutes_part = parts[1].strip().replace("m", "").strip()
                    if minutes_part:
                        minutes = int(minutes_part)
                        total_minutes += minutes
            elif "m" in duration_text:
                # Only minutes
                minutes = int(duration_text.replace("m", "").strip())
                total_minutes = minutes

            return total_minutes if total_minutes > 0 else None

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse duration '{duration_text}': {e}")
            return None
