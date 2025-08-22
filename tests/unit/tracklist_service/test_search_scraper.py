"""Tests for the search scraper."""

from datetime import date
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from services.tracklist_service.src.models.search_models import (
    SearchRequest,
    SearchResponse,
    SearchType,
)
from services.tracklist_service.src.scraper.search_scraper import SearchScraper


class TestSearchScraper:
    """Test SearchScraper functionality."""

    def test_build_search_url_dj_search(self):
        """Test URL building for DJ search."""
        scraper = SearchScraper()
        request = SearchRequest(
            query="Deadmau5",
            search_type=SearchType.DJ,
            page=1,
        )

        url = scraper._build_search_url(request)

        assert "https://1001tracklists.com/dj" in url
        assert "q=Deadmau5" in url
        assert "page=1" in url

    def test_build_search_url_with_dates(self):
        """Test URL building with date filters."""
        scraper = SearchScraper()
        request = SearchRequest(
            query="Ultra",
            search_type=SearchType.EVENT,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            page=2,
        )

        url = scraper._build_search_url(request)

        assert "https://1001tracklists.com/event" in url
        assert "q=Ultra" in url
        assert "page=2" in url
        assert "start_date=2023-01-01" in url
        assert "end_date=2023-12-31" in url

    def test_parse_single_result_valid(self):
        """Test parsing a valid search result."""
        scraper = SearchScraper()

        # Create mock HTML for a search result
        html = """
        <div class="tlItem">
            <a class="tlLink" href="/tracklist/123456">Tracklist</a>
            <span class="djName">Carl Cox</span>
            <span class="eventName">Space Ibiza</span>
            <span class="tlDate">2023-07-15</span>
            <span class="venue">Space</span>
            <span class="setType">DJ Set</span>
            <span class="trackCount">45 tracks</span>
            <span class="genre">Techno</span>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        container = soup.find("div", class_="tlItem")

        result = scraper._parse_single_result(container, SearchType.DJ)

        assert result is not None
        assert result.dj_name == "Carl Cox"
        assert result.event_name == "Space Ibiza"
        assert result.venue == "Space"
        assert result.set_type == "DJ Set"
        assert result.track_count == 45
        assert result.genre == "Techno"
        assert "tracklist/123456" in result.url

    def test_parse_single_result_missing_fields(self):
        """Test parsing result with missing optional fields."""
        scraper = SearchScraper()

        # Minimal HTML with only required fields
        html = """
        <div class="tlItem">
            <a class="tlLink" href="/tracklist/789">Tracklist</a>
            <span class="djName">Test DJ</span>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        container = soup.find("div", class_="tlItem")

        result = scraper._parse_single_result(container, SearchType.DJ)

        assert result is not None
        assert result.dj_name == "Test DJ"
        assert result.event_name is None
        assert result.venue is None
        assert result.track_count is None

    def test_parse_single_result_invalid(self):
        """Test parsing invalid result returns None."""
        scraper = SearchScraper()

        # HTML without required link
        html = """
        <div class="tlItem">
            <span class="djName">Test DJ</span>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        container = soup.find("div", class_="tlItem")

        result = scraper._parse_single_result(container, SearchType.DJ)

        assert result is None

    def test_parse_pagination_info(self):
        """Test parsing pagination information."""
        scraper = SearchScraper()

        html = """
        <div>
            <div class="pagination">
                <a class="last">5</a>
            </div>
            <span class="resultCount">Found 95 results</span>
            <div class="tlItem">Result 1</div>
            <div class="tlItem">Result 2</div>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")

        pagination = scraper._parse_pagination(soup, current_page=2, limit=20)

        assert pagination.page == 2
        assert pagination.limit == 20
        assert pagination.total_pages == 5
        assert pagination.total_items == 95
        assert pagination.has_next is True
        assert pagination.has_previous is True

    def test_parse_pagination_no_pagination_element(self):
        """Test parsing when no pagination element exists."""
        scraper = SearchScraper()

        html = """
        <div>
            <div class="tlItem">Result 1</div>
            <div class="tlItem">Result 2</div>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")

        pagination = scraper._parse_pagination(soup, current_page=1, limit=20)

        assert pagination.page == 1
        assert pagination.limit == 20
        assert pagination.total_pages == 1
        assert pagination.total_items == 2
        assert pagination.has_next is False
        assert pagination.has_previous is False

    @patch.object(SearchScraper, "get_page")
    def test_search_success(self, mock_get_page):
        """Test successful search execution."""
        scraper = SearchScraper()

        # Mock HTML response
        html = """
        <div>
            <div class="tlItem">
                <a class="tlLink" href="/tracklist/1">Tracklist 1</a>
                <span class="djName">DJ One</span>
            </div>
            <div class="tlItem">
                <a class="tlLink" href="/tracklist/2">Tracklist 2</a>
                <span class="djName">DJ Two</span>
            </div>
        </div>
        """
        mock_get_page.return_value = BeautifulSoup(html, "lxml")

        request = SearchRequest(
            query="Test",
            search_type=SearchType.DJ,
            page=1,
            limit=20,
        )

        response = scraper.search(request)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2
        assert response.results[0].dj_name == "DJ One"
        assert response.results[1].dj_name == "DJ Two"
        assert response.cache_hit is False
        assert response.correlation_id == request.correlation_id

    @patch.object(SearchScraper, "get_page")
    def test_search_with_exception(self, mock_get_page):
        """Test search handles exceptions properly."""
        scraper = SearchScraper()

        # Mock exception
        mock_get_page.side_effect = Exception("Network error")

        request = SearchRequest(
            query="Test",
            search_type=SearchType.DJ,
        )

        with pytest.raises(Exception) as exc_info:
            scraper.search(request)

        assert "Network error" in str(exc_info.value)

    @patch.object(SearchScraper, "get_page")
    def test_get_dj_tracklists(self, mock_get_page):
        """Test getting DJ-specific tracklists."""
        scraper = SearchScraper()

        html = """
        <div>
            <div class="tlItem">
                <a class="tlLink" href="/tracklist/1">Set 1</a>
                <span class="djName">Deadmau5</span>
                <span class="eventName">Ultra 2023</span>
            </div>
        </div>
        """
        mock_get_page.return_value = BeautifulSoup(html, "lxml")

        response = scraper.get_dj_tracklists("deadmau5", page=1, limit=20)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].dj_name == "Deadmau5"
        assert response.results[0].event_name == "Ultra 2023"
        assert response.query_info["dj_slug"] == "deadmau5"

    @patch.object(SearchScraper, "get_page")
    def test_get_event_tracklists(self, mock_get_page):
        """Test getting event-specific tracklists."""
        scraper = SearchScraper()

        html = """
        <div>
            <div class="tlItem">
                <a class="tlLink" href="/tracklist/1">Set 1</a>
                <span class="djName">Carl Cox</span>
                <span class="eventName">Tomorrowland 2023</span>
            </div>
        </div>
        """
        mock_get_page.return_value = BeautifulSoup(html, "lxml")

        response = scraper.get_event_tracklists("tomorrowland-2023", page=1, limit=20)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].dj_name == "Carl Cox"
        assert response.results[0].event_name == "Tomorrowland 2023"
        assert response.query_info["event_slug"] == "tomorrowland-2023"
