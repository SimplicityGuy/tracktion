"""Tests for search API endpoints."""

from datetime import date
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from services.tracklist_service.src.main import create_app
from services.tracklist_service.src.models.search_models import PaginationInfo, SearchResponse, SearchResult, SearchType


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestSearchAPI:
    """Test search API endpoints."""

    @patch("services.tracklist_service.src.api.search.SearchScraper")
    def test_search_tracklists_success(self, mock_scraper_class, client):
        """Test successful search request."""
        # Mock scraper instance
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Mock search response
        mock_response = SearchResponse(
            results=[
                SearchResult(
                    dj_name="Test DJ",
                    url="https://1001tracklists.com/test",
                    source_url="https://1001tracklists.com/test",
                )
            ],
            pagination=PaginationInfo(
                page=1,
                limit=20,
                total_pages=1,
                total_items=1,
                has_next=False,
                has_previous=False,
            ),
            query_info={"query": "test", "type": "dj"},
            cache_hit=False,
            response_time_ms=100.0,
            correlation_id=uuid4(),
        )
        mock_scraper.search.return_value = mock_response

        # Make request
        response = client.get("/api/v1/search/?query=test")

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["dj_name"] == "Test DJ"
        assert data["pagination"]["total_items"] == 1

    def test_search_tracklists_missing_query(self, client):
        """Test search with missing query parameter."""
        response = client.get("/api/v1/search/")

        assert response.status_code == 422
        data = response.json()
        assert "query" in str(data["detail"])

    def test_search_tracklists_invalid_date_format(self, client):
        """Test search with invalid date format."""
        response = client.get("/api/v1/search/?query=test&start_date=invalid")

        assert response.status_code == 400
        assert "Invalid start_date format" in response.json()["detail"]

    def test_search_tracklists_with_all_parameters(self, client):
        """Test search with all optional parameters."""
        with patch("services.tracklist_service.src.api.search.SearchScraper") as mock_scraper_class:
            mock_scraper = MagicMock()
            mock_scraper_class.return_value = mock_scraper

            # Mock response
            mock_response = SearchResponse(
                results=[],
                pagination=PaginationInfo(
                    page=2,
                    limit=50,
                    total_pages=3,
                    total_items=150,
                    has_next=True,
                    has_previous=True,
                ),
                query_info={},
                cache_hit=False,
                response_time_ms=0.0,
                correlation_id=uuid4(),
            )
            mock_scraper.search.return_value = mock_response

            response = client.get(
                "/api/v1/search/?"
                "query=deadmau5&"
                "search_type=event&"
                "page=2&"
                "limit=50&"
                "start_date=2023-01-01&"
                "end_date=2023-12-31"
            )

            if response.status_code != 200:
                print(f"Response: {response.json()}")
            assert response.status_code == 200

            # Verify the scraper was called with correct parameters
            call_args = mock_scraper.search.call_args[0][0]
            assert call_args.query == "deadmau5"
            assert call_args.search_type == SearchType.EVENT
            assert call_args.page == 2
            assert call_args.limit == 50
            assert call_args.start_date == date(2023, 1, 1)
            assert call_args.end_date == date(2023, 12, 31)

    @patch("services.tracklist_service.src.api.search.SearchScraper")
    def test_search_tracklists_scraper_exception(self, mock_scraper_class, client):
        """Test search handles scraper exceptions."""
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper.search.side_effect = Exception("Scraping failed")

        response = client.get("/api/v1/search/?query=test")

        assert response.status_code == 500
        data = response.json()
        assert "SEARCH_FAILED" in str(data["detail"])

    @patch("services.tracklist_service.src.api.search.SearchScraper")
    def test_get_dj_tracklists_success(self, mock_scraper_class, client):
        """Test getting DJ tracklists."""
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Mock response
        mock_response = SearchResponse(
            results=[
                SearchResult(
                    dj_name="Deadmau5",
                    event_name="Ultra 2023",
                    url="https://1001tracklists.com/test",
                    source_url="https://1001tracklists.com/test",
                )
            ],
            pagination=PaginationInfo(
                page=1,
                limit=20,
                total_pages=1,
                total_items=1,
                has_next=False,
                has_previous=False,
            ),
            query_info={"dj_slug": "deadmau5"},
            cache_hit=False,
            response_time_ms=50.0,
            correlation_id=uuid4(),
        )
        mock_scraper.get_dj_tracklists.return_value = mock_response

        response = client.get("/api/v1/search/dj/deadmau5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["dj_name"] == "Deadmau5"
        assert data["results"][0]["event_name"] == "Ultra 2023"

    @patch("services.tracklist_service.src.api.search.SearchScraper")
    def test_get_dj_tracklists_with_pagination(self, mock_scraper_class, client):
        """Test getting DJ tracklists with pagination."""
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Mock empty response
        mock_response = SearchResponse(
            results=[],
            pagination=PaginationInfo(
                page=3,
                limit=50,
                total_pages=5,
                total_items=250,
                has_next=True,
                has_previous=True,
            ),
            query_info={},
            cache_hit=False,
            response_time_ms=0.0,
            correlation_id=uuid4(),
        )
        mock_scraper.get_dj_tracklists.return_value = mock_response

        response = client.get("/api/v1/search/dj/carl-cox?page=3&limit=50")

        assert response.status_code == 200

        # Verify scraper was called with correct parameters
        mock_scraper.get_dj_tracklists.assert_called_once_with("carl-cox", 3, 50)

    @patch("services.tracklist_service.src.api.search.SearchScraper")
    def test_get_event_tracklists_success(self, mock_scraper_class, client):
        """Test getting event tracklists."""
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Mock response
        mock_response = SearchResponse(
            results=[
                SearchResult(
                    dj_name="Carl Cox",
                    event_name="Tomorrowland 2023",
                    url="https://1001tracklists.com/test",
                    source_url="https://1001tracklists.com/test",
                )
            ],
            pagination=PaginationInfo(
                page=1,
                limit=20,
                total_pages=1,
                total_items=1,
                has_next=False,
                has_previous=False,
            ),
            query_info={"event_slug": "tomorrowland-2023"},
            cache_hit=False,
            response_time_ms=75.0,
            correlation_id=uuid4(),
        )
        mock_scraper.get_event_tracklists.return_value = mock_response

        response = client.get("/api/v1/search/event/tomorrowland-2023")

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["dj_name"] == "Carl Cox"
        assert data["results"][0]["event_name"] == "Tomorrowland 2023"

    @patch("services.tracklist_service.src.api.search.SearchScraper")
    def test_get_event_tracklists_exception(self, mock_scraper_class, client):
        """Test event tracklists handles exceptions."""
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper.get_event_tracklists.side_effect = Exception("Event not found")

        response = client.get("/api/v1/search/event/invalid-event")

        assert response.status_code == 500
        assert "Failed to retrieve tracklists for event" in response.json()["detail"]

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/search/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tracklist-service"
        assert "cache_enabled" in data
        assert data["scraping_enabled"] is True

    def test_search_with_invalid_page(self, client):
        """Test search with invalid page number."""
        response = client.get("/api/v1/search/?query=test&page=0")

        assert response.status_code == 422

    def test_search_with_invalid_limit(self, client):
        """Test search with invalid limit."""
        response = client.get("/api/v1/search/?query=test&limit=101")

        assert response.status_code == 422

    def test_search_with_empty_query(self, client):
        """Test search with empty query string."""
        response = client.get("/api/v1/search/?query=")

        assert response.status_code == 422
