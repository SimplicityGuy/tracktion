"""
Tests for search API endpoints.

Test suite for 1001tracklists search functionality including
pagination, caching, and error handling.
"""

from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from services.tracklist_service.src.api.search_api import router, SearchResult


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_search_results():
    """Create mock search results."""
    return [
        SearchResult(
            id="12345",
            url="https://1001tracklists.com/tracklist/12345/amazing-trance-set",
            title="Amazing Trance Set",
            dj_name="DJ Example",
            date="2024-01-15",
            event_name="Winter Festival 2024",
            track_count=25,
            duration="1:30:00",
            genre="Trance",
            confidence=0.95
        ),
        SearchResult(
            id="12346",
            url="https://1001tracklists.com/tracklist/12346/progressive-house-journey",
            title="Progressive House Journey",
            dj_name="DJ Example",
            date="2024-01-10",
            event_name="Club Night",
            track_count=20,
            duration="1:00:00",
            genre="Progressive House",
            confidence=0.88
        )
    ]


class TestSearchEndpoints:
    """Test search API endpoints."""
    
    @patch("services.tracklist_service.src.api.search_api._perform_search")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_1001tracklists_success(
        self, 
        mock_cache,
        mock_perform_search,
        client,
        mock_search_results
    ):
        """Test successful search with query parameter."""
        # Setup mocks
        mock_cache.get.return_value = None  # No cached results
        mock_cache.set = AsyncMock()
        mock_perform_search.return_value = mock_search_results
        
        # Make request
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={
                "query": "trance",
                "page": 1,
                "page_size": 20
            }
        )
        
        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert len(response_data["results"]) == 2
        assert response_data["total_count"] == 2
        assert response_data["page"] == 1
        assert response_data["page_size"] == 20
        assert response_data["has_more"] is False
        assert response_data["cached"] is False
        assert "processing_time_ms" in response_data
        assert "correlation_id" in response_data
        
        # Verify first result
        first_result = response_data["results"][0]
        assert first_result["title"] == "Amazing Trance Set"
        assert first_result["dj_name"] == "DJ Example"
        assert first_result["genre"] == "Trance"
        
        # Verify search was performed and cache was set
        mock_perform_search.assert_called_once()
        mock_cache.set.assert_called_once()
    
    @patch("services.tracklist_service.src.api.search_api._perform_search")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_with_artist_filter(
        self, 
        mock_cache,
        mock_perform_search,
        client,
        mock_search_results
    ):
        """Test search with artist parameter."""
        mock_cache.get.return_value = None
        mock_cache.set = AsyncMock()
        mock_perform_search.return_value = mock_search_results
        
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={
                "artist": "DJ Example",
                "page": 1,
                "page_size": 10
            }
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert response_data["page_size"] == 10
        
        # Verify search parameters were passed correctly
        search_call = mock_perform_search.call_args[0][0]
        assert search_call["artist"] == "DJ Example"
    
    @patch("services.tracklist_service.src.api.search_api._perform_search")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_with_pagination(
        self, 
        mock_cache,
        mock_perform_search,
        client,
        mock_search_results
    ):
        """Test search with pagination."""
        # Create larger result set
        large_results = mock_search_results * 10  # 20 results
        
        mock_cache.get.return_value = None
        mock_cache.set = AsyncMock()
        mock_perform_search.return_value = large_results
        
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={
                "query": "test",
                "page": 2,
                "page_size": 5
            }
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert response_data["total_count"] == 20
        assert response_data["page"] == 2
        assert response_data["page_size"] == 5
        assert response_data["has_more"] is True
        assert len(response_data["results"]) == 5
    
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_cached_results(self, mock_cache, client):
        """Test search returning cached results."""
        cached_response = {
            "success": True,
            "results": [
                {
                    "id": "12345",
                    "url": "https://1001tracklists.com/tracklist/12345/test",
                    "title": "Cached Set",
                    "dj_name": "Cached DJ",
                    "confidence": 1.0
                }
            ],
            "total_count": 1,
            "page": 1,
            "page_size": 20,
            "has_more": False,
            "cached": False
        }
        
        mock_cache.get.return_value = cached_response
        
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={"query": "test"}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert response_data["cached"] is True
        assert response_data["results"][0]["title"] == "Cached Set"
    
    def test_search_no_parameters_error(self, client):
        """Test search without any search parameters."""
        response = client.get("/api/v1/tracklists/search/1001tracklists")
        
        assert response.status_code == 400
        assert "At least one search parameter" in response.json()["detail"]
    
    def test_search_invalid_pagination(self, client):
        """Test search with invalid pagination parameters."""
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={
                "query": "test",
                "page": 0,  # Invalid page
                "page_size": 200  # Exceeds maximum
            }
        )
        
        # Should fail validation
        assert response.status_code == 422
    
    @patch("services.tracklist_service.src.api.search_api._perform_search")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_service_error(self, mock_cache, mock_perform_search, client):
        """Test search service error handling."""
        mock_cache.get.return_value = None
        mock_perform_search.side_effect = Exception("Search service unavailable")
        
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={"query": "test"}
        )
        
        assert response.status_code == 200  # Graceful error handling
        response_data = response.json()
        
        assert response_data["success"] is False
        assert "Search failed" in response_data["error"]
        assert response_data["results"] == []
        assert response_data["total_count"] == 0
    
    @patch("services.tracklist_service.src.api.search_api._perform_search")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_with_date_filters(
        self, 
        mock_cache,
        mock_perform_search,
        client,
        mock_search_results
    ):
        """Test search with date range filters."""
        mock_cache.get.return_value = None
        mock_cache.set = AsyncMock()
        mock_perform_search.return_value = mock_search_results
        
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={
                "query": "house",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
                "genre": "House"
            }
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        
        # Verify date filters were passed
        search_call = mock_perform_search.call_args[0][0]
        assert search_call["date_from"] == "2024-01-01"
        assert search_call["date_to"] == "2024-12-31"
        assert search_call["genre"] == "House"
    
    @patch("services.tracklist_service.src.api.search_api._perform_search")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_force_refresh(
        self, 
        mock_cache,
        mock_perform_search,
        client,
        mock_search_results
    ):
        """Test search with force refresh parameter."""
        # Setup cached data that should be ignored
        cached_data = {"results": [{"title": "Cached Result"}]}
        mock_cache.get.return_value = cached_data
        mock_cache.set = AsyncMock()
        mock_perform_search.return_value = mock_search_results
        
        response = client.get(
            "/api/v1/tracklists/search/1001tracklists",
            params={
                "query": "test",
                "force_refresh": True
            }
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should get fresh results, not cached
        assert response_data["success"] is True
        assert response_data["results"][0]["title"] == "Amazing Trance Set"
        
        # Cache should not have been checked due to force_refresh
        mock_perform_search.assert_called_once()
    
    @patch("services.tracklist_service.src.api.search_api.ImportService")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_health_check_healthy(
        self,
        mock_cache,
        mock_import_service_class,
        client
    ):
        """Test search health check with all services healthy."""
        mock_import_service_class.return_value = MagicMock()
        mock_cache.ping = AsyncMock()
        
        response = client.get("/api/v1/tracklists/search/health")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["service"] == "tracklist_search_api"
        assert response_data["status"] == "healthy"
        assert response_data["components"]["import_service"] == "healthy"
        assert response_data["components"]["cache"] == "healthy"
    
    @patch("services.tracklist_service.src.api.search_api.ImportService")
    @patch("services.tracklist_service.src.api.search_api.cache")
    def test_search_health_check_degraded(
        self,
        mock_cache,
        mock_import_service_class,
        client
    ):
        """Test search health check with service failures."""
        mock_import_service_class.side_effect = Exception("Import service down")
        mock_cache.ping.side_effect = Exception("Cache unavailable")
        
        response = client.get("/api/v1/tracklists/search/health")
        
        assert response.status_code == 503
        response_data = response.json()
        
        assert response_data["status"] == "degraded"
        assert "unhealthy" in response_data["components"]["import_service"]
        assert "unhealthy" in response_data["components"]["cache"]


class TestPerformSearch:
    """Test the _perform_search helper function."""
    
    @pytest.mark.asyncio
    async def test_perform_search_query_matching(self):
        """Test search query matching logic."""
        from services.tracklist_service.src.api.search_api import _perform_search
        
        # Test query matching
        search_params = {"query": "trance"}
        results = await _perform_search(search_params)
        
        # Should return results containing "trance"
        matching_results = [r for r in results if "trance" in r.title.lower() or "trance" in r.genre.lower()]
        assert len(matching_results) > 0
    
    @pytest.mark.asyncio
    async def test_perform_search_artist_filtering(self):
        """Test search artist filtering logic."""
        from services.tracklist_service.src.api.search_api import _perform_search
        
        # Test artist filtering
        search_params = {"artist": "DJ Example"}
        results = await _perform_search(search_params)
        
        # All results should match the artist
        for result in results:
            assert "dj example" in result.dj_name.lower()
    
    @pytest.mark.asyncio
    async def test_perform_search_genre_filtering(self):
        """Test search genre filtering logic."""
        from services.tracklist_service.src.api.search_api import _perform_search
        
        # Test genre filtering
        search_params = {"genre": "Progressive House"}
        results = await _perform_search(search_params)
        
        # Should return only Progressive House results
        for result in results:
            assert result.genre and "progressive house" in result.genre.lower()
    
    @pytest.mark.asyncio
    async def test_perform_search_no_matches(self):
        """Test search with no matching results."""
        from services.tracklist_service.src.api.search_api import _perform_search
        
        # Test search that should return no results
        search_params = {"query": "nonexistent genre"}
        results = await _perform_search(search_params)
        
        # Should return empty list
        assert len(results) == 0