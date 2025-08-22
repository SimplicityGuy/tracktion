#!/usr/bin/env python3
"""
Example client for Tracklist Service API.

Demonstrates how to use the tracklist service to search for and retrieve tracklists.
"""

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp
import requests  # type: ignore[import-untyped]


class TracklistClient:
    """Client for interacting with the Tracklist Service API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the tracklist service
        """
        self.base_url = base_url
        self.api_prefix = "/api/v1"

    def search(
        self,
        search_type: str,
        query: str,
        limit: int = 20,
        page: int = 1,
    ) -> Dict[str, Any]:
        """Search for DJs, events, or tracklists.

        Args:
            search_type: Type of search (dj, event, tracklist)
            query: Search query
            limit: Number of results to return
            page: Page number for pagination

        Returns:
            Search results
        """
        url = f"{self.base_url}{self.api_prefix}/search"
        payload = {
            "search_type": search_type,
            "query": query,
            "limit": limit,
            "page": page,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def get_tracklist(
        self,
        url: str,
        force_refresh: bool = False,
        include_transitions: bool = True,
        async_processing: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve a tracklist from a URL.

        Args:
            url: 1001tracklists.com URL
            force_refresh: Force re-scraping even if cached
            include_transitions: Include transition information
            async_processing: Process asynchronously

        Returns:
            Tracklist data or job information
        """
        endpoint = f"{self.base_url}{self.api_prefix}/tracklist"
        if async_processing:
            endpoint += "?async_processing=true"

        payload = {
            "url": url,
            "force_refresh": force_refresh,
            "include_transitions": include_transitions,
        }

        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def get_job_status(self, correlation_id: str) -> Dict[str, Any]:
        """Get the status of an async job.

        Args:
            correlation_id: Job correlation ID

        Returns:
            Job status and result if completed
        """
        url = f"{self.base_url}{self.api_prefix}/tracklist/status/{correlation_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def wait_for_job(
        self,
        correlation_id: str,
        timeout: int = 60,
        poll_interval: int = 2,
    ) -> Dict[str, Any]:
        """Wait for an async job to complete.

        Args:
            correlation_id: Job correlation ID
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds

        Returns:
            Completed job result

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(correlation_id)

            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {correlation_id} did not complete within {timeout} seconds")

    def clear_cache(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Clear tracklist cache.

        Args:
            url: Specific URL to clear, or None to clear all

        Returns:
            Cache clearing result
        """
        endpoint = f"{self.base_url}{self.api_prefix}/tracklist/cache"
        if url:
            endpoint += f"?url={url}"

        response = requests.delete(endpoint)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health status
        """
        url = f"{self.base_url}{self.api_prefix}/health"
        response = requests.get(url)
        return response.json()  # type: ignore[no-any-return]


class AsyncTracklistClient:
    """Async client for the Tracklist Service API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the async client."""
        self.base_url = base_url
        self.api_prefix = "/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncTracklistClient":
        """Enter async context."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        if self.session:
            await self.session.close()

    async def search(
        self,
        search_type: str,
        query: str,
        limit: int = 20,
        page: int = 1,
    ) -> Dict[str, Any]:
        """Async search for DJs, events, or tracklists."""
        url = f"{self.base_url}{self.api_prefix}/search"
        payload = {
            "search_type": search_type,
            "query": query,
            "limit": limit,
            "page": page,
        }

        async with self.session.post(url, json=payload) as response:  # type: ignore[union-attr]
            response.raise_for_status()
            return await response.json()  # type: ignore[no-any-return]

    async def get_tracklist(
        self,
        url: str,
        force_refresh: bool = False,
        include_transitions: bool = True,
    ) -> Dict[str, Any]:
        """Async retrieve a tracklist."""
        endpoint = f"{self.base_url}{self.api_prefix}/tracklist"
        payload = {
            "url": url,
            "force_refresh": force_refresh,
            "include_transitions": include_transitions,
        }

        async with self.session.post(endpoint, json=payload) as response:  # type: ignore[union-attr]
            response.raise_for_status()
            return await response.json()  # type: ignore[no-any-return]


def main() -> None:
    """Example usage of the Tracklist Client."""

    # Create client
    client = TracklistClient()

    # Check health
    print("Checking service health...")
    health = client.health_check()
    print(f"Service status: {health['status']}")
    print(f"Components: {health['components']}")
    print()

    # Search for a DJ
    print("Searching for Amelie Lens...")
    search_results = client.search("dj", "Amelie Lens", limit=5)

    if search_results["success"] and search_results["results"]:
        print(f"Found {len(search_results['results'])} results:")
        for result in search_results["results"]:
            print(f"  - {result['name']} ({result['result_type']})")
            if result["result_type"] == "tracklist":
                print(f"    URL: {result['url']}")
                print(f"    Tracks: {result['metadata'].get('track_count', 'N/A')}")
        print()

        # Get the first tracklist
        tracklist_results = [r for r in search_results["results"] if r["result_type"] == "tracklist"]

        if tracklist_results:
            first_tracklist = tracklist_results[0]
            print(f"Retrieving tracklist: {first_tracklist['name']}")

            # Retrieve tracklist (sync)
            tracklist_data = client.get_tracklist(first_tracklist["url"])

            if tracklist_data["success"]:
                tracklist = tracklist_data["tracklist"]
                print(f"DJ: {tracklist['dj_name']}")
                print(f"Event: {tracklist.get('event_name', 'N/A')}")
                print(f"Date: {tracklist.get('date', 'N/A')}")
                print(f"Tracks: {len(tracklist['tracks'])}")
                print()

                # Print first 5 tracks
                print("First 5 tracks:")
                for track in tracklist["tracks"][:5]:
                    timestamp = track["timestamp"]["formatted_time"] if track.get("timestamp") else "??:??"
                    artist = track["artist"]
                    title = track["title"]
                    remix = f" ({track['remix']})" if track.get("remix") else ""
                    print(f"  {timestamp} - {artist} - {title}{remix}")
                print()

                # Print transitions if available
                if tracklist.get("transitions"):
                    print(f"Transitions: {len(tracklist['transitions'])}")
                    for trans in tracklist["transitions"][:3]:
                        print(f"  Track {trans['from_track']} → {trans['to_track']}: {trans['transition_type']}")

    # Example async processing
    print("\nExample of async processing:")
    print("Submitting tracklist for async retrieval...")

    test_url = "https://www.1001tracklists.com/tracklist/2npkg8l1/carl-cox-the-revolution-2024.html"
    async_result = client.get_tracklist(test_url, async_processing=True)

    if async_result["success"]:
        correlation_id = async_result["correlation_id"]
        print(f"Job submitted with ID: {correlation_id}")
        print("Waiting for completion...")

        try:
            completed = client.wait_for_job(correlation_id, timeout=30)
            print(f"Job completed! Status: {completed['status']}")
            if "tracklist" in completed:
                print(f"Retrieved {len(completed['tracklist']['tracks'])} tracks")
        except TimeoutError as e:
            print(f"Job timed out: {e}")
        except RuntimeError as e:
            print(f"Job failed: {e}")


async def async_main() -> None:
    """Example async usage."""

    async with AsyncTracklistClient() as client:
        # Search asynchronously
        results = await client.search("event", "Tomorrowland", limit=3)
        print(f"Found {len(results['results'])} events")

        # Get multiple tracklists concurrently
        urls = [
            "https://www.1001tracklists.com/tracklist/example1",
            "https://www.1001tracklists.com/tracklist/example2",
        ]

        tasks = [client.get_tracklist(url) for url in urls]
        tracklists = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(tracklists):
            if isinstance(result, Exception):
                print(f"Failed to get tracklist {i + 1}: {result}")
            else:
                print(f"Tracklist {i + 1}: {result.get('tracklist', {}).get('dj_name', 'Unknown')}")  # type: ignore[union-attr]


if __name__ == "__main__":
    # Run sync example
    main()

    # Uncomment to run async example
    # asyncio.run(async_main())
