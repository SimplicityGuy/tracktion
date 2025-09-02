"""
Tracklist scraper for 1001tracklists.com.

Extends the base scraper to parse complete tracklist pages including
tracks, timestamps, transitions, and metadata.
"""

import hashlib
import logging
import re
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from services.tracklist_service.src.models.tracklist_models import (
    CuePoint,
    Track,
    Tracklist,
    TracklistMetadata,
    Transition,
    TransitionType,
)

from .base_scraper import ScraperBase

logger = logging.getLogger(__name__)


class TracklistScraper(ScraperBase):
    """Scraper for retrieving complete tracklist data."""

    def __init__(self) -> None:
        """Initialize the tracklist scraper."""
        super().__init__()
        self.base_url = "https://www.1001tracklists.com"

    def scrape_tracklist(self, url: str) -> Tracklist:
        """
        Scrape a complete tracklist from a URL.

        Args:
            url: The 1001tracklists.com URL to scrape

        Returns:
            Tracklist object with all parsed data

        Raises:
            ValueError: If URL is invalid or page cannot be parsed
            requests.RequestException: If network error occurs
        """
        # Validate URL
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid 1001tracklists.com URL: {url}")

        # Fetch the page
        response = self._make_request(url)
        html = response.text
        soup = self.parse_html(html)

        # Calculate HTML hash for change detection
        html_hash = hashlib.sha256(html.encode()).hexdigest()

        # Extract basic information
        dj_name = self._extract_dj_name(soup)
        event_info = self._extract_event_info(soup)

        # Extract tracks
        tracks = self._extract_tracks(soup)

        # Extract transitions if available
        transitions = self._extract_transitions(soup, tracks)

        # Extract metadata
        metadata = self._extract_metadata(soup)

        # Create and return tracklist
        return Tracklist(
            url=url,
            dj_name=dj_name,
            event_name=event_info.get("event_name"),
            venue=event_info.get("venue"),
            date=event_info.get("date"),
            tracks=tracks,
            transitions=transitions,
            metadata=metadata,
            scraped_at=datetime.now(UTC),
            source_html_hash=html_hash,
        )

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is a valid 1001tracklists.com URL."""
        parsed = urlparse(url)
        valid_hosts = ["1001tracklists.com", "www.1001tracklists.com"]
        return parsed.hostname in valid_hosts and parsed.scheme in ["http", "https"]

    def _extract_dj_name(self, soup: BeautifulSoup) -> str:
        """Extract DJ/artist name from the page."""
        # Try multiple selectors for DJ name
        selectors = [
            "div.tlHead h1.marL10",
            "h1.djName",
            "div.djHeader h1",
            "meta[property='og:title']",
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == "meta":
                    content = element.get("content", "")
                    if isinstance(content, str):
                        # Parse from meta content like "DJ Name @ Event"
                        if "@" in content:
                            return content.split("@")[0].strip()
                        return content.strip()
                    return "Unknown DJ"
                text = element.get_text(strip=True)
                if text:
                    return str(text)

        # Fallback
        return "Unknown DJ"

    def _extract_event_info(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract event information from the page."""
        info = {}

        # Event name
        event_elem = soup.select_one("div.tlHead h2, h2.eventName, div.eventInfo h2")
        if event_elem:
            info["event_name"] = event_elem.get_text(strip=True)

        # Venue
        venue_elem = soup.select_one("span.venue, div.venueInfo, a.venue")
        if venue_elem:
            info["venue"] = venue_elem.get_text(strip=True)

        # Date
        date_elem = soup.select_one("div.when time, time.eventDate, span.dateInfo")
        if date_elem:
            datetime_attr = date_elem.get("datetime")
            if isinstance(datetime_attr, str):
                parsed_date = self._parse_date(datetime_attr)
                info["date"] = parsed_date.isoformat() if parsed_date else ""
            else:
                date_text = date_elem.get_text(strip=True)
                parsed_date = self._parse_date(date_text)
                info["date"] = parsed_date.isoformat() if parsed_date else ""

        return info

    def _parse_date(self, date_str: str) -> date | None:
        """Parse date string to date object."""
        if not date_str:
            return None

        # Try multiple date formats
        formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%d %B %Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=UTC).date()
            except ValueError:
                continue

        return None

    def _extract_tracks(self, soup: BeautifulSoup) -> list[Track]:
        """Extract all tracks from the tracklist."""
        tracks = []

        # Find track container (usually a table or list)
        track_containers = soup.select("div.tlpTog, table.tracklist tr, div.trackItem, li.track")

        for idx, container in enumerate(track_containers, 1):
            track = self._parse_track(container, idx)
            if track:
                tracks.append(track)

        return tracks

    def _parse_track(self, element: Tag, track_number: int) -> Track | None:
        """Parse individual track from HTML element."""
        try:
            # Extract timestamp/cue point
            timestamp = self._extract_timestamp(element, track_number)

            # Extract artist and title
            artist, title = self._extract_artist_title(element)

            # Check if it's an ID track
            is_id = self._is_id_track(artist, title)

            # Extract remix info
            remix = self._extract_remix(title)
            if remix:
                # Remove remix from title if found
                title = title.replace(f"({remix})", "").replace(f"[{remix}]", "").strip()

            # Extract label
            label = self._extract_label(element)

            # Extract additional metadata
            bpm = self._extract_bpm(element)
            key = self._extract_key(element)
            genre = self._extract_genre(element)

            return Track(
                number=track_number,
                timestamp=timestamp,
                artist=artist,
                title=title,
                remix=remix,
                label=label,
                is_id=is_id,
                bpm=bpm,
                key=key,
                genre=genre,
                notes=None,  # Added missing field
            )
        except (ValueError, AttributeError, KeyError) as e:
            # Log parsing error for debugging but continue processing
            # Specific exceptions: ValueError for invalid data, AttributeError for missing methods,
            # KeyError for missing dictionary keys
            logger.debug(f"Failed to parse track {track_number}: {e}")
            return None

    def _extract_timestamp(self, element: Tag, track_number: int) -> CuePoint | None:
        """Extract timestamp/cue point from track element."""
        # Look for time elements
        time_selectors = [
            "span.cue",
            "div.time",
            "td.time",
            "span.trackTime",
            "div.cueTime",
        ]

        for selector in time_selectors:
            time_elem = element.select_one(selector)
            if time_elem:
                time_str = time_elem.get_text(strip=True)
                timestamp_ms = self._parse_time_to_ms(time_str)
                if timestamp_ms is not None:
                    return CuePoint(
                        track_number=track_number,
                        timestamp_ms=timestamp_ms,
                        formatted_time=time_str,
                    )

        return None

    def _parse_time_to_ms(self, time_str: str) -> int | None:
        """Parse time string to milliseconds."""
        if not time_str:
            return None

        # Remove any non-numeric characters except :
        time_str = re.sub(r"[^\d:]", "", time_str)

        parts = time_str.split(":")
        if len(parts) == 2:  # MM:SS
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return (minutes * 60 + seconds) * 1000
            except ValueError:
                return None
        elif len(parts) == 3:  # HH:MM:SS
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return (hours * 3600 + minutes * 60 + seconds) * 1000
            except ValueError:
                return None

        return None

    def _extract_artist_title(self, element: Tag) -> tuple[str, str]:
        """Extract artist and title from track element."""
        # Try to find artist and title in separate elements
        artist_elem = element.select_one("span.artist, div.artistName, a.artist, td.artist")
        title_elem = element.select_one("span.track, div.trackName, a.track, td.title")

        if artist_elem and title_elem:
            return (artist_elem.get_text(strip=True), title_elem.get_text(strip=True))

        # Try to parse from combined format "Artist - Title"
        combined = element.select_one("div.track, span.trackInfo, td.track")
        if combined:
            text = combined.get_text(strip=True)
            if " - " in text:
                parts = text.split(" - ", 1)
                return parts[0].strip(), parts[1].strip()

        # Fallback
        return "Unknown Artist", "Unknown Title"

    def _is_id_track(self, artist: str, title: str) -> bool:
        """Check if track is an ID/unknown track."""
        id_patterns = ["ID", "Unknown", "???", "Unreleased", "Untitled"]

        artist_lower = artist.lower()
        title_lower = title.lower()

        for pattern in id_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in artist_lower or pattern_lower in title_lower:
                return True

        return False

    def _extract_remix(self, title: str) -> str | None:
        """Extract remix information from track title."""
        # Look for remix patterns in parentheses or brackets
        patterns = [
            r"\(([^)]*(?:Remix|Mix|Edit|Bootleg|Rework|VIP)[^)]*)\)",
            r"\[([^]]*(?:Remix|Mix|Edit|Bootleg|Rework|VIP)[^]]*)\]",
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_label(self, element: Tag) -> str | None:
        """Extract record label from track element."""
        label_elem = element.select_one("span.label, div.labelName, a.label, td.label")

        if label_elem:
            return str(label_elem.get_text(strip=True))

        return None

    def _extract_bpm(self, element: Tag) -> float | None:
        """Extract BPM from track element."""
        bpm_elem = element.select_one("span.bpm, div.bpm, td.bpm")

        if bpm_elem:
            bpm_text = bpm_elem.get_text(strip=True)
            # Extract numeric value
            match = re.search(r"(\d+(?:\.\d+)?)", bpm_text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return None

    def _extract_key(self, element: Tag) -> str | None:
        """Extract musical key from track element."""
        key_elem = element.select_one("span.key, div.key, td.key")

        if key_elem:
            return str(key_elem.get_text(strip=True))

        return None

    def _extract_genre(self, element: Tag) -> str | None:
        """Extract genre from track element."""
        genre_elem = element.select_one("span.genre, div.genre, td.genre")

        if genre_elem:
            return str(genre_elem.get_text(strip=True))

        return None

    def _extract_transitions(self, soup: BeautifulSoup, tracks: list[Track]) -> list[Transition]:
        """Extract transition information between tracks."""
        transitions = []

        # Look for transition markers
        transition_elems = soup.select("div.transition, span.mixInfo, div.transitionType")

        for elem in transition_elems:
            transition = self._parse_transition(elem, tracks)
            if transition:
                transitions.append(transition)

        # If no explicit transitions, infer from track timestamps
        if not transitions and len(tracks) > 1:
            transitions = self._infer_transitions(tracks)

        return transitions

    def _parse_transition(self, element: Tag, tracks: list[Track]) -> Transition | None:
        """Parse transition from HTML element."""
        # This would need to be adapted based on actual HTML structure
        # For now, return None as we'd need real examples
        return None

    def _infer_transitions(self, tracks: list[Track]) -> list[Transition]:
        """Infer transitions from track timestamps."""
        transitions = []

        for i in range(len(tracks) - 1):
            current_timestamp = tracks[i].timestamp
            next_timestamp = tracks[i + 1].timestamp

            if current_timestamp is not None and next_timestamp is not None:
                # Calculate overlap or gap
                current_end = current_timestamp.timestamp_ms + 180000  # Assume 3 min avg
                next_start = next_timestamp.timestamp_ms

                # Determine transition type based on timing
                trans_type = TransitionType.BLEND if abs(current_end - next_start) < 5000 else TransitionType.CUT

                transitions.append(
                    Transition(
                        from_track=tracks[i].number,
                        to_track=tracks[i + 1].number,
                        transition_type=trans_type,
                        timestamp_ms=next_start,
                        duration_ms=None,  # Added missing field
                        notes=None,  # Added missing field
                    )
                )

        return transitions

    def _extract_metadata(self, soup: BeautifulSoup) -> TracklistMetadata | None:
        """Extract additional metadata from the page."""
        # Initialize with all required fields
        metadata = TracklistMetadata(
            recording_type=None,
            duration_minutes=None,
            play_count=None,
            favorite_count=None,
            comment_count=None,
            download_url=None,
            stream_url=None,
            soundcloud_url=None,
            mixcloud_url=None,
            youtube_url=None,
        )

        # Recording type
        type_elem = soup.select_one("span.setType, div.recordingType")
        if type_elem:
            metadata.recording_type = type_elem.get_text(strip=True)

        # Duration
        duration_elem = soup.select_one("span.duration, div.setLength")
        if duration_elem:
            duration_text = duration_elem.get_text(strip=True)
            # Parse duration to minutes
            match = re.search(r"(\d+)", duration_text)
            if match:
                metadata.duration_minutes = int(match.group(1))

        # Play count
        plays_elem = soup.select_one("span.plays, div.playCount")
        if plays_elem:
            plays_text = plays_elem.get_text(strip=True)
            match = re.search(r"(\d+)", plays_text.replace(",", ""))
            if match:
                metadata.play_count = int(match.group(1))

        # Favorite count
        fav_elem = soup.select_one("span.favorites, div.favCount")
        if fav_elem:
            fav_text = fav_elem.get_text(strip=True)
            match = re.search(r"(\d+)", fav_text.replace(",", ""))
            if match:
                metadata.favorite_count = int(match.group(1))

        # External links
        for link in soup.select("a.external, a.streamLink"):
            href = link.get("href", "")
            if isinstance(href, str):
                href_lower = href.lower()
                if "soundcloud.com" in href_lower:
                    metadata.soundcloud_url = href
                elif "mixcloud.com" in href_lower:
                    metadata.mixcloud_url = href
                elif "youtube.com" in href_lower or "youtu.be" in href_lower:
                    metadata.youtube_url = href

        # Tags/genres
        tag_elems = soup.select("a.tag, span.genre, div.styleTag")
        metadata.tags = [elem.get_text(strip=True) for elem in tag_elems]

        return (
            metadata
            if any(
                [
                    metadata.recording_type,
                    metadata.duration_minutes,
                    metadata.play_count,
                    metadata.tags,
                ]
            )
            else None
        )
