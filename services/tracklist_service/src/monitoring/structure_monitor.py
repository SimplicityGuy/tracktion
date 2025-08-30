"""HTML structure monitoring for detecting site changes."""

import hashlib
import json
from datetime import datetime, UTC
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from bs4 import BeautifulSoup, Tag
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of structural changes detected."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    REORDERED = "reordered"


@dataclass
class StructuralChange:
    """Represents a single structural change."""

    element_path: str
    change_type: ChangeType
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    impact_score: float = 0.0


@dataclass
class ChangeReport:
    """Report of structural changes between two HTML versions."""

    page_type: str
    timestamp: datetime
    changes: List[StructuralChange] = field(default_factory=list)
    severity: str = "low"
    fingerprint_match_percentage: float = 100.0
    requires_manual_review: bool = False

    @property
    def has_breaking_changes(self) -> bool:
        """Check if any changes are breaking."""
        return any(c.impact_score > 0.7 for c in self.changes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for storage/transmission."""
        return {
            "page_type": self.page_type,
            "timestamp": self.timestamp.isoformat(),
            "changes": [
                {
                    "element_path": c.element_path,
                    "change_type": c.change_type.value,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "impact_score": c.impact_score,
                }
                for c in self.changes
            ],
            "severity": self.severity,
            "fingerprint_match_percentage": self.fingerprint_match_percentage,
            "requires_manual_review": self.requires_manual_review,
            "has_breaking_changes": self.has_breaking_changes,
        }


class StructureMonitor:
    """Monitor HTML structure changes for 1001tracklists pages."""

    def __init__(self, redis_client: Optional[redis.Redis[bytes]] = None, config_path: Optional[Path] = None):
        """Initialize structure monitor.

        Args:
            redis_client: Redis client for storing baselines
            config_path: Path to selector configuration file
        """
        self.redis_client = redis_client
        self.config_path = config_path or Path("config/selectors.json")
        self._selectors_config = self._load_selector_config()
        self._baseline_cache: Dict[str, Dict[str, Any]] = {}

    def _load_selector_config(self) -> Dict[str, Any]:
        """Load selector configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config: Dict[str, Any] = json.load(f)
                return config
        return self._get_default_selectors()

    def _get_default_selectors(self) -> Dict[str, Any]:
        """Get default selector configuration for 1001tracklists."""
        return {
            "search": {
                "critical": [".search-results", ".result-item", "a.result-link"],
                "important": [".result-date", ".result-venue", ".result-type"],
                "optional": [".result-image", ".result-tags"],
            },
            "tracklist": {
                "critical": [".tracklist-container", ".track-item", ".track-time", ".track-title"],
                "important": [".track-artist", ".track-label", ".mix-info"],
                "optional": [".track-id", ".track-notes", ".social-stats"],
            },
            "dj": {
                "critical": [".dj-profile", ".sets-list", ".set-item"],
                "important": [".dj-name", ".dj-bio", ".set-date"],
                "optional": [".dj-image", ".social-links"],
            },
        }

    def capture_structure_fingerprint(self, html: str, page_type: str) -> Dict[str, Any]:
        """Capture structural fingerprint of HTML page.

        Args:
            html: HTML content to analyze
            page_type: Type of page (search, tracklist, dj)

        Returns:
            Dictionary containing structural fingerprint
        """
        soup = BeautifulSoup(html, "html.parser")
        fingerprint = {
            "page_type": page_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "checksum": hashlib.sha256(html.encode()).hexdigest(),
            "structure": {},
            "selectors": {},
        }

        # Capture overall structure
        fingerprint["structure"] = self._analyze_structure(soup)

        # Capture selector-specific information
        if page_type in self._selectors_config:
            fingerprint["selectors"] = self._analyze_selectors(soup, self._selectors_config[page_type])

        # Version tracking
        version = self._detect_version(soup)
        if version:
            fingerprint["version"] = version

        return fingerprint

    def _analyze_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze overall HTML structure."""
        structure: Dict[str, Any] = {
            "tag_counts": {},
            "class_names": set(),
            "id_names": set(),
            "data_attributes": set(),
            "depth_profile": [],
            "text_density": 0,
        }

        # Count tags
        for tag in soup.find_all():
            if not hasattr(tag, "name"):
                continue
            tag_name = tag.name
            structure["tag_counts"][tag_name] = structure["tag_counts"].get(tag_name, 0) + 1

            # Collect classes
            if hasattr(tag, "get") and tag.get("class"):
                structure["class_names"].update(tag.get("class"))

            # Collect IDs
            if hasattr(tag, "get") and tag.get("id"):
                structure["id_names"].add(tag.get("id"))

            # Collect data attributes
            if hasattr(tag, "attrs"):
                for attr in tag.attrs:
                    if attr.startswith("data-"):
                        structure["data_attributes"].add(attr)

        # Convert sets to lists for JSON serialization
        structure["class_names"] = list(structure["class_names"])
        structure["id_names"] = list(structure["id_names"])
        structure["data_attributes"] = list(structure["data_attributes"])

        # Calculate text density
        text_length = len(soup.get_text(strip=True))
        html_length = len(str(soup))
        structure["text_density"] = text_length / html_length if html_length > 0 else 0

        return structure

    def _analyze_selectors(self, soup: BeautifulSoup, selectors: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific selectors for a page type."""
        results: Dict[str, Dict[str, Any]] = {}

        for priority, selector_list in selectors.items():
            results[priority] = {}
            for selector in selector_list:
                elements = soup.select(selector)
                results[priority][selector] = {
                    "count": len(elements),
                    "exists": len(elements) > 0,
                    "first_text": elements[0].get_text(strip=True)[:100] if elements else None,
                    "attributes": self._get_element_attributes(elements[0]) if elements else {},
                }

        return results

    def _get_element_attributes(self, element: Tag) -> Dict[str, Any]:
        """Get relevant attributes from an element."""
        if not element:
            return {}

        classes_raw = element.get("class")
        classes: List[str]
        if classes_raw is None:
            classes = []
        elif isinstance(classes_raw, str):
            classes = [classes_raw]
        elif isinstance(classes_raw, list):
            classes = classes_raw
        else:
            classes = []

        return {
            "tag": element.name,
            "classes": classes,
            "id": element.get("id"),
            "data_attrs": {k: v for k, v in element.attrs.items() if k.startswith("data-")},
        }

    def _detect_version(self, soup: BeautifulSoup) -> Optional[str]:
        """Try to detect site version from meta tags or comments."""
        # Check meta tags
        version_meta = soup.find("meta", attrs={"name": "version"})
        if version_meta and hasattr(version_meta, "get"):
            content = version_meta.get("content")
            return str(content) if content else None

        # Check for version in comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and "version" in text.lower()):
            if "version" in str(comment).lower():
                import re

                match = re.search(r"version[:\s]+([0-9.]+)", str(comment), re.IGNORECASE)
                if match:
                    return match.group(1)

        return None

    def compare_structures(self, current: dict, baseline: Dict[str, Any]) -> ChangeReport:
        """Compare current structure with baseline.

        Args:
            current: Current structure fingerprint
            baseline: Baseline structure fingerprint

        Returns:
            ChangeReport detailing the differences
        """
        report = ChangeReport(page_type=current.get("page_type", "unknown"), timestamp=datetime.now(UTC))

        # Quick checksum comparison
        if current.get("checksum") == baseline.get("checksum"):
            report.fingerprint_match_percentage = 100.0
            return report

        # Compare structures
        changes = []

        # Compare tag counts
        current_tags = current.get("structure", {}).get("tag_counts", {})
        baseline_tags = baseline.get("structure", {}).get("tag_counts", {})

        for tag in set(current_tags.keys()) | set(baseline_tags.keys()):
            current_count = current_tags.get(tag, 0)
            baseline_count = baseline_tags.get(tag, 0)

            if current_count != baseline_count:
                change_type = (
                    ChangeType.ADDED
                    if baseline_count == 0
                    else (ChangeType.REMOVED if current_count == 0 else ChangeType.MODIFIED)
                )
                changes.append(
                    StructuralChange(
                        element_path=f"tag:{tag}",
                        change_type=change_type,
                        old_value=str(baseline_count),
                        new_value=str(current_count),
                        impact_score=0.3 if tag in ["div", "span"] else 0.5,
                    )
                )

        # Compare critical selectors
        current_selectors = current.get("selectors", {})
        baseline_selectors = baseline.get("selectors", {})

        for priority in ["critical", "important", "optional"]:
            impact_base = {"critical": 0.8, "important": 0.5, "optional": 0.2}[priority]

            current_pri = current_selectors.get(priority, {})
            baseline_pri = baseline_selectors.get(priority, {})

            for selector in set(current_pri.keys()) | set(baseline_pri.keys()):
                current_info = current_pri.get(selector, {})
                baseline_info = baseline_pri.get(selector, {})

                if current_info.get("exists") != baseline_info.get("exists"):
                    changes.append(
                        StructuralChange(
                            element_path=f"selector:{selector}",
                            change_type=ChangeType.ADDED
                            if baseline_info.get("exists") is False
                            else ChangeType.REMOVED,
                            old_value=str(baseline_info.get("exists", False)),
                            new_value=str(current_info.get("exists", False)),
                            impact_score=impact_base,
                        )
                    )
                elif current_info.get("count", 0) != baseline_info.get("count", 0):
                    changes.append(
                        StructuralChange(
                            element_path=f"selector:{selector}",
                            change_type=ChangeType.MODIFIED,
                            old_value=f"count:{baseline_info.get('count', 0)}",
                            new_value=f"count:{current_info.get('count', 0)}",
                            impact_score=impact_base * 0.7,
                        )
                    )

        report.changes = changes

        # Calculate severity
        if any(c.impact_score >= 0.8 for c in changes):
            report.severity = "critical"
            report.requires_manual_review = True
        elif any(c.impact_score >= 0.5 for c in changes):
            report.severity = "high"
        elif any(c.impact_score >= 0.3 for c in changes):
            report.severity = "medium"
        else:
            report.severity = "low"

        # Calculate match percentage
        total_elements = len(current_tags) + len(current_selectors.get("critical", {}))
        changed_elements = len([c for c in changes if c.impact_score > 0.3])
        report.fingerprint_match_percentage = max(0, 100 * (1 - changed_elements / max(total_elements, 1)))

        return report

    async def store_baseline(self, page_type: str, structure: Dict[str, Any]) -> None:
        """Store baseline structure in Redis.

        Args:
            page_type: Type of page
            structure: Structure fingerprint to store
        """
        if self.redis_client:
            key = f"structure:baseline:{page_type}"
            await self.redis_client.set(
                key,
                json.dumps(structure),
                ex=86400 * 30,  # 30 days expiry
            )

            # Also store in version history
            history_key = f"structure:history:{page_type}"
            await self.redis_client.lpush(
                history_key, json.dumps({"timestamp": datetime.now(UTC).isoformat(), "structure": structure})
            )
            # Keep only last 10 versions
            await self.redis_client.ltrim(history_key, 0, 9)

        # Update local cache
        self._baseline_cache[page_type] = structure

        logger.info(f"Stored baseline structure for {page_type}")

    async def get_baseline(self, page_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve baseline structure from Redis.

        Args:
            page_type: Type of page

        Returns:
            Baseline structure or None if not found
        """
        # Check local cache first
        if page_type in self._baseline_cache:
            return self._baseline_cache[page_type]

        if self.redis_client:
            key = f"structure:baseline:{page_type}"
            data = await self.redis_client.get(key)
            if data:
                structure: Dict[str, Any] = json.loads(data)
                self._baseline_cache[page_type] = structure
                return structure

        return None

    async def check_for_changes(self, html: str, page_type: str) -> Optional[ChangeReport]:
        """Check if HTML structure has changed from baseline.

        Args:
            html: Current HTML to check
            page_type: Type of page

        Returns:
            ChangeReport if changes detected, None if no baseline exists
        """
        baseline = await self.get_baseline(page_type)
        if not baseline:
            # No baseline, store this as the baseline
            structure = self.capture_structure_fingerprint(html, page_type)
            await self.store_baseline(page_type, structure)
            return None

        current = self.capture_structure_fingerprint(html, page_type)
        report = self.compare_structures(current, baseline)

        # If significant changes, update baseline after manual review
        if report.has_breaking_changes:
            logger.warning(f"Breaking changes detected for {page_type}: {len(report.changes)} changes")

        return report
