"""Adaptive parser framework with self-healing capabilities."""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib

from services.tracklist_service.src.scrapers.resilient_extractor import (
    ResilientExtractor,
    ExtractionStrategy,
    CSSStrategy,
    XPathStrategy,
    TextStrategy,
    RegexStrategy,
    ExtractedData,
)

logger = logging.getLogger(__name__)


@dataclass
class ParserVersion:
    """Version information for parser configuration."""

    version: str
    created_at: datetime
    success_rate: float = 0.0
    usage_count: int = 0
    strategies: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "strategies": self.strategies,
            "metadata": self.metadata,
            "active": self.active,
        }


@dataclass
class ExtractionPattern:
    """Learned extraction pattern."""

    field: str
    strategy_type: str
    selector: str
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[datetime] = None
    confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def update_stats(self, success: bool) -> None:
        """Update pattern statistics."""
        if success:
            self.success_count += 1
            self.last_success = datetime.now(UTC)
        else:
            self.failure_count += 1

        # Update confidence based on recent performance
        self.confidence = self.success_rate * (0.9 if self.last_success else 0.5)


@dataclass
class ABTestResult:
    """Results from A/B testing strategies."""

    strategy_a: str
    strategy_b: str
    winner: Optional[str] = None
    a_success_rate: float = 0.0
    b_success_rate: float = 0.0
    sample_size: int = 0
    confidence_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveParser:
    """Parser that adapts and learns from successful extractions."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        learning_enabled: bool = True,
        hot_reload_interval: int = 60,
    ):
        """Initialize adaptive parser.

        Args:
            config_path: Path to configuration file
            learning_enabled: Enable pattern learning
            hot_reload_interval: Config reload check interval in seconds
        """
        self.config_path = config_path or Path("config/parser_config.json")
        self.learning_enabled = learning_enabled
        self.hot_reload_interval = hot_reload_interval
        self._config: Dict[str, Any] = {}
        self._config_hash: Optional[str] = None
        self._patterns: Dict[str, List[ExtractionPattern]] = {}
        self._versions: Dict[str, ParserVersion] = {}
        self._current_version = "1.0.0"
        self._extractor = ResilientExtractor()
        self._reload_task: Optional[asyncio.Task] = None
        self._ab_tests: Dict[str, ABTestResult] = {}

        # Load initial configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                content = f.read()
                self._config = json.loads(content)
                self._config_hash = hashlib.md5(content.encode()).hexdigest()
        else:
            self._config = self._get_default_config()
            self._save_config()

        # Initialize version
        if self._current_version not in self._versions:
            self._versions[self._current_version] = ParserVersion(
                version=self._current_version,
                created_at=datetime.now(UTC),
                strategies=self._config.get("strategies", {}),
            )

    def _get_default_config(self) -> dict:
        """Get default parser configuration."""
        return {
            "version": "1.0.0",
            "strategies": {
                "search": {
                    "title": [
                        {"type": "CSS", "selector": ".result-title"},
                        {"type": "CSS", "selector": "h3 a"},
                        {"type": "XPath", "selector": "//div[@class='result']//a/text()"},
                    ],
                    "url": [
                        {"type": "CSS", "selector": ".result-link", "attribute": "href"},
                        {"type": "XPath", "selector": "//a[@class='result-link']/@href"},
                    ],
                    "date": [
                        {"type": "CSS", "selector": ".result-date"},
                        {"type": "Regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
                    ],
                },
                "tracklist": {
                    "tracks": [
                        {"type": "CSS", "selector": ".track-item"},
                        {"type": "XPath", "selector": "//div[@class='tracklist']//div[@class='track']"},
                    ],
                    "time": [
                        {"type": "CSS", "selector": ".track-time"},
                        {"type": "Regex", "pattern": r"\d{2}:\d{2}"},
                    ],
                },
            },
            "learning": {
                "enabled": True,
                "min_confidence": 0.7,
                "pattern_threshold": 10,
            },
            "ab_testing": {
                "enabled": False,
                "sample_size": 100,
                "confidence_threshold": 0.95,
            },
        }

    def _save_config(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def learn_patterns(self, successful_extractions: List[dict]) -> None:
        """Learn from successful extractions.

        Args:
            successful_extractions: List of successful extraction results
        """
        if not self.learning_enabled:
            return

        for extraction in successful_extractions:
            field = extraction.get("field")
            strategy_type = extraction.get("strategy_type")
            selector = extraction.get("selector")

            if not all([field, strategy_type, selector]) or not isinstance(field, str):
                continue

            field_str = str(field)
            strategy_type_str = str(strategy_type)
            selector_str = str(selector)

            # Find or create pattern
            if field_str not in self._patterns:
                self._patterns[field_str] = []

            pattern = next(
                (
                    p
                    for p in self._patterns[field_str]
                    if p.strategy_type == strategy_type_str and p.selector == selector_str
                ),
                None,
            )

            if not pattern:
                pattern = ExtractionPattern(
                    field=field_str,
                    strategy_type=strategy_type_str,
                    selector=selector_str,
                )
                self._patterns[field_str].append(pattern)

            # Update pattern statistics
            pattern.update_stats(success=True)

            # Promote pattern if confidence is high
            if pattern.confidence > self._config.get("learning", {}).get("min_confidence", 0.7):
                self._promote_pattern(pattern)

    def _promote_pattern(self, pattern: ExtractionPattern) -> None:
        """Promote a successful pattern to configuration.

        Args:
            pattern: Pattern to promote
        """
        page_type = pattern.field.split(".")[0] if "." in pattern.field else "default"
        field_name = pattern.field.split(".")[-1]

        if page_type not in self._config["strategies"]:
            self._config["strategies"][page_type] = {}

        if field_name not in self._config["strategies"][page_type]:
            self._config["strategies"][page_type][field_name] = []

        # Check if pattern already exists
        strategies = self._config["strategies"][page_type][field_name]
        exists = any(s["type"] == pattern.strategy_type and s["selector"] == pattern.selector for s in strategies)

        if not exists:
            # Add new strategy at the beginning (higher priority)
            strategies.insert(
                0,
                {
                    "type": pattern.strategy_type,
                    "selector": pattern.selector,
                    "confidence": pattern.confidence,
                    "learned": True,
                },
            )
            logger.info(f"Promoted pattern for {pattern.field}: {pattern.strategy_type} - {pattern.selector}")
            self._save_config()

    def hot_reload_config(self) -> bool:
        """Hot reload configuration if changed.

        Returns:
            True if configuration was reloaded
        """
        if not self.config_path.exists():
            return False

        with open(self.config_path, "r") as f:
            content = f.read()
            new_hash = hashlib.md5(content.encode()).hexdigest()

        if new_hash != self._config_hash:
            try:
                new_config = json.loads(content)
                self._config = new_config
                self._config_hash = new_hash
                logger.info("Configuration hot-reloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to hot-reload configuration: {e}")

        return False

    async def start_hot_reload(self) -> None:
        """Start hot reload monitoring."""

        async def reload_loop() -> None:
            while True:
                try:
                    self.hot_reload_config()
                    await asyncio.sleep(self.hot_reload_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Hot reload error: {e}")
                    await asyncio.sleep(self.hot_reload_interval)

        self._reload_task = asyncio.create_task(reload_loop())

    async def stop_hot_reload(self) -> None:
        """Stop hot reload monitoring."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
            self._reload_task = None

    def ab_test_strategies(self, strategies: List[ExtractionStrategy]) -> ABTestResult:
        """Run A/B test on extraction strategies.

        Args:
            strategies: List of strategies to test

        Returns:
            Test results
        """
        if len(strategies) < 2:
            raise ValueError("At least 2 strategies required for A/B testing")

        # For simplicity, test first two strategies
        strategy_a = strategies[0]
        strategy_b = strategies[1]

        test_id = f"{strategy_a.__class__.__name__}_{strategy_b.__class__.__name__}"

        if test_id not in self._ab_tests:
            self._ab_tests[test_id] = ABTestResult(
                strategy_a=str(strategy_a),
                strategy_b=str(strategy_b),
            )

        return self._ab_tests[test_id]

    def update_ab_test(self, test_id: str, strategy: str, success: bool) -> None:
        """Update A/B test results.

        Args:
            test_id: Test identifier
            strategy: Strategy that was used
            success: Whether extraction was successful
        """
        if test_id not in self._ab_tests:
            return

        test = self._ab_tests[test_id]
        test.sample_size += 1

        # This is simplified - real implementation would track individual results
        if "a" in strategy.lower():
            if success:
                test.a_success_rate = (test.a_success_rate * (test.sample_size - 1) + 1) / test.sample_size
            else:
                test.a_success_rate = (test.a_success_rate * (test.sample_size - 1)) / test.sample_size
        else:
            if success:
                test.b_success_rate = (test.b_success_rate * (test.sample_size - 1) + 1) / test.sample_size
            else:
                test.b_success_rate = (test.b_success_rate * (test.sample_size - 1)) / test.sample_size

        # Determine winner if enough samples
        if test.sample_size >= self._config.get("ab_testing", {}).get("sample_size", 100):
            if test.a_success_rate > test.b_success_rate:
                test.winner = "A"
                test.confidence_level = self._calculate_confidence(
                    test.a_success_rate, test.b_success_rate, test.sample_size
                )
            else:
                test.winner = "B"
                test.confidence_level = self._calculate_confidence(
                    test.b_success_rate, test.a_success_rate, test.sample_size
                )

    def _calculate_confidence(self, rate_a: float, rate_b: float, sample_size: int) -> float:
        """Calculate confidence level for A/B test.

        Args:
            rate_a: Success rate of winner
            rate_b: Success rate of loser
            sample_size: Number of samples

        Returns:
            Confidence level (0-1)
        """
        # Simplified confidence calculation
        diff = abs(rate_a - rate_b)
        if diff > 0.2:
            return 0.99
        elif diff > 0.1:
            return 0.95
        elif diff > 0.05:
            return 0.90
        else:
            return 0.80

    def get_strategies_for_field(self, page_type: str, field: str) -> List[ExtractionStrategy]:
        """Get extraction strategies for a field.

        Args:
            page_type: Type of page (search, tracklist, etc.)
            field: Field name

        Returns:
            List of extraction strategies
        """
        strategies = []

        # Get configured strategies
        if page_type in self._config.get("strategies", {}):
            if field in self._config["strategies"][page_type]:
                for strategy_config in self._config["strategies"][page_type][field]:
                    strategy = self._create_strategy(strategy_config)
                    if strategy:
                        strategies.append(strategy)

        # Add learned patterns
        pattern_key = f"{page_type}.{field}"
        if pattern_key in self._patterns:
            for pattern in sorted(self._patterns[pattern_key], key=lambda p: p.confidence, reverse=True):
                if pattern.confidence > 0.5:
                    strategy = self._create_strategy(
                        {
                            "type": pattern.strategy_type,
                            "selector": pattern.selector,
                        }
                    )
                    if strategy:
                        strategies.append(strategy)

        # Fallback to default strategies
        if not strategies:
            strategies = self._extractor.create_default_strategies(field)

        return strategies

    def _create_strategy(self, config: dict) -> Optional[ExtractionStrategy]:
        """Create extraction strategy from configuration.

        Args:
            config: Strategy configuration

        Returns:
            Extraction strategy or None
        """
        strategy_type = config.get("type", "").upper()
        selector = config.get("selector", "")
        attribute = config.get("attribute")

        if not selector:
            return None

        if strategy_type == "CSS":
            return CSSStrategy(selector, attribute)
        elif strategy_type == "XPATH":
            return XPathStrategy(selector)
        elif strategy_type == "TEXT":
            context = config.get("context")
            return TextStrategy(selector, context)
        elif strategy_type == "REGEX":
            group = config.get("group", 0)
            return RegexStrategy(selector, group)

        return None

    def create_version(self, version: str, strategies: Dict[str, Any]) -> ParserVersion:
        """Create a new parser version.

        Args:
            version: Version identifier
            strategies: Strategy configuration

        Returns:
            New parser version
        """
        parser_version = ParserVersion(
            version=version,
            created_at=datetime.now(UTC),
            strategies=strategies,
        )
        self._versions[version] = parser_version
        return parser_version

    def rollback_version(self, version: str) -> bool:
        """Rollback to a previous parser version.

        Args:
            version: Version to rollback to

        Returns:
            True if rollback successful
        """
        if version not in self._versions:
            logger.error(f"Version {version} not found")
            return False

        parser_version = self._versions[version]
        self._config["strategies"] = parser_version.strategies
        self._current_version = version
        self._save_config()
        logger.info(f"Rolled back to version {version}")
        return True

    def get_version_history(self) -> List[ParserVersion]:
        """Get parser version history.

        Returns:
            List of parser versions
        """
        return sorted(self._versions.values(), key=lambda v: v.created_at, reverse=True)

    def parse_with_adaptation(self, soup: BeautifulSoup, page_type: str, fields: List[str]) -> ExtractedData:
        """Parse with adaptive strategies.

        Args:
            soup: BeautifulSoup object
            page_type: Type of page
            fields: Fields to extract

        Returns:
            Extracted data
        """
        field_strategies = {}
        for field_name in fields:
            strategies = self.get_strategies_for_field(page_type, field_name)
            if strategies:
                field_strategies[field_name] = strategies

        # Extract data
        result = self._extractor.extract_multiple_fields(soup, field_strategies)

        # Learn from results if enabled
        if self.learning_enabled:
            successful = []
            for field_name, value in result.data.items():
                if value and field_name in result.strategies_used:
                    successful.append(
                        {
                            "field": f"{page_type}.{field_name}",
                            "strategy_type": result.strategies_used[field_name],
                            "selector": "unknown",  # Would need to track this
                        }
                    )
            if successful:
                self.learn_patterns(successful)

        return result

    async def update_selector_config(
        self,
        page_type: str,
        field_name: str,
        selector_type: str,
        selector_value: str,
        priority: int = 1,
    ) -> None:
        """Update selector configuration.

        Args:
            page_type: Type of page
            field_name: Field name
            selector_type: Type of selector
            selector_value: Selector value
            priority: Priority in fallback chain
        """
        if page_type not in self._config["strategies"]:
            self._config["strategies"][page_type] = {}

        if field_name not in self._config["strategies"][page_type]:
            self._config["strategies"][page_type][field_name] = []

        # Add new strategy configuration
        strategy_config = {
            "type": selector_type.upper(),
            "selector": selector_value,
            "priority": priority,
        }

        # Insert based on priority
        strategies = self._config["strategies"][page_type][field_name]
        inserted = False
        for i, existing in enumerate(strategies):
            if existing.get("priority", 1) > priority:
                strategies.insert(i, strategy_config)
                inserted = True
                break

        if not inserted:
            strategies.append(strategy_config)

        # Save updated configuration
        self._save_config()

    async def version_exists(self, version: str) -> bool:
        """Check if a version exists.

        Args:
            version: Version to check

        Returns:
            True if version exists
        """
        return version in self._versions

    async def rollback_to_version(self, version: str, force: bool = False) -> None:
        """Rollback to a specific version.

        Args:
            version: Version to rollback to
            force: Force rollback even if version is old

        Raises:
            ValueError: If version doesn't exist or rollback fails
        """
        if not await self.version_exists(version):
            raise ValueError(f"Version {version} does not exist")

        target_version = self._versions[version]
        current_time = datetime.now(UTC)

        # Check if version is too old (more than 30 days) unless forced
        if not force:
            age_days = (current_time - target_version.created_at).days
            if age_days > 30:
                raise ValueError(f"Version {version} is {age_days} days old. Use force=True to rollback.")

        # Perform rollback
        if not self.rollback_version(version):
            raise ValueError(f"Failed to rollback to version {version}")

    def get_current_version(self) -> ParserVersion:
        """Get current parser version.

        Returns:
            Current parser version
        """
        if self._current_version and self._current_version in self._versions:
            return self._versions[self._current_version]

        # Return default version if no current version set
        default_version = ParserVersion(
            version="default",
            created_at=datetime.now(UTC),
            strategies=self._config.get("strategies", {}),
        )
        return default_version
