"""Tests for adaptive parser framework."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from bs4 import BeautifulSoup

from services.tracklist_service.src.scrapers.adaptive_parser import (
    ABTestResult,
    AdaptiveParser,
    ExtractionPattern,
    ParserVersion,
)
from services.tracklist_service.src.scrapers.resilient_extractor import (
    CSSStrategy,
    ExtractedData,
    RegexStrategy,
    TextStrategy,
    XPathStrategy,
)


class TestParserVersion:
    """Test ParserVersion data structure."""

    def test_parser_version_creation(self):
        """Test parser version creation."""
        now = datetime.now(UTC)
        strategies = {"tracklist": {"title": [{"type": "CSS", "selector": "h1"}]}}

        version = ParserVersion(
            version="1.0.0",
            created_at=now,
            strategies=strategies,
        )

        assert version.version == "1.0.0"
        assert version.created_at == now
        assert version.success_rate == 0.0
        assert version.usage_count == 0
        assert version.strategies == strategies
        assert version.active is True

    def test_parser_version_to_dict(self):
        """Test parser version serialization."""
        now = datetime.now(UTC)
        strategies = {"test": {"field": [{"type": "CSS", "selector": "div"}]}}

        version = ParserVersion(
            version="1.0.0",
            created_at=now,
            success_rate=0.95,
            usage_count=100,
            strategies=strategies,
            metadata={"test": "metadata"},
        )

        version_dict = version.to_dict()

        assert version_dict["version"] == "1.0.0"
        assert version_dict["created_at"] == now.isoformat()
        assert version_dict["success_rate"] == 0.95
        assert version_dict["usage_count"] == 100
        assert version_dict["strategies"] == strategies
        assert version_dict["metadata"] == {"test": "metadata"}
        assert version_dict["active"] is True


class TestExtractionPattern:
    """Test ExtractionPattern data structure."""

    def test_extraction_pattern_creation(self):
        """Test extraction pattern creation."""
        pattern = ExtractionPattern(
            field="title",
            strategy_type="CSS",
            selector="h1",
        )

        assert pattern.field == "title"
        assert pattern.strategy_type == "CSS"
        assert pattern.selector == "h1"
        assert pattern.usage_count == 0
        assert pattern.success_count == 0
        assert pattern.confidence == 0.0

    def test_extraction_pattern_update_stats_success(self):
        """Test updating pattern statistics with success."""
        pattern = ExtractionPattern(
            field="title",
            strategy_type="CSS",
            selector="h1",
        )

        pattern.update_stats(success=True)

        assert pattern.usage_count == 1
        assert pattern.success_count == 1
        assert pattern.confidence == 1.0

    def test_extraction_pattern_update_stats_failure(self):
        """Test updating pattern statistics with failure."""
        pattern = ExtractionPattern(
            field="title",
            strategy_type="CSS",
            selector="h1",
        )

        pattern.update_stats(success=False)

        assert pattern.usage_count == 1
        assert pattern.success_count == 0
        assert pattern.confidence == 0.0

    def test_extraction_pattern_confidence_calculation(self):
        """Test confidence calculation over multiple updates."""
        pattern = ExtractionPattern(
            field="title",
            strategy_type="CSS",
            selector="h1",
        )

        # 3 successes, 1 failure
        pattern.update_stats(success=True)
        pattern.update_stats(success=True)
        pattern.update_stats(success=True)
        pattern.update_stats(success=False)

        assert pattern.usage_count == 4
        assert pattern.success_count == 3
        assert pattern.confidence == 0.75


class TestABTestResult:
    """Test A/B testing functionality."""

    def test_ab_test_creation(self):
        """Test A/B test creation."""
        test = ABTestResult(
            test_id="test_1",
            field="title",
            strategy_a={"type": "CSS", "selector": "h1"},
            strategy_b={"type": "CSS", "selector": ".title"},
        )

        assert test.test_id == "test_1"
        assert test.field == "title"
        assert test.sample_size == 0
        assert test.a_success_rate == 0.0
        assert test.b_success_rate == 0.0
        assert test.winner is None

    def test_ab_test_update_results(self):
        """Test updating A/B test results."""
        test = ABTestResult(
            test_id="test_1",
            field="title",
            strategy_a={"type": "CSS", "selector": "h1"},
            strategy_b={"type": "CSS", "selector": ".title"},
        )

        # Simulate test results
        test.sample_size = 100
        test.a_success_rate = 0.85
        test.b_success_rate = 0.92
        test.winner = "B"
        test.confidence_level = 0.95

        assert test.winner == "B"
        assert test.confidence_level == 0.95


class TestAdaptiveParser:
    """Test AdaptiveParser functionality."""

    @pytest.fixture
    def config_data(self):
        """Sample configuration data."""
        return {
            "strategies": {
                "tracklist": {
                    "title": [
                        {"type": "CSS", "selector": "h1", "priority": 1},
                        {"type": "CSS", "selector": ".title", "priority": 2},
                    ],
                    "artist": [{"type": "CSS", "selector": ".artist", "priority": 1}],
                }
            },
            "learning": {"enabled": True, "min_confidence": 0.7},
            "ab_testing": {"enabled": True, "sample_size": 100},
        }

    @pytest.fixture
    def adaptive_parser(self, config_data):
        """Create AdaptiveParser instance."""
        with (
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
            patch("pathlib.Path.exists", return_value=True),
        ):
            return AdaptiveParser(config_path=Path("test_config.json"))

    def test_adaptive_parser_init(self, adaptive_parser):
        """Test AdaptiveParser initialization."""
        assert adaptive_parser.learning_enabled is True
        assert adaptive_parser.ab_testing_enabled is True
        assert "tracklist" in adaptive_parser._config["strategies"]

    def test_adaptive_parser_init_no_config(self):
        """Test AdaptiveParser initialization without config file."""
        with patch("pathlib.Path.exists", return_value=False):
            parser = AdaptiveParser(config_path=Path("nonexistent.json"))

            # Should use default config
            assert parser._config["strategies"] == {}
            assert parser.learning_enabled is True

    def test_get_strategies_for_field(self, adaptive_parser):
        """Test getting strategies for a field."""
        strategies = adaptive_parser.get_strategies_for_field("tracklist", "title")

        assert len(strategies) == 2
        assert strategies[0].selector == "h1"  # Priority 1
        assert strategies[1].selector == ".title"  # Priority 2

    def test_get_strategies_for_field_missing(self, adaptive_parser):
        """Test getting strategies for missing field."""
        strategies = adaptive_parser.get_strategies_for_field("tracklist", "missing_field")

        assert strategies == []

    def test_get_strategies_for_field_missing_page_type(self, adaptive_parser):
        """Test getting strategies for missing page type."""
        strategies = adaptive_parser.get_strategies_for_field("missing_page", "title")

        assert strategies == []

    def test_learn_patterns(self, adaptive_parser):
        """Test pattern learning."""
        successful_extractions = [
            {"field": "tracklist.title", "strategy_type": "CSS", "selector": "h1"},
            {
                "field": "tracklist.artist",
                "strategy_type": "CSS",
                "selector": ".artist",
            },
        ]

        adaptive_parser.learn_patterns(successful_extractions)

        # Should have learned patterns
        assert "tracklist.title" in adaptive_parser._patterns
        assert "tracklist.artist" in adaptive_parser._patterns

        title_patterns = adaptive_parser._patterns["tracklist.title"]
        assert len(title_patterns) == 1
        assert title_patterns[0].selector == "h1"
        assert title_patterns[0].success_count == 1

    def test_learn_patterns_disabled(self, adaptive_parser):
        """Test pattern learning when disabled."""
        adaptive_parser.learning_enabled = False

        successful_extractions = [{"field": "tracklist.title", "strategy_type": "CSS", "selector": "h1"}]

        adaptive_parser.learn_patterns(successful_extractions)

        # Should not have learned patterns
        assert len(adaptive_parser._patterns) == 0

    def test_learn_patterns_invalid_data(self, adaptive_parser):
        """Test pattern learning with invalid data."""
        invalid_extractions = [
            {"field": None, "strategy_type": "CSS", "selector": "h1"},
            {"field": "title", "strategy_type": None, "selector": "h1"},
            {"field": "title", "strategy_type": "CSS", "selector": None},
            {"field": "title", "strategy_type": "CSS"},  # Missing selector
        ]

        adaptive_parser.learn_patterns(invalid_extractions)

        # Should not have learned any patterns
        assert len(adaptive_parser._patterns) == 0

    def test_promote_pattern(self, adaptive_parser):
        """Test pattern promotion to configuration."""
        # Create a high-confidence pattern
        pattern = ExtractionPattern(
            field="tracklist.description",
            strategy_type="CSS",
            selector=".description",
        )
        # Simulate high confidence
        for _ in range(10):
            pattern.update_stats(success=True)

        adaptive_parser._patterns["tracklist.description"] = [pattern]
        adaptive_parser._promote_pattern(pattern)

        # Should be added to configuration
        strategies = adaptive_parser._config["strategies"]["tracklist"]
        assert "description" in strategies
        assert any(s["selector"] == ".description" for s in strategies["description"])

    def test_hot_reload_config(self, adaptive_parser, config_data):
        """Test hot reloading configuration."""
        # Modify config data
        new_config_data = config_data.copy()
        new_config_data["strategies"]["tracklist"]["new_field"] = [
            {"type": "CSS", "selector": ".new-field", "priority": 1}
        ]

        with patch("builtins.open", mock_open(read_data=json.dumps(new_config_data))):
            adaptive_parser.hot_reload_config()

        # Should have new field
        assert "new_field" in adaptive_parser._config["strategies"]["tracklist"]

    def test_hot_reload_config_file_error(self, adaptive_parser):
        """Test hot reload with file error."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            # Should not raise error
            adaptive_parser.hot_reload_config()

    def test_start_ab_test(self, adaptive_parser):
        """Test starting A/B test."""
        strategy_a = {"type": "CSS", "selector": "h1"}
        strategy_b = {"type": "CSS", "selector": ".title"}

        test = adaptive_parser.start_ab_test("title", strategy_a, strategy_b, sample_size=50)

        assert test.test_id.startswith("title_")
        assert test.field == "title"
        assert test.strategy_a == strategy_a
        assert test.strategy_b == strategy_b
        assert test in adaptive_parser._ab_tests.values()

    def test_start_ab_test_disabled(self, adaptive_parser):
        """Test A/B test when disabled."""
        adaptive_parser.ab_testing_enabled = False

        test = adaptive_parser.start_ab_test(
            "title",
            {"type": "CSS", "selector": "h1"},
            {"type": "CSS", "selector": ".title"},
        )

        assert test is None

    def test_update_ab_test_results(self, adaptive_parser):
        """Test updating A/B test results."""
        test = ABTestResult(
            test_id="test_1",
            field="title",
            strategy_a={"type": "CSS", "selector": "h1"},
            strategy_b={"type": "CSS", "selector": ".title"},
        )
        adaptive_parser._ab_tests["test_1"] = test

        # Update with strategy A success
        adaptive_parser.update_ab_test_results("test_1", "A", success=True)

        assert test.sample_size == 1
        assert test.a_success_rate == 1.0

        # Update with strategy B failure
        adaptive_parser.update_ab_test_results("test_1", "B", success=False)

        assert test.sample_size == 2
        assert test.b_success_rate == 0.0

    def test_create_strategy_from_config(self, adaptive_parser):
        """Test creating strategy from configuration."""
        # CSS strategy
        css_config = {"type": "CSS", "selector": "h1", "attribute": "text"}
        strategy = adaptive_parser._create_strategy_from_config(css_config)
        assert isinstance(strategy, CSSStrategy)
        assert strategy.selector == "h1"

        # XPath strategy
        xpath_config = {"type": "XPATH", "selector": "//h1"}
        strategy = adaptive_parser._create_strategy_from_config(xpath_config)
        assert isinstance(strategy, XPathStrategy)

        # Text strategy
        text_config = {"type": "TEXT", "selector": "Title:", "context": {"before": 10}}
        strategy = adaptive_parser._create_strategy_from_config(text_config)
        assert isinstance(strategy, TextStrategy)

        # Regex strategy
        regex_config = {"type": "REGEX", "selector": r"Title: (.+)", "group": 1}
        strategy = adaptive_parser._create_strategy_from_config(regex_config)
        assert isinstance(strategy, RegexStrategy)

    def test_create_strategy_from_config_invalid(self, adaptive_parser):
        """Test creating strategy from invalid configuration."""
        # Invalid type
        invalid_config = {"type": "INVALID", "selector": "h1"}
        strategy = adaptive_parser._create_strategy_from_config(invalid_config)
        assert strategy is None

        # Missing selector
        missing_selector_config = {"type": "CSS"}
        strategy = adaptive_parser._create_strategy_from_config(missing_selector_config)
        assert strategy is None

    def test_create_version(self, adaptive_parser):
        """Test creating parser version."""
        strategies = {"tracklist": {"title": [{"type": "CSS", "selector": "h1"}]}}

        version = adaptive_parser.create_version("2.0.0", strategies)

        assert version.version == "2.0.0"
        assert version.strategies == strategies
        assert "2.0.0" in adaptive_parser._versions

    def test_rollback_version(self, adaptive_parser):
        """Test rolling back to previous version."""
        # Create a version to rollback to
        old_strategies = {"tracklist": {"title": [{"type": "CSS", "selector": ".old-title"}]}}
        adaptive_parser.create_version("1.0.0", old_strategies)

        # Rollback
        success = adaptive_parser.rollback_version("1.0.0")

        assert success is True
        assert adaptive_parser._current_version == "1.0.0"
        assert adaptive_parser._config["strategies"] == old_strategies

    def test_rollback_version_missing(self, adaptive_parser):
        """Test rolling back to missing version."""
        success = adaptive_parser.rollback_version("missing_version")

        assert success is False

    def test_get_version_history(self, adaptive_parser):
        """Test getting version history."""
        # Create multiple versions
        adaptive_parser.create_version("1.0.0", {"old": "config"})
        adaptive_parser.create_version("2.0.0", {"new": "config"})

        history = adaptive_parser.get_version_history()

        assert len(history) == 2
        # Should be sorted by creation time (newest first)
        assert history[0].version == "2.0.0"
        assert history[1].version == "1.0.0"

    @pytest.mark.asyncio
    async def test_update_selector_config(self, adaptive_parser):
        """Test updating selector configuration."""
        await adaptive_parser.update_selector_config(
            page_type="tracklist",
            field_name="description",
            selector_type="css",
            selector_value=".description",
            priority=1,
        )

        strategies = adaptive_parser._config["strategies"]["tracklist"]["description"]
        assert len(strategies) == 1
        assert strategies[0]["type"] == "CSS"
        assert strategies[0]["selector"] == ".description"
        assert strategies[0]["priority"] == 1

    @pytest.mark.asyncio
    async def test_update_selector_config_with_priority(self, adaptive_parser):
        """Test updating selector config with priority ordering."""
        # Add multiple selectors with different priorities
        await adaptive_parser.update_selector_config("tracklist", "title", "css", ".title-low", priority=3)
        await adaptive_parser.update_selector_config("tracklist", "title", "css", ".title-high", priority=1)
        await adaptive_parser.update_selector_config("tracklist", "title", "css", ".title-mid", priority=2)

        strategies = adaptive_parser._config["strategies"]["tracklist"]["title"]
        # Should be ordered by priority (lower number = higher priority)
        selectors = [s["selector"] for s in strategies]
        assert selectors == ["h1", ".title-high", ".title-mid", ".title", ".title-low"]

    @pytest.mark.asyncio
    async def test_version_exists(self, adaptive_parser):
        """Test checking if version exists."""
        adaptive_parser.create_version("1.0.0", {})

        assert await adaptive_parser.version_exists("1.0.0") is True
        assert await adaptive_parser.version_exists("missing") is False

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, adaptive_parser):
        """Test rolling back to specific version."""
        # Create version to rollback to
        old_strategies = {"old": "config"}
        adaptive_parser.create_version("1.0.0", old_strategies)

        await adaptive_parser.rollback_to_version("1.0.0")

        assert adaptive_parser._current_version == "1.0.0"
        assert adaptive_parser._config["strategies"] == old_strategies

    @pytest.mark.asyncio
    async def test_rollback_to_version_missing(self, adaptive_parser):
        """Test rolling back to missing version."""
        with pytest.raises(ValueError, match="does not exist"):
            await adaptive_parser.rollback_to_version("missing")

    @pytest.mark.asyncio
    async def test_rollback_to_version_too_old(self, adaptive_parser):
        """Test rolling back to old version without force."""
        # Create old version
        old_time = datetime.now(UTC) - timedelta(days=35)
        old_version = ParserVersion(version="old", created_at=old_time, strategies={})
        adaptive_parser._versions["old"] = old_version

        with pytest.raises(ValueError, match="days old"):
            await adaptive_parser.rollback_to_version("old", force=False)

    @pytest.mark.asyncio
    async def test_rollback_to_version_force_old(self, adaptive_parser):
        """Test rolling back to old version with force."""
        # Create old version
        old_time = datetime.now(UTC) - timedelta(days=35)
        old_version = ParserVersion(version="old", created_at=old_time, strategies={"old": "config"})
        adaptive_parser._versions["old"] = old_version

        # Should succeed with force=True
        await adaptive_parser.rollback_to_version("old", force=True)
        assert adaptive_parser._current_version == "old"

    def test_get_current_version(self, adaptive_parser):
        """Test getting current version."""
        # Create and set current version
        strategies = {"current": "config"}
        adaptive_parser.create_version("current", strategies)
        adaptive_parser._current_version = "current"

        current = adaptive_parser.get_current_version()

        assert current.version == "current"
        assert current.strategies == strategies

    def test_get_current_version_default(self, adaptive_parser):
        """Test getting current version when none set."""
        current = adaptive_parser.get_current_version()

        assert current.version == "default"
        assert current.strategies == adaptive_parser._config.get("strategies", {})

    def test_parse_with_adaptation(self, adaptive_parser):
        """Test parsing with adaptive strategies."""
        html = """
        <html>
            <body>
                <h1>Test Title</h1>
                <div class="artist">Test Artist</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")

        # Mock the extractor
        mock_result = ExtractedData(
            data={"title": "Test Title", "artist": "Test Artist"},
            strategies_used={"title": "CSS", "artist": "CSS"},
            quality_score=1.0,
            metadata={},
        )

        with patch.object(
            adaptive_parser._extractor,
            "extract_multiple_fields",
            return_value=mock_result,
        ):
            result = adaptive_parser.parse_with_adaptation(soup, "tracklist", ["title", "artist"])

        assert result.data["title"] == "Test Title"
        assert result.data["artist"] == "Test Artist"

    def test_parse_with_adaptation_learning(self, adaptive_parser):
        """Test parsing with learning enabled."""
        html = "<html><h1>Test</h1></html>"
        soup = BeautifulSoup(html, "html.parser")

        mock_result = ExtractedData(
            data={"title": "Test"},
            strategies_used={"title": "CSS"},
            quality_score=1.0,
            metadata={},
        )

        with (
            patch.object(
                adaptive_parser._extractor,
                "extract_multiple_fields",
                return_value=mock_result,
            ),
            patch.object(adaptive_parser, "learn_patterns") as mock_learn,
        ):
            adaptive_parser.parse_with_adaptation(soup, "tracklist", ["title"])

            # Should call learn_patterns
            mock_learn.assert_called_once()
            call_args = mock_learn.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0]["field"] == "tracklist.title"

    @pytest.mark.asyncio
    async def test_start_hot_reload(self, adaptive_parser):
        """Test starting hot reload monitoring."""
        # Mock the reload interval to be very short for testing
        adaptive_parser.hot_reload_interval = 0.01

        with patch.object(adaptive_parser, "hot_reload_config") as mock_reload:
            # Start hot reload
            await adaptive_parser.start_hot_reload()

            # Let it run briefly
            await asyncio.sleep(0.05)

            # Stop hot reload
            await adaptive_parser.stop_hot_reload()

            # Should have called hot_reload_config at least once
            assert mock_reload.call_count >= 1

    @pytest.mark.asyncio
    async def test_stop_hot_reload(self, adaptive_parser):
        """Test stopping hot reload monitoring."""
        # Start hot reload
        adaptive_parser.hot_reload_interval = 0.01
        await adaptive_parser.start_hot_reload()

        # Stop it
        await adaptive_parser.stop_hot_reload()

        # Tasks should be cleared
        assert len(adaptive_parser._hot_reload_tasks) == 0

    def test_save_config(self, adaptive_parser):
        """Test saving configuration."""
        with patch("builtins.open", mock_open()) as mock_file:
            adaptive_parser._save_config()

            # Should have written to file
            mock_file.assert_called_once()
            handle = mock_file()
            handle.write.assert_called()

    def test_save_config_error(self, adaptive_parser):
        """Test saving configuration with error."""
        with patch("builtins.open", side_effect=OSError("Write error")):
            # Should not raise error
            adaptive_parser._save_config()

    def test_calculate_confidence(self, adaptive_parser):
        """Test confidence calculation for A/B tests."""
        # Simple confidence calculation (actual implementation may vary)
        confidence = adaptive_parser._calculate_confidence(0.95, 0.85, 100)

        assert 0.0 <= confidence <= 1.0
        # Higher success rate difference should give higher confidence
        high_confidence = adaptive_parser._calculate_confidence(0.95, 0.70, 100)
        low_confidence = adaptive_parser._calculate_confidence(0.95, 0.90, 100)

        assert high_confidence >= low_confidence
