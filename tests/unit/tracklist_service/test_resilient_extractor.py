"""Tests for resilient data extraction with fallback strategies."""

import pytest
from bs4 import BeautifulSoup

from services.tracklist_service.src.scrapers.resilient_extractor import (
    CSSStrategy,
    ExtractedData,
    RegexStrategy,
    ResilientExtractor,
    TextStrategy,
    XPathStrategy,
)


@pytest.fixture
def sample_html():
    """Sample HTML for testing extraction."""
    return """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <div class="container">
            <h1 class="title">Main Title</h1>
            <p class="date">2024-01-15</p>
            <div class="content">
                <p>This is some content text.</p>
                <a href="/link1" class="link">Link 1</a>
                <a href="/link2" class="link">Link 2</a>
            </div>
            <div class="metadata">
                <span data-id="123">ID: 123</span>
                <span class="author">John Doe</span>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def complex_html():
    """Complex HTML with inconsistent structure."""
    return """
    <html>
    <body>
        <div class="results">
            <article>
                <h2>Article Title</h2>
                <time datetime="2024-01-20">January 20, 2024</time>
                <div class="text">Article content here</div>
            </article>
            <article>
                <h2>Another Title</h2>
                <!-- Missing time element -->
                <div class="text">More content</div>
            </article>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def extractor():
    """Create resilient extractor instance."""
    return ResilientExtractor(default_quality_threshold=0.5)


class TestExtractionStrategies:
    """Test individual extraction strategies."""

    def test_css_strategy_single_element(self, sample_html):
        """Test CSS strategy with single element."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = CSSStrategy("h1.title")

        result = strategy.extract(soup)

        assert result.success is True
        assert result.value == "Main Title"
        assert result.strategy_used == "CSS"
        assert result.quality_score == 1.0

    def test_css_strategy_multiple_elements(self, sample_html):
        """Test CSS strategy with multiple elements."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = CSSStrategy("a.link")

        result = strategy.extract(soup)

        assert result.success is True
        assert isinstance(result.value, list)
        assert len(result.value) == 2
        assert "Link 1" in result.value
        assert "Link 2" in result.value

    def test_css_strategy_with_attribute(self, sample_html):
        """Test CSS strategy extracting attribute."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = CSSStrategy("a.link", attribute="href")

        result = strategy.extract(soup)

        assert result.success is True
        assert isinstance(result.value, list)
        assert "/link1" in result.value
        assert "/link2" in result.value

    def test_css_strategy_not_found(self, sample_html):
        """Test CSS strategy when element not found."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = CSSStrategy(".nonexistent")

        result = strategy.extract(soup)

        assert result.success is False
        assert result.value is None
        assert len(result.errors) > 0

    def test_xpath_strategy(self, sample_html):
        """Test XPath strategy."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = XPathStrategy("//h1[@class='title']/text()")

        result = strategy.extract(soup)

        assert result.success is True
        assert result.value == "Main Title"
        assert result.strategy_used == "XPath"
        assert result.quality_score == 0.95

    def test_xpath_strategy_multiple(self, sample_html):
        """Test XPath strategy with multiple results."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = XPathStrategy("//a/@href")

        result = strategy.extract(soup)

        assert result.success is True
        assert isinstance(result.value, list)
        assert "/link1" in result.value
        assert "/link2" in result.value

    def test_text_strategy(self, sample_html):
        """Test text-based strategy."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = TextStrategy("content text")

        result = strategy.extract(soup)

        assert result.success is True
        assert "content text" in result.value
        assert result.strategy_used == "Text"
        assert result.quality_score == 0.7

    def test_text_strategy_with_context(self, sample_html):
        """Test text strategy with context."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = TextStrategy("John", context=".metadata")

        result = strategy.extract(soup)

        assert result.success is True
        assert "John Doe" in result.value

    def test_regex_strategy(self, sample_html):
        """Test regex strategy."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = RegexStrategy(r"\d{4}-\d{2}-\d{2}")

        result = strategy.extract(soup)

        assert result.success is True
        assert result.value == "2024-01-15"
        assert result.strategy_used == "Regex"
        assert result.quality_score == 0.6

    def test_regex_strategy_with_groups(self, sample_html):
        """Test regex strategy with groups."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategy = RegexStrategy(r"ID:\s*(\d+)", group=1)

        result = strategy.extract(soup)

        assert result.success is True
        assert result.value == "123"


class TestResilientExtractor:
    """Test ResilientExtractor class."""

    def test_extract_with_fallback_first_success(self, extractor, sample_html):
        """Test extraction with fallback using first successful strategy."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategies = [
            CSSStrategy(".nonexistent"),  # Will fail
            CSSStrategy("h1.title"),  # Will succeed
            XPathStrategy("//h1/text()"),  # Won't be tried
        ]

        result = extractor.extract_with_fallback(soup, strategies, "title")

        assert "title" in result.data
        assert result.data["title"] == "Main Title"
        assert result.strategies_used["title"] == "CSS"
        assert result.quality_score == 1.0
        assert not result.partial_extraction

    def test_extract_with_fallback_quality_preference(self, extractor, sample_html):
        """Test extraction prefers higher quality results."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategies = [
            RegexStrategy(r"Main.*"),  # Lower quality (0.6)
            CSSStrategy("h1.title"),  # Higher quality (1.0)
        ]

        result = extractor.extract_with_fallback(soup, strategies, "title")

        assert result.data["title"] == "Main Title"
        assert result.strategies_used["title"] == "CSS"
        assert result.quality_score == 1.0

    def test_extract_with_fallback_all_fail(self, extractor, sample_html):
        """Test extraction when all strategies fail."""
        soup = BeautifulSoup(sample_html, "html.parser")
        strategies = [CSSStrategy(".nonexistent"), XPathStrategy("//missing/text()"), TextStrategy("not found")]

        result = extractor.extract_with_fallback(soup, strategies, "missing")

        assert "missing" not in result.data
        assert "missing" in result.missing_fields
        assert result.partial_extraction is True
        assert len(result.extraction_errors) > 0

    def test_extract_multiple_fields(self, extractor, sample_html):
        """Test extracting multiple fields."""
        soup = BeautifulSoup(sample_html, "html.parser")
        field_strategies = {
            "title": [CSSStrategy("h1.title")],
            "date": [CSSStrategy(".date")],
            "author": [CSSStrategy(".author")],
            "links": [CSSStrategy("a", attribute="href")],
        }

        result = extractor.extract_multiple_fields(soup, field_strategies)

        assert "title" in result.data
        assert result.data["title"] == "Main Title"
        assert "date" in result.data
        assert result.data["date"] == "2024-01-15"
        assert "author" in result.data
        assert result.data["author"] == "John Doe"
        assert "links" in result.data
        assert isinstance(result.data["links"], list)
        assert not result.partial_extraction
        assert result.quality_score > 0.9

    def test_extract_multiple_fields_partial(self, extractor, complex_html):
        """Test partial extraction when some fields missing."""
        soup = BeautifulSoup(complex_html, "html.parser")
        field_strategies = {
            "title": [CSSStrategy("h2")],
            "date": [CSSStrategy("time")],
            "content": [CSSStrategy(".text")],
        }

        result = extractor.extract_multiple_fields(soup, field_strategies)

        assert "title" in result.data
        assert "content" in result.data
        # Date might be present (first article) or missing (overall)
        assert result.partial_extraction is False or "date" in result.missing_fields

    def test_calculate_quality_score(self, extractor):
        """Test quality score calculation."""
        # High quality data
        data1 = ExtractedData(
            data={"field1": "value1", "field2": "value2"}, quality_score=0.9, partial_extraction=False
        )
        score1 = extractor.calculate_quality_score(data1)
        assert score1 == 0.9

        # Partial extraction with missing fields
        data2 = ExtractedData(
            data={"field1": "value1"}, quality_score=0.8, partial_extraction=True, missing_fields=["field2", "field3"]
        )
        score2 = extractor.calculate_quality_score(data2)
        assert score2 < 0.8  # Penalized for missing fields and partial extraction

        # Data with errors
        data3 = ExtractedData(
            data={"field1": "value1"}, quality_score=0.7, extraction_errors=["error1", "error2", "error3"]
        )
        score3 = extractor.calculate_quality_score(data3)
        assert score3 < 0.7  # Penalized for errors

    def test_create_default_strategies(self, extractor):
        """Test creating default strategies for common field types."""
        # Title strategies
        title_strategies = extractor.create_default_strategies("title")
        assert len(title_strategies) > 0
        assert any(isinstance(s, CSSStrategy) for s in title_strategies)
        assert any(isinstance(s, XPathStrategy) for s in title_strategies)

        # Date strategies
        date_strategies = extractor.create_default_strategies("date")
        assert len(date_strategies) > 0
        assert any(isinstance(s, RegexStrategy) for s in date_strategies)

        # Link strategies
        link_strategies = extractor.create_default_strategies("link")
        assert len(link_strategies) > 0
        css_link = next((s for s in link_strategies if isinstance(s, CSSStrategy)), None)
        assert css_link is not None
        assert css_link.attribute == "href"

        # Generic strategies
        generic_strategies = extractor.create_default_strategies("custom")
        assert len(generic_strategies) > 0

    def test_quality_threshold(self, sample_html):
        """Test quality threshold filtering."""
        extractor_strict = ResilientExtractor(default_quality_threshold=0.8)
        soup = BeautifulSoup(sample_html, "html.parser")

        # Regex strategy returns quality 0.6, below threshold
        strategies = [RegexStrategy(r"Main.*")]

        result = extractor_strict.extract_with_fallback(soup, strategies, "title")

        # Should reject due to low quality
        assert "title" not in result.data
        assert "title" in result.missing_fields
