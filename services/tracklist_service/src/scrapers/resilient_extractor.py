"""Resilient data extraction with fallback strategies."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from bs4 import BeautifulSoup, Tag
import re
import logging
from lxml import etree
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of extraction strategies."""

    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    REGEX = "regex"
    ATTRIBUTE = "attribute"


@dataclass
class ExtractionResult:
    """Result from an extraction attempt."""

    value: Optional[Any] = None
    success: bool = False
    strategy_used: Optional[str] = None
    quality_score: float = 0.0
    partial: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class ExtractedData:
    """Container for extracted data with metadata."""

    data: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    strategies_used: Dict[str, str] = field(default_factory=dict)
    partial_extraction: bool = False
    missing_fields: List[str] = field(default_factory=list)
    extraction_errors: List[str] = field(default_factory=list)


class ExtractionStrategy(ABC):
    """Base class for extraction strategies."""

    def __init__(self, selector: str, attribute: Optional[str] = None):
        """Initialize extraction strategy.

        Args:
            selector: Selector string (CSS, XPath, or pattern)
            attribute: Optional attribute to extract
        """
        self.selector = selector
        self.attribute = attribute

    @abstractmethod
    def extract(self, soup: BeautifulSoup) -> ExtractionResult:
        """Extract data from BeautifulSoup object.

        Args:
            soup: BeautifulSoup object to extract from

        Returns:
            ExtractionResult with extracted data
        """
        pass

    def _get_text_or_attribute(self, element: Tag) -> Optional[str]:
        """Get text or attribute from element."""
        if not element:
            return None

        if self.attribute:
            attr_value = element.get(self.attribute)
            return str(attr_value) if attr_value is not None else None
        text: str = element.get_text(strip=True)
        return text


class CSSStrategy(ExtractionStrategy):
    """CSS selector-based extraction strategy."""

    def extract(self, soup: BeautifulSoup) -> ExtractionResult:
        """Extract using CSS selector."""
        try:
            elements = soup.select(self.selector)
            if not elements:
                return ExtractionResult(success=False, errors=[f"No elements found for selector: {self.selector}"])

            value: Optional[Union[str, List[Optional[str]]]]
            if len(elements) == 1:
                value = self._get_text_or_attribute(elements[0])
            else:
                value = [self._get_text_or_attribute(el) for el in elements]

            return ExtractionResult(value=value, success=True, strategy_used="CSS", quality_score=1.0)
        except Exception as e:
            return ExtractionResult(success=False, errors=[f"CSS extraction failed: {str(e)}"])


class XPathStrategy(ExtractionStrategy):
    """XPath-based extraction strategy."""

    def extract(self, soup: BeautifulSoup) -> ExtractionResult:
        """Extract using XPath."""
        try:
            # Convert BeautifulSoup to lxml
            html_str = str(soup)
            tree = etree.HTML(html_str)

            # Execute XPath
            results = tree.xpath(self.selector)

            if not results:
                return ExtractionResult(success=False, errors=[f"No elements found for XPath: {self.selector}"])

            # Process results
            values = []
            for result in results:
                if isinstance(result, str):
                    values.append(result)
                elif hasattr(result, "text"):
                    values.append(result.text)
                elif self.attribute and hasattr(result, "get"):
                    values.append(result.get(self.attribute))

            value = values[0] if len(values) == 1 else values

            return ExtractionResult(value=value, success=True, strategy_used="XPath", quality_score=0.95)
        except Exception as e:
            return ExtractionResult(success=False, errors=[f"XPath extraction failed: {str(e)}"])


class TextStrategy(ExtractionStrategy):
    """Text-based extraction strategy using string matching."""

    def __init__(self, pattern: str, context: Optional[str] = None):
        """Initialize text strategy.

        Args:
            pattern: Text pattern to search for
            context: Optional context to narrow search
        """
        super().__init__(pattern)
        self.context = context

    def extract(self, soup: BeautifulSoup) -> ExtractionResult:
        """Extract using text search."""
        try:
            # Search in specific context if provided
            search_area = soup
            if self.context:
                context_elements = soup.select(self.context) or soup.find_all(class_=self.context)
                if context_elements:
                    search_area = context_elements[0]

            # Find text containing pattern
            text_elements = search_area.find_all(string=re.compile(self.selector, re.IGNORECASE))

            if not text_elements:
                return ExtractionResult(success=False, errors=[f"Text pattern not found: {self.selector}"])

            # Get parent elements for context
            values = []
            for text in text_elements:
                parent = text.parent
                if parent:
                    values.append(parent.get_text(strip=True))
                else:
                    values.append(str(text).strip())

            value = values[0] if len(values) == 1 else values

            return ExtractionResult(
                value=value,
                success=True,
                strategy_used="Text",
                quality_score=0.7,
                partial=True if len(values) > 1 else False,
            )
        except Exception as e:
            return ExtractionResult(success=False, errors=[f"Text extraction failed: {str(e)}"])


class RegexStrategy(ExtractionStrategy):
    """Regular expression-based extraction strategy."""

    def __init__(self, pattern: str, group: int = 0):
        """Initialize regex strategy.

        Args:
            pattern: Regular expression pattern
            group: Group number to extract (0 for full match)
        """
        super().__init__(pattern)
        self.group = group
        self.compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def extract(self, soup: BeautifulSoup) -> ExtractionResult:
        """Extract using regular expression."""
        try:
            text = soup.get_text()
            matches = self.compiled_pattern.findall(text)

            if not matches:
                return ExtractionResult(success=False, errors=[f"No regex matches for pattern: {self.selector}"])

            # Extract specific group if requested
            if self.group > 0 and isinstance(matches[0], tuple):
                values = [m[self.group - 1] if len(m) >= self.group else m for m in matches]
            else:
                values = matches

            value = values[0] if len(values) == 1 else values

            return ExtractionResult(value=value, success=True, strategy_used="Regex", quality_score=0.6, partial=True)
        except Exception as e:
            return ExtractionResult(success=False, errors=[f"Regex extraction failed: {str(e)}"])


class ResilientExtractor:
    """Resilient data extractor with fallback strategies."""

    def __init__(self, default_quality_threshold: float = 0.5):
        """Initialize resilient extractor.

        Args:
            default_quality_threshold: Minimum quality score to accept extraction
        """
        self.quality_threshold = default_quality_threshold

    def extract_with_fallback(
        self, soup: BeautifulSoup, strategies: List[ExtractionStrategy], field_name: Optional[str] = None
    ) -> ExtractedData:
        """Extract data using multiple strategies with fallback.

        Args:
            soup: BeautifulSoup object to extract from
            strategies: List of extraction strategies to try
            field_name: Optional field name for logging

        Returns:
            ExtractedData with best extraction result
        """
        extracted_data = ExtractedData()
        best_result = None
        all_errors = []

        for strategy in strategies:
            result = strategy.extract(soup)

            if result.success:
                # Use first successful extraction or highest quality one
                if best_result is None or result.quality_score > best_result.quality_score:
                    best_result = result

                # If we have high quality extraction, stop trying
                if result.quality_score >= 0.95:
                    break
            else:
                all_errors.extend(result.errors)

        # Process best result
        if best_result and best_result.quality_score >= self.quality_threshold:
            if field_name:
                extracted_data.data[field_name] = best_result.value
                extracted_data.strategies_used[field_name] = best_result.strategy_used or "Unknown"
            else:
                extracted_data.data = {"value": best_result.value}
                extracted_data.strategies_used = {"value": best_result.strategy_used or "Unknown"}

            extracted_data.quality_score = best_result.quality_score
            extracted_data.partial_extraction = best_result.partial
        else:
            # No acceptable extraction
            if field_name:
                extracted_data.missing_fields.append(field_name)
            extracted_data.extraction_errors = all_errors
            extracted_data.partial_extraction = True

            logger.warning(f"Failed to extract {field_name or 'data'}: {all_errors}")

        return extracted_data

    def extract_multiple_fields(
        self, soup: BeautifulSoup, field_strategies: Dict[str, List[ExtractionStrategy]]
    ) -> ExtractedData:
        """Extract multiple fields with their respective strategies.

        Args:
            soup: BeautifulSoup object to extract from
            field_strategies: Dictionary mapping field names to strategy lists

        Returns:
            ExtractedData with all extracted fields
        """
        combined_data = ExtractedData()
        total_quality = 0.0
        field_count = 0

        for field_name, strategies in field_strategies.items():
            field_data = self.extract_with_fallback(soup, strategies, field_name)

            # Merge results
            if field_name in field_data.data:
                combined_data.data[field_name] = field_data.data[field_name]
                combined_data.strategies_used[field_name] = field_data.strategies_used.get(field_name, "Unknown")
                total_quality += field_data.quality_score
                field_count += 1
            else:
                combined_data.missing_fields.append(field_name)
                combined_data.partial_extraction = True

            combined_data.extraction_errors.extend(field_data.extraction_errors)

        # Calculate overall quality score
        if field_count > 0:
            combined_data.quality_score = total_quality / field_count
        else:
            combined_data.quality_score = 0.0

        # Determine if extraction is partial
        if combined_data.missing_fields:
            combined_data.partial_extraction = True
            logger.info(f"Partial extraction completed. Missing fields: {combined_data.missing_fields}")

        return combined_data

    def calculate_quality_score(self, data: ExtractedData) -> float:
        """Calculate quality score for extracted data.

        Args:
            data: ExtractedData to score

        Returns:
            Quality score between 0 and 1
        """
        if not data.data:
            return 0.0

        # Base score from extraction quality
        base_score = data.quality_score

        # Penalties
        missing_penalty = len(data.missing_fields) * 0.1
        error_penalty = min(len(data.extraction_errors) * 0.05, 0.3)
        partial_penalty = 0.2 if data.partial_extraction else 0.0

        # Calculate final score
        final_score = max(0.0, base_score - missing_penalty - error_penalty - partial_penalty)

        return min(1.0, final_score)

    def create_default_strategies(self, field_type: str) -> List[ExtractionStrategy]:
        """Create default extraction strategies for common field types.

        Args:
            field_type: Type of field (e.g., 'title', 'date', 'link')

        Returns:
            List of extraction strategies
        """
        strategies = []

        if field_type == "title":
            strategies = [
                CSSStrategy("h1"),
                CSSStrategy(".title"),
                CSSStrategy("[class*='title']"),
                XPathStrategy("//h1/text()"),
                TextStrategy(r"^[A-Z].*"),
            ]
        elif field_type == "date":
            strategies = [
                CSSStrategy(".date"),
                CSSStrategy("[class*='date']"),
                CSSStrategy("time"),
                XPathStrategy("//time/@datetime"),
                RegexStrategy(r"\d{4}-\d{2}-\d{2}"),
            ]
        elif field_type == "link":
            strategies = [
                CSSStrategy("a", attribute="href"),
                XPathStrategy("//a/@href"),
                RegexStrategy(r"https?://[^\s]+"),
            ]
        elif field_type == "text":
            strategies = [CSSStrategy("p"), CSSStrategy(".content"), XPathStrategy("//p/text()")]
        else:
            # Generic strategies
            strategies = [
                CSSStrategy(f".{field_type}"),
                CSSStrategy(f"[class*='{field_type}']"),
                TextStrategy(field_type),
            ]

        return strategies
