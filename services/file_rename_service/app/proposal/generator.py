"""Rename proposal generator integrating ML predictions with template system."""

import logging
import re
import string
from typing import Any

from services.file_rename_service.app.ml.predictor import Predictor
from services.file_rename_service.app.tokenizer.tokenizer import Tokenizer

from .models import NamingTemplate, RenameProposal

logger = logging.getLogger(__name__)


class ProposalGenerator:
    """Generate rename proposals using ML predictions and custom templates."""

    def __init__(
        self,
        predictor: Predictor | None = None,
        tokenizer: Tokenizer | None = None,
        model_dir: str = "models/",
    ):
        """Initialize the proposal generator.

        Args:
            predictor: ML predictor instance (will create if None)
            tokenizer: Tokenizer instance (will create if None)
            model_dir: Directory containing ML models
        """
        self.predictor = predictor or Predictor(model_dir)
        self.tokenizer = tokenizer or Tokenizer()

        # Template variable extractors
        self._variable_extractors = {
            "artist": self._extract_artist,
            "date": self._extract_date,
            "venue": self._extract_venue,
            "quality": self._extract_quality,
            "format": self._extract_format,
            "source": self._extract_source,
            "track": self._extract_track,
            "set": self._extract_set,
            "tour": self._extract_tour,
            "label": self._extract_label,
            "catalog": self._extract_catalog,
        }

    async def generate_proposal(
        self,
        filename: str,
        ml_predictions: dict[str, Any] | None = None,
        templates: list[NamingTemplate] | None = None,
    ) -> RenameProposal:
        """Generate rename proposal for a filename.

        Args:
            filename: Original filename to generate proposal for
            ml_predictions: Pre-computed ML predictions (will generate if None)
            templates: Custom naming templates to consider

        Returns:
            Complete rename proposal with alternatives
        """
        try:
            # Get tokenization for feature extraction
            tokenized = self.tokenizer.tokenize(filename)

            # Get ML predictions if not provided
            if ml_predictions is None:
                token_dicts = [
                    {
                        "value": token.value,
                        "category": token.category.value,
                        "confidence": token.confidence,
                        "original_text": token.original_text,
                    }
                    for token in tokenized.tokens
                ]
                ml_predictions = self.predictor.predict(
                    filename=filename,
                    tokens=token_dicts,
                    return_probabilities=True,
                    top_k=3,
                )

            # Extract top ML prediction as base proposal
            top_prediction = ml_predictions["predictions"][0]
            base_proposal = top_prediction["suggested_name"]
            base_confidence = top_prediction["confidence"]

            # Create context from tokenized filename for template application
            context = self._create_template_context(tokenized)

            # Generate template-based alternatives if templates provided
            template_alternatives: list[str] = []
            template_patterns: list[str] = []
            if templates:
                template_alternatives, template_patterns = await self._generate_template_alternatives(
                    filename, templates, context
                )

            # Generate ML-based alternatives
            ml_alternatives = [pred["suggested_name"] for pred in ml_predictions["predictions"][1:]]

            # Combine and deduplicate alternatives
            all_alternatives = list(dict.fromkeys(template_alternatives + ml_alternatives))

            # Generate additional pattern-based alternatives
            pattern_alternatives = await self._generate_pattern_alternatives(filename, context, count=2)
            all_alternatives.extend(alt for alt in pattern_alternatives if alt not in all_alternatives)

            # Limit to top alternatives
            final_alternatives = all_alternatives[:5]

            # Create explanation
            explanation = self._create_explanation(
                tokenized, ml_predictions, template_patterns, base_proposal, base_confidence
            )

            # Determine patterns used
            patterns_used = [*template_patterns, f"ml_model_{ml_predictions.get('model_version', 'unknown')}"]

            return RenameProposal(
                original_filename=filename,
                proposed_filename=base_proposal,
                confidence_score=base_confidence,
                explanation=explanation,
                patterns_used=patterns_used,
                alternatives=final_alternatives,
            )

        except Exception as e:
            logger.error(f"Error generating proposal for '{filename}': {e}")
            # Return a fallback proposal
            return RenameProposal(
                original_filename=filename,
                proposed_filename=filename,  # Keep original as fallback
                confidence_score=0.1,
                explanation="Error occurred during proposal generation. Please try again or rename manually.",
                patterns_used=["fallback"],
                alternatives=[],
            )

    def apply_template(self, filename: str, template: NamingTemplate, context: dict[str, str]) -> str:
        """Apply a naming template to generate a filename.

        Args:
            filename: Original filename
            template: Naming template to apply
            context: Context variables extracted from filename

        Returns:
            Generated filename using template
        """
        try:
            # Use string.Template for safe variable substitution
            template_obj = string.Template(template.pattern)

            # Apply template with available context
            result = template_obj.safe_substitute(**context)

            # Clean up any remaining placeholders that couldn't be filled
            result = re.sub(r"\$\{[^}]+\}", "", result)  # Remove ${var} style
            result = re.sub(r"\$\w+", "", result)  # Remove $var style

            # Clean up extra whitespace and separators
            result = re.sub(r"[-\s]+", " ", result).strip()
            result = re.sub(r"\s*-\s*$", "", result)  # Remove trailing dash
            result = re.sub(r"^\s*-\s*", "", result)  # Remove leading dash

            return result if result.strip() else filename

        except Exception as e:
            logger.warning(f"Error applying template '{template.name}': {e}")
            return filename

    async def generate_alternatives(self, filename: str, base_proposal: str, count: int = 3) -> list[str]:
        """Generate alternative rename proposals.

        Args:
            filename: Original filename
            base_proposal: Base proposal to generate alternatives from
            count: Number of alternatives to generate

        Returns:
            List of alternative filename proposals
        """
        alternatives: list[str] = []

        try:
            # Get tokenization for context
            tokenized = self.tokenizer.tokenize(filename)
            context = self._create_template_context(tokenized)

            # Generate pattern-based alternatives
            pattern_alts = await self._generate_pattern_alternatives(filename, context, count)
            alternatives.extend(pattern_alts)

            # Generate variations of base proposal
            variations = self._generate_variations(base_proposal, context)
            alternatives.extend(variations)

            # Remove duplicates and limit count
            unique_alternatives = list(dict.fromkeys(alternatives))
            return unique_alternatives[:count]

        except Exception as e:
            logger.error(f"Error generating alternatives for '{filename}': {e}")
            return []

    def _create_template_context(self, tokenized: Any) -> dict[str, str]:
        """Create template context from tokenized filename.

        Args:
            tokenized: TokenizedFilename object

        Returns:
            Context dictionary for template substitution
        """
        context = {}

        # Extract values for each token category
        for token in tokenized.tokens:
            category_name = token.category.value.lower()
            if category_name in self._variable_extractors and category_name not in context:
                # Use the first occurrence of each category
                context[category_name] = token.value

        # Add computed fields
        context["filename"] = tokenized.original_filename
        context["basename"] = (
            tokenized.original_filename.rsplit(".", 1)[0]
            if "." in tokenized.original_filename
            else tokenized.original_filename
        )

        return context

    async def _generate_template_alternatives(
        self, filename: str, templates: list[NamingTemplate], context: dict[str, str]
    ) -> tuple[list[str], list[str]]:
        """Generate alternatives using provided templates.

        Args:
            filename: Original filename
            templates: List of naming templates
            context: Template context variables

        Returns:
            Tuple of (alternatives list, pattern names used)
        """
        alternatives = []
        patterns_used = []

        for template in templates:
            if not template.is_active:
                continue

            try:
                result = self.apply_template(filename, template, context)
                if result != filename and result not in alternatives:
                    alternatives.append(result)
                    patterns_used.append(f"template_{template.name}")

            except Exception as e:
                logger.warning(f"Error applying template '{template.name}': {e}")

        return alternatives, patterns_used

    async def _generate_pattern_alternatives(self, filename: str, context: dict[str, str], count: int) -> list[str]:
        """Generate alternatives using common patterns.

        Args:
            filename: Original filename
            context: Extracted context variables
            count: Maximum number of alternatives to generate

        Returns:
            List of pattern-based alternatives
        """
        alternatives: list[str] = []

        # Common patterns for concert recordings
        patterns = [
            "{artist} - {date} - {venue}",
            "{artist} - {venue} - {date}",
            "{date} - {artist} - {venue}",
            "{artist} ({date}) {venue}",
            "{artist} - {date} ({venue})",
        ]

        for pattern in patterns:
            if len(alternatives) >= count:
                break

            try:
                template_obj = string.Template(pattern)
                result = template_obj.safe_substitute(**context)

                # Clean up incomplete substitutions
                if "$" not in result and result.strip():
                    result = re.sub(r"[-\s]+", " ", result).strip()
                    result = re.sub(r"\s*-\s*$", "", result)

                    if result and result != filename and result not in alternatives:
                        alternatives.append(result)

            except Exception as e:
                logger.debug(f"Error with pattern '{pattern}': {e}")

        return alternatives

    def _generate_variations(self, base_proposal: str, context: dict[str, str]) -> list[str]:
        """Generate variations of the base proposal.

        Args:
            base_proposal: Base filename proposal
            context: Context variables

        Returns:
            List of filename variations
        """
        variations = []

        try:
            # Add quality/format suffixes if available
            if "quality" in context:
                quality_var = f" [{context['quality']}]"
                if quality_var not in base_proposal:
                    variations.append(f"{base_proposal}{quality_var}")

            if "format" in context:
                format_var = f" ({context['format']})"
                if format_var not in base_proposal:
                    variations.append(f"{base_proposal}{format_var}")

            # Date format variations
            if "date" in context:
                date_val = context["date"]
                # Try different date positions
                if date_val not in base_proposal:
                    variations.append(f"{date_val} - {base_proposal}")
                    variations.append(f"{base_proposal} - {date_val}")

        except Exception as e:
            logger.debug(f"Error generating variations: {e}")

        return variations

    def _create_explanation(
        self,
        tokenized: Any,
        ml_predictions: dict[str, Any],
        template_patterns: list[str],
        proposal: str,
        confidence: float,
    ) -> str:
        """Create human-readable explanation for the proposal.

        Args:
            tokenized: TokenizedFilename object
            ml_predictions: ML prediction results
            template_patterns: Template patterns used
            proposal: Final proposal
            confidence: Confidence score

        Returns:
            Explanation string
        """
        try:
            explanation_parts = []

            # Base explanation
            explanation_parts.append(f"Based on analysis of '{tokenized.original_filename}'")

            # Token information
            if tokenized.tokens:
                token_categories = [token.category.value.lower() for token in tokenized.tokens]
                unique_categories = list(dict.fromkeys(token_categories))
                explanation_parts.append(f"identified {', '.join(unique_categories)}")

            # ML model contribution
            model_version = ml_predictions.get("model_version", "unknown")
            explanation_parts.append(f"using ML model {model_version}")

            # Template usage
            if template_patterns:
                explanation_parts.append(f"and custom templates ({len(template_patterns)} applied)")

            # Confidence interpretation
            if confidence >= 0.8:
                conf_desc = "high confidence"
            elif confidence >= 0.6:
                conf_desc = "moderate confidence"
            else:
                conf_desc = "lower confidence"

            explanation_parts.append(f"with {conf_desc} ({confidence:.2f})")

            return ", ".join(explanation_parts) + "."

        except Exception as e:
            logger.warning(f"Error creating explanation: {e}")
            return f"Rename proposal generated with {confidence:.2f} confidence."

    # Token extraction methods for template variables
    def _extract_artist(self, tokens: list[Any]) -> str | None:
        """Extract artist name from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "artist":
                return str(token.value)
        return None

    def _extract_date(self, tokens: list[Any]) -> str | None:
        """Extract date from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "date":
                return str(token.value)
        return None

    def _extract_venue(self, tokens: list[Any]) -> str | None:
        """Extract venue from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "venue":
                return str(token.value)
        return None

    def _extract_quality(self, tokens: list[Any]) -> str | None:
        """Extract quality from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "quality":
                return str(token.value)
        return None

    def _extract_format(self, tokens: list[Any]) -> str | None:
        """Extract format from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "format":
                return str(token.value)
        return None

    def _extract_source(self, tokens: list[Any]) -> str | None:
        """Extract source from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "source":
                return str(token.value)
        return None

    def _extract_track(self, tokens: list[Any]) -> str | None:
        """Extract track from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "track":
                return str(token.value)
        return None

    def _extract_set(self, tokens: list[Any]) -> str | None:
        """Extract set from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "set":
                return str(token.value)
        return None

    def _extract_tour(self, tokens: list[Any]) -> str | None:
        """Extract tour from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "tour":
                return str(token.value)
        return None

    def _extract_label(self, tokens: list[Any]) -> str | None:
        """Extract label from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "label":
                return str(token.value)
        return None

    def _extract_catalog(self, tokens: list[Any]) -> str | None:
        """Extract catalog from tokens."""
        for token in tokens:
            if hasattr(token, "category") and token.category.value.lower() == "catalog":
                return str(token.value)
        return None
