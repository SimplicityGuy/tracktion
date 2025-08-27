"""CUE file validation module.

Provides comprehensive validation of CUE files including syntax,
file references, timing consistency, and format compatibility.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .parser import CueParser
from .models import CueSheet
from .exceptions import CueParsingError
from .validation_rules import (
    ValidationRule,
    SyntaxValidator,
    CommandOrderValidator,
    FileReferenceValidator,
    TimestampValidator,
    CrossReferenceValidator,
    CompatibilityValidator,
    Severity,
    ValidationIssue,
)
from .audio_analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Complete validation result for a CUE file."""

    file_path: str
    is_valid: bool  # True if no ERRORS
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    audio_duration_ms: Optional[float] = None
    cue_duration_ms: Optional[float] = None
    format_compatibility: Dict[str, bool] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the appropriate list based on severity."""
        if issue.severity == Severity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == Severity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append(f"Validation Report for: {self.file_path}")
        lines.append("=" * 60)

        # Summary
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        lines.append(f"Info: {len(self.info)}")

        # Duration comparison
        if self.audio_duration_ms is not None and self.cue_duration_ms is not None:
            audio_sec = self.audio_duration_ms / 1000
            cue_sec = self.cue_duration_ms / 1000
            diff_sec = abs(audio_sec - cue_sec)
            lines.append("\nDuration Analysis:")
            lines.append(f"  Audio: {audio_sec:.2f}s")
            lines.append(f"  CUE: {cue_sec:.2f}s")
            lines.append(f"  Difference: {diff_sec:.2f}s")

        # Format compatibility
        if self.format_compatibility:
            lines.append("\nFormat Compatibility:")
            for fmt, compatible in self.format_compatibility.items():
                status = "✅" if compatible else "❌"
                lines.append(f"  {fmt}: {status}")

        # Issues by severity
        if self.errors:
            lines.append("\n" + "=" * 60)
            lines.append("ERRORS (Must Fix)")
            lines.append("-" * 40)
            for error in self.errors:
                lines.append(self._format_issue(error))

        if self.warnings:
            lines.append("\n" + "=" * 60)
            lines.append("WARNINGS (Should Fix)")
            lines.append("-" * 40)
            for warning in self.warnings:
                lines.append(self._format_issue(warning))

        if self.info:
            lines.append("\n" + "=" * 60)
            lines.append("INFO (Suggestions)")
            lines.append("-" * 40)
            for info_item in self.info:
                lines.append(self._format_issue(info_item))

        return "\n".join(lines)

    def _format_issue(self, issue: ValidationIssue) -> str:
        """Format a single issue for display."""
        lines = []
        line_ref = f"Line {issue.line_number}: " if issue.line_number else ""
        lines.append(f"\n{line_ref}[{issue.category}] {issue.message}")
        if issue.suggestion:
            lines.append(f"  → Suggestion: {issue.suggestion}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Generate JSON report for programmatic use."""
        return {
            "file_path": self.file_path,
            "is_valid": self.is_valid,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info),
            },
            "duration": {
                "audio_ms": self.audio_duration_ms,
                "cue_ms": self.cue_duration_ms,
            },
            "format_compatibility": self.format_compatibility,
            "errors": [self._issue_to_dict(i) for i in self.errors],
            "warnings": [self._issue_to_dict(i) for i in self.warnings],
            "info": [self._issue_to_dict(i) for i in self.info],
        }

    def _issue_to_dict(self, issue: ValidationIssue) -> dict:
        """Convert issue to dictionary."""
        return {
            "severity": issue.severity.value,
            "line_number": issue.line_number,
            "category": issue.category,
            "message": issue.message,
            "suggestion": issue.suggestion,
        }


class CueValidator:
    """Main CUE file validator."""

    def __init__(self) -> None:
        """Initialize validator with all validation rules."""
        self.parser = CueParser()
        self.audio_analyzer = AudioAnalyzer()

        # Initialize validation rules
        self.rules: List[ValidationRule] = [
            SyntaxValidator(),
            CommandOrderValidator(),
            FileReferenceValidator(),
            TimestampValidator(),
            CrossReferenceValidator(),
            CompatibilityValidator(),
        ]

    def validate(self, cue_file_path: str) -> ValidationResult:
        """Validate a single CUE file.

        Args:
            cue_file_path: Path to the CUE file to validate

        Returns:
            ValidationResult containing all issues found
        """
        cue_path = Path(cue_file_path).resolve()
        result = ValidationResult(file_path=str(cue_path), is_valid=True)

        # Check file exists
        if not cue_path.exists():
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    line_number=None,
                    category="file",
                    message=f"CUE file not found: {cue_path}",
                    suggestion="Check the file path and ensure the file exists",
                )
            )
            return result

        # Parse CUE file
        try:
            with open(cue_path, "r", encoding="utf-8-sig") as f:
                cue_content = f.read()
            cue_sheet = self.parser.parse(cue_content)
        except CueParsingError as e:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    line_number=e.line_number if hasattr(e, "line_number") else None,
                    category="syntax",
                    message=f"Parse error: {str(e)}",
                    suggestion="Fix the syntax error and try again",
                )
            )
            return result
        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    line_number=None,
                    category="parse",
                    message=f"Unexpected error parsing CUE: {str(e)}",
                    suggestion="Check the file encoding and format",
                )
            )
            return result

        # Run all validation rules
        for rule in self.rules:
            try:
                rule_issues = rule.validate(cue_sheet, cue_path, cue_content)
                for issue in rule_issues:
                    result.add_issue(issue)
            except Exception as e:
                logger.error(f"Error running validation rule {rule.__class__.__name__}: {e}")
                result.add_issue(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        line_number=None,
                        category="validation",
                        message=f"Could not complete {rule.__class__.__name__} validation: {str(e)}",
                        suggestion=None,
                    )
                )

        # Analyze audio durations
        try:
            audio_duration, cue_duration = self.audio_analyzer.analyze_durations(cue_sheet, cue_path)
            result.audio_duration_ms = audio_duration
            result.cue_duration_ms = cue_duration

            # Check duration mismatch
            if audio_duration and cue_duration:
                diff_ms = abs(audio_duration - cue_duration)
                if diff_ms > 2000:  # More than 2 seconds difference
                    severity = Severity.ERROR if diff_ms > 10000 else Severity.WARNING
                    result.add_issue(
                        ValidationIssue(
                            severity=severity,
                            line_number=None,
                            category="timing",
                            message=f"CUE duration ({cue_duration / 1000:.1f}s) differs from audio duration ({audio_duration / 1000:.1f}s) by {diff_ms / 1000:.1f}s",
                            suggestion="Check track timings and ensure they match the actual audio file",
                        )
                    )
        except Exception as e:
            logger.warning(f"Could not analyze audio durations: {e}")

        # Check format compatibility
        result.format_compatibility = self._check_compatibility(cue_sheet, result)

        return result

    def validate_batch(self, cue_file_paths: List[str]) -> List[ValidationResult]:
        """Validate multiple CUE files.

        Args:
            cue_file_paths: List of paths to CUE files

        Returns:
            List of ValidationResult objects
        """
        results = []
        for path in cue_file_paths:
            try:
                result = self.validate(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error validating {path}: {e}")
                result = ValidationResult(file_path=path, is_valid=False)
                result.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        line_number=None,
                        category="system",
                        message=f"Validation failed: {str(e)}",
                        suggestion="Check the file and try again",
                    )
                )
                results.append(result)

        return results

    def _check_compatibility(self, cue_sheet: CueSheet, result: ValidationResult) -> Dict[str, bool]:
        """Check compatibility with various DJ software.

        Args:
            cue_sheet: Parsed CUE sheet
            result: Current validation result

        Returns:
            Dictionary of format compatibility
        """
        compatibility = {
            "CDJ": True,
            "Traktor": True,
            "Serato": True,
            "Rekordbox": True,
            "Kodi": True,
        }

        # Check for format-specific issues
        for error in result.errors:
            # Any error likely breaks compatibility
            for fmt in compatibility:
                compatibility[fmt] = False

        # Check warnings that might affect specific formats
        for warning in result.warnings:
            if "character limit" in warning.message.lower():
                # CDJ has strict character limits
                compatibility["CDJ"] = False
            if "multi-file" in warning.message.lower():
                # Some formats don't support multi-file CUEs well
                compatibility["Serato"] = False
                compatibility["Traktor"] = False

        return compatibility
