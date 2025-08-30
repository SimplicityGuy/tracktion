"""
Pattern definitions and matching logic for filename tokenization.
"""

import re

from .models import Pattern, Token, TokenCategory


class PatternMatcher:
    """Manages regex patterns for tokenizing filenames."""

    def __init__(self) -> None:
        """Initialize the pattern matcher with default patterns."""
        self.patterns = self._initialize_patterns()
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _initialize_patterns(self) -> list[Pattern]:
        """Initialize default regex patterns for common filename formats."""
        return [
            # Date patterns (highest priority)
            Pattern(
                regex=r"\b(?:19|20)\d{2}[-._](?:0[1-9]|1[0-2])[-._](?:0[1-9]|[12]\d|3[01])\b",
                category=TokenCategory.DATE,
                priority=100,
                description="ISO date format (YYYY-MM-DD)",
                examples=["2024-01-29", "2023_12_31", "1999.07.04"],
            ),
            Pattern(
                regex=r"\b(?:0[1-9]|1[0-2])[-._](?:0[1-9]|[12]\d|3[01])[-._](?:19|20)?\d{2}\b",
                category=TokenCategory.DATE,
                priority=99,
                description="US date format (MM-DD-YYYY or MM-DD-YY)",
                examples=["01-29-2024", "12_31_23", "07.04.99"],
            ),
            Pattern(
                regex=r"\b(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b",
                category=TokenCategory.DATE,
                priority=98,
                description="Compact date format (YYYYMMDD)",
                examples=["20240129", "19990704"],
            ),
            # Quality indicators
            Pattern(
                regex=r"\b(FLAC|flac|MP3|mp3|WAV|wav|APE|ape|SHN|shn|M4A|m4a)\b",
                category=TokenCategory.FORMAT,
                priority=90,
                description="Audio format indicators",
                examples=["FLAC", "mp3", "WAV"],
            ),
            Pattern(
                regex=r"\b(320kbps|256kbps|192kbps|128kbps|V0|V2|LOSSLESS|lossless|24bit|16bit|96kHz|48kHz|44\.1kHz)\b",
                category=TokenCategory.QUALITY,
                priority=89,
                description="Audio quality indicators",
                examples=["320kbps", "V0", "LOSSLESS", "24bit"],
            ),
            Pattern(
                regex=r"\b(SBD|sbd|AUD|aud|MTX|mtx|FM|fm|WEB|web|STREAM|stream)\b",
                category=TokenCategory.SOURCE,
                priority=88,
                description="Recording source indicators",
                examples=["SBD", "AUD", "FM"],
            ),
            # Set and track patterns
            Pattern(
                regex=r"\b(?:Set|SET|set)[-_\s]?(?:[1-3]|I{1,3}|One|Two|Three)\b",
                category=TokenCategory.SET,
                priority=85,
                description="Set indicators",
                examples=["Set1", "SET_2", "Set III"],
            ),
            Pattern(
                regex=r"\b(?:CD|cd|Disc|DISC|disc)[-_\s]?(?:\d{1,2})\b",
                category=TokenCategory.SET,
                priority=84,
                description="Disc/CD indicators",
                examples=["CD1", "Disc_2", "disc 3"],
            ),
            Pattern(
                regex=r"\b(?:Track|TRACK|track|T|t)[-_\s]?(?:\d{1,3})\b",
                category=TokenCategory.TRACK,
                priority=83,
                description="Track number indicators",
                examples=["Track01", "T12", "track_05"],
            ),
            Pattern(
                regex=r"^(?:\d{1,3})[-_\s]",
                category=TokenCategory.TRACK,
                priority=82,
                description="Leading track numbers",
                examples=["01 - ", "12_", "5 "],
            ),
            # Venue patterns
            Pattern(
                regex=r"@\s*([^@\[\]\(\)]+?)(?:\s*[-,]\s*|$)",
                category=TokenCategory.VENUE,
                priority=75,
                description="Venue with @ symbol",
                examples=["@Madison Square Garden", "@The Fillmore"],
            ),
            Pattern(
                regex=r"\b(?:at|At|AT)\s+([A-Z][^,\[\]\(\)]{2,}?)(?:\s*[-,]\s*|$)",
                category=TokenCategory.VENUE,
                priority=74,
                description="Venue with 'at' keyword",
                examples=["at Madison Square Garden", "At The Forum"],
            ),
            Pattern(
                regex=r"\b(Madison Square Garden|BartonHall|Barton Hall|Free Trade Hall)\b",
                category=TokenCategory.VENUE,
                priority=95,
                description="Known venue names",
                examples=["Madison Square Garden", "BartonHall"],
            ),
            Pattern(
                regex=r"\b[A-Z]{2,3}\b(?:\s+[A-Z]{2})?$",  # NYC, LA, SF or City ST
                category=TokenCategory.VENUE,
                priority=50,
                description="City/State codes",
                examples=["NYC", "LA", "City ST"],
            ),
            # Tour patterns
            Pattern(
                regex=r"\b(\d{4})\s+(Spring|Summer|Fall|Winter|Tour|tour)\b",
                category=TokenCategory.TOUR,
                priority=70,
                description="Tour indicators",
                examples=["2023 Summer", "1999 Tour"],
            ),
            # Label and catalog patterns
            Pattern(
                regex=r"\[([A-Z]{2,}[_\-]?\d{3,})\]",
                category=TokenCategory.CATALOG,
                priority=65,
                description="Catalog numbers in brackets",
                examples=["[CAT001]", "[LABEL_123]"],
            ),
            # Artist patterns (common bands and artist indicators)
            Pattern(
                regex=r"\b(Phish|GratefulDead|Grateful Dead|Radiohead|Dylan|Bob Dylan|Miles Davis|Beatles|Pink Floyd|Led Zeppelin|Artist|BandName)\b",
                category=TokenCategory.ARTIST,
                priority=95,
                description="Known artist names",
                examples=["Phish", "GratefulDead", "Radiohead"],
            ),
            Pattern(
                regex=r"^[A-Z][a-zA-Z]+(?:[A-Z][a-z]+)*(?=\s+\d{4}|\s+19\d{2}|\s+20\d{2})",
                category=TokenCategory.ARTIST,
                priority=60,
                description="Artist name before year",
                examples=["BandName 2023", "Artist 1999"],
            ),
            Pattern(
                regex=r"\b(Intro|Outro|Jam|Encore|Interlude)\b",
                category=TokenCategory.TRACK,
                priority=70,
                description="Track/segment names",
                examples=["Intro", "Outro", "Encore"],
            ),
            # Special characters and delimiters
            Pattern(
                regex=r"[-_]{2,}",
                category=TokenCategory.UNKNOWN,
                priority=10,
                description="Multiple delimiters",
                examples=["---", "___"],
            ),
        ]

    def _compile_patterns(self) -> None:
        """Compile all regex patterns for better performance."""
        for pattern in self.patterns:
            try:
                self._compiled_patterns[pattern.regex] = re.compile(pattern.regex, re.IGNORECASE)
            except re.error as e:
                print(f"Error compiling pattern {pattern.regex}: {e}")

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a new pattern to the matcher."""
        self.patterns.append(pattern)
        try:
            self._compiled_patterns[pattern.regex] = re.compile(pattern.regex, re.IGNORECASE)
        except re.error as e:
            print(f"Error compiling pattern {pattern.regex}: {e}")

    def match(self, text: str) -> list[tuple[Token, int, int]]:
        """
        Match text against all patterns and return tokens with positions.

        Returns list of (Token, start_pos, end_pos) tuples.
        """
        matches = []

        # Sort patterns by priority (highest first)
        sorted_patterns = sorted(self.patterns, key=lambda p: p.priority, reverse=True)

        for pattern in sorted_patterns:
            compiled = self._compiled_patterns.get(pattern.regex)
            if not compiled:
                continue

            for match in compiled.finditer(text):
                # Extract the matched text
                matched_text = match.group(0)

                # For patterns with capturing groups, extract the meaningful part
                # Otherwise use the full match
                if match.groups() and any(g is not None for g in match.groups()):
                    # Get the first non-None captured group
                    value = next((g for g in match.groups() if g is not None), matched_text)
                else:
                    value = matched_text

                # Create token
                token = Token(
                    value=value,
                    category=pattern.category,
                    confidence=0.9,  # Base confidence for regex matches
                    original_text=matched_text,
                    position=match.start(),
                )

                # Increment pattern match count
                pattern.increment_match_count()

                matches.append((token, match.start(), match.end()))

        return matches

    def extract_unmatched(self, text: str, matches: list[tuple[Token, int, int]]) -> list[str]:
        """Extract segments of text that weren't matched by any pattern."""
        if not matches:
            return [text] if text.strip() else []

        # Sort matches by position
        sorted_matches = sorted(matches, key=lambda m: m[1])

        unmatched = []
        last_end = 0

        for _, start, end in sorted_matches:
            if start > last_end:
                segment = text[last_end:start].strip()
                if segment:
                    unmatched.append(segment)
            last_end = max(last_end, end)

        # Check for remaining text after last match
        if last_end < len(text):
            segment = text[last_end:].strip()
            if segment:
                unmatched.append(segment)

        return unmatched

    def get_patterns_by_category(self, category: TokenCategory) -> list[Pattern]:
        """Get all patterns for a specific category."""
        return [p for p in self.patterns if p.category == category]

    def get_pattern_statistics(self) -> dict[str, dict]:
        """Get statistics about pattern usage."""
        stats = {}
        for pattern in self.patterns:
            stats[pattern.regex] = {
                "category": pattern.category.value,
                "priority": pattern.priority,
                "match_count": pattern.match_count,
                "description": pattern.description,
            }
        return stats
