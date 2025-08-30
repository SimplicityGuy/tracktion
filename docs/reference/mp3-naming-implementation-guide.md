# MP3 Naming Convention Implementation Guide

## Quick Start

This guide provides practical implementation details for the MP3 Release Rules 4.1 naming conventions in the Tracktion project.

## Core Implementation Components

### 1. Naming Pattern Engine

```python
class MP3NamingEngine:
    """Core engine for MP3 naming convention compliance."""

    def generate_dirname(
        self,
        artist: str,
        title: str,
        year: int,
        source: str,
        group: str = "TRACKTION",
        additional_tags: List[str] = None
    ) -> str:
        """
        Generate compliant directory name.

        Pattern: Artist-Title-[Tags]-Source-Year-Group
        """
        pass

    def generate_filename(
        self,
        track_num: int,
        artist: str,
        title: str,
        is_va: bool = False
    ) -> str:
        """
        Generate compliant filename.

        Pattern: TrackNum-Artist-Title.mp3
        """
        pass
```

### 2. Character Sanitization

```python
class CharacterSanitizer:
    """Handle character restrictions and replacements."""

    VALID_CHARS = set('abcdefghijklmnopqrstuvwxyz'
                      'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                      '0123456789'
                      '_.-() ')

    REPLACEMENTS = {
        ' ': '_',      # Space to underscore
        '&': '_and_',  # Ampersand
        '@': '_at_',   # At symbol
        '#': '_',      # Hash
        '$': '_',      # Dollar
        '%': '_',      # Percent
        '^': '_',      # Caret
        '*': '_',      # Asterisk
        '+': '_',      # Plus
        '=': '_',      # Equals
        '{': '(',      # Left brace
        '}': ')',      # Right brace
        '[': '(',      # Left bracket
        ']': ')',      # Right bracket
        '|': '_',      # Pipe
        '\\': '_',     # Backslash
        '/': '_',      # Forward slash
        '<': '_',      # Less than
        '>': '_',      # Greater than
        '?': '_',      # Question mark
        ':': '_',      # Colon
        ';': '_',      # Semicolon
        '"': '_',      # Quote
        "'": '_',      # Apostrophe
        ',': '_',      # Comma
        '!': '_',      # Exclamation
        '~': '_',      # Tilde
        '`': '_',      # Backtick
    }
```

### 3. Source Detection

```python
class SourceDetector:
    """Detect and validate source types."""

    PHYSICAL_SOURCES = {
        'CD': ['CD', 'CDA', 'CDEP'],
        'CDS': ['CDS', 'CDM'],
        'VINYL': ['VINYL', 'VLS', 'LP'],
        'DVD': ['DVD', 'DVDS'],
        'TAPE': ['TAPE'],
        'FLASH': ['FLASH'],
    }

    DIGITAL_SOURCES = {
        'WEB': ['WEB'],
        'MP3CD': ['MP3CD'],
    }

    LIVE_SOURCES = {
        'FM': ['FM'],
        'SAT': ['SAT'],
        'CABLE': ['CABLE'],
        'DAB': ['DAB'],
        'SBD': ['SBD'],
        'AUD': ['AUD'],
        'LINE': ['LINE'],
    }

    def detect_source(self, metadata: Dict) -> str:
        """Detect source type from metadata."""
        pass

    def validate_source_tag(self, tag: str) -> bool:
        """Validate if source tag is compliant."""
        pass
```

### 4. Special Cases Handler

```python
class SpecialCasesHandler:
    """Handle VA, OST, bootlegs, and other special cases."""

    def handle_va_release(self, tracks: List[Track]) -> str:
        """Handle Various Artists releases."""
        # Use "VA" as artist in dirname
        # Include individual artists in filenames
        pass

    def handle_multi_disc(self, disc_count: int, source: str) -> str:
        """Generate multi-disc notation."""
        # -2CD-, -3DVD-, etc.
        pass

    def handle_web_single(self, track_count: int, duration: int) -> str:
        """Determine if -SINGLE- tag needed."""
        # Single track, under 25 minutes
        pass

    def handle_ost(self, is_ost: bool) -> str:
        """Add -OST- tag after title if needed."""
        pass

    def handle_bootleg(self, is_bootleg: bool) -> str:
        """Add BOOTLEG to dirname if needed."""
        pass
```

### 5. Length Compliance Manager

```python
class LengthComplianceManager:
    """Ensure 255 character limit compliance."""

    MAX_PATH_LENGTH = 255

    def check_length(self, dirname: str, filename: str) -> bool:
        """Check if combined path exceeds limit."""
        return len(dirname) + len(filename) + 1 <= self.MAX_PATH_LENGTH

    def truncate_intelligently(
        self,
        dirname: str,
        filename: str
    ) -> Tuple[str, str]:
        """
        Truncate while preserving essential information.

        Priority:
        1. Keep track number
        2. Keep source tag
        3. Keep year
        4. Truncate title before artist
        5. Truncate from right, preserve left
        """
        pass
```

## Integration Points

### With renaming_service.py

```python
# In services/renaming_service.py

from mp3_naming import MP3NamingEngine

class RenamingService:
    def __init__(self):
        self.mp3_engine = MP3NamingEngine()

    def rename_file(self, file_info: FileInfo) -> str:
        """Rename file according to MP3 conventions."""
        if self.is_mp3_file(file_info):
            return self.mp3_engine.generate_filename(
                track_num=file_info.track_number,
                artist=file_info.artist,
                title=file_info.title,
                is_va=file_info.is_various_artists
            )
```

### With media_discovery_service.py

```python
# In services/media_discovery_service.py

class MediaDiscoveryService:
    def organize_release(self, release: Release) -> str:
        """Organize release according to MP3 conventions."""
        dirname = self.mp3_engine.generate_dirname(
            artist=release.artist,
            title=release.title,
            year=release.year,
            source=self.detect_source(release),
            additional_tags=self.get_additional_tags(release)
        )
        return dirname
```

## Common Patterns

### Standard Album
```python
def format_standard_album(album: Album) -> str:
    """Format standard album release."""
    return f"{album.artist}-{album.title}-CD-{album.year}-GROUP"
```

### Various Artists Compilation
```python
def format_va_compilation(compilation: Compilation) -> str:
    """Format VA compilation."""
    return f"VA-{compilation.title}-2CD-{compilation.year}-GROUP"
```

### Web Single
```python
def format_web_single(single: Single) -> str:
    """Format web single release."""
    return f"{single.artist}-{single.title}-SINGLE-WEB-{single.year}-GROUP"
```

### Live Recording
```python
def format_live_recording(live: LiveRecording) -> str:
    """Format live recording."""
    source = live.source  # FM, SAT, SBD, etc.
    return f"{live.artist}-{live.venue}-{source}-{live.year}-GROUP"
```

## Validation Checklist

```python
class MP3NamingValidator:
    """Validate naming convention compliance."""

    def validate_release(self, release_path: str) -> ValidationResult:
        """
        Complete validation of release naming.

        Checks:
        - [ ] Dirname format compliance
        - [ ] Filename format compliance
        - [ ] Character restrictions
        - [ ] Length limits
        - [ ] Source tag validity
        - [ ] Special case handling
        - [ ] Multi-disc notation
        - [ ] VA formatting
        """
        pass
```

## Edge Cases

### 1. Long Artist/Title Names
```python
# Original: "The Amazing Super Long Artist Name Featuring Many People"
# Truncated: "The_Amazing_Super_Long_Artist_Name_Feat"
```

### 2. Foreign Characters
```python
# Original: "Björk - Homogénic"
# Converted: "Bjork-Homogenic"
```

### 3. Multiple Source Formats
```python
# Boxset with CD+Vinyl: Release separately
# "Artist-Title-CD-2021-GROUP"
# "Artist-Title-VINYL-2021-GROUP"
```

### 4. Bootleg VA Compilation
```python
# "VA-Underground_Mix-BOOTLEG-2021-GROUP"
```

## Testing Strategy

### Unit Tests
```python
def test_dirname_generation():
    """Test directory name generation."""
    assert generate_dirname(
        "Artist", "Title", 2021, "CD", "GROUP"
    ) == "Artist-Title-CD-2021-GROUP"

def test_character_sanitization():
    """Test invalid character replacement."""
    assert sanitize("Artist & Friend") == "Artist_and_Friend"

def test_length_compliance():
    """Test 255 character limit."""
    long_name = "A" * 300
    result = truncate_intelligently(long_name, "01-track.mp3")
    assert len(result[0]) + len(result[1]) + 1 <= 255
```

### Integration Tests
```python
def test_full_release_naming():
    """Test complete release naming workflow."""
    release = create_test_release()
    result = naming_engine.process_release(release)
    assert validate_mp3_compliance(result)
```

## Migration Strategy

### Phase 1: Analysis
1. Scan existing library for non-compliant names
2. Generate compliance report
3. Identify patterns needing correction

### Phase 2: Implementation
1. Deploy naming engine with validation mode
2. Test on sample releases
3. Gather feedback and adjust

### Phase 3: Migration
1. Create backup of existing names
2. Generate rename mapping
3. Execute batch rename with rollback capability

### Phase 4: Enforcement
1. Enable strict mode for new releases
2. Validate all incoming files
3. Auto-correct where possible

## Configuration

```yaml
# config/mp3_naming.yaml
mp3_naming:
  enabled: true
  strict_mode: false
  group_tag: "TRACKTION"

  character_handling:
    replace_spaces: true
    remove_accents: true

  length_limits:
    max_path: 255
    truncate_title_first: true

  source_detection:
    auto_detect: true
    default_source: "WEB"

  special_cases:
    detect_va: true
    detect_ost: true
    detect_bootleg: false

  validation:
    enforce_on_write: true
    warn_on_read: true
```

## API Reference

### Main Functions

```python
def generate_dirname(artist, title, year, source, group="TRACKTION", tags=None):
    """Generate MP3-compliant directory name."""

def generate_filename(track_num, artist, title, is_va=False):
    """Generate MP3-compliant filename."""

def sanitize_characters(text):
    """Replace invalid characters."""

def validate_naming(path):
    """Validate MP3 naming compliance."""

def migrate_library(source_dir, target_dir, dry_run=True):
    """Migrate existing library to MP3 conventions."""
```

## Troubleshooting

### Common Issues

1. **Character Encoding Errors**
   - Ensure UTF-8 encoding throughout
   - Use unicodedata for accent removal

2. **Path Too Long**
   - Enable intelligent truncation
   - Consider abbreviations for common terms

3. **Source Detection Failures**
   - Provide manual override option
   - Log uncertain detections for review

4. **VA Detection**
   - Check for multiple artists in tracks
   - Look for "Various", "V.A.", "Compilation" keywords

5. **Multi-Disc Handling**
   - Detect from metadata disc_number field
   - Check for "CD1", "Disc 1" patterns in existing names
