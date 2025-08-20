# Epic 5: Build a Robust CUE File Handler

## Epic Overview
**Epic ID:** EPIC-5
**Epic Name:** Build a Robust CUE File Handler
**Priority:** High
**Dependencies:** Epic 1 (Foundation & Core Services)
**Estimated Effort:** 2-3 weeks

## Business Value
CUE files are essential for DJ workflows, providing track markers and metadata for continuous mixes. A robust CUE file handler will:
- Enable precise track navigation within DJ mixes
- Support both reading and writing CUE files with full spec compliance
- Facilitate mix reconstruction and track identification
- Provide compatibility with all major DJ software (CDJ, Traktor, Serato, Rekordbox)
- Enable automated CUE generation from various sources

## Technical Scope

### Core Requirements
1. **CUE File Parser**
   - Complete CUE sheet specification support per Wikipedia, Hydrogenaudio, and GNU standards
   - Parse all essential commands: FILE, TRACK, INDEX
   - Parse metadata commands: REM, TITLE, PERFORMER, SONGWRITER
   - Parse disc commands: CATALOG (UPC/EAN), CDTEXTFILE
   - Parse track commands: FLAGS, ISRC, PREGAP, POSTGAP
   - Parse extended REM metadata fields:
     * REM GENRE - Music genre classification
     * REM DATE - Release date information
     * REM DISCID - Disc identification number
     * REM COMMENT - General comments and tool signatures
     * REM DISCNUMBER - Multi-disc set numbering (1-15)
     * REM COMPOSER - Composer information
     * REM REPLAYGAIN_ALBUM_GAIN/PEAK - Album-level replay gain
     * REM REPLAYGAIN_TRACK_GAIN/PEAK - Track-level replay gain
   - Support multiple FILE entries (multi-file CUE sheets)
   - Encoding detection and handling (UTF-8, extended ASCII, Latin-1)
   - Support alternative comment syntax (";" and "//" prefixes)
   - Command ordering validation (CATALOG first, FILE before TRACK, etc.)
   - Graceful error handling for malformed files

2. **CUE File Generator**
   - Standards-compliant CUE generation following Wikipedia, Hydrogenaudio, and GNU specs
   - Essential command generation (FILE, TRACK, INDEX)
   - Metadata command support (TITLE, PERFORMER, REM, SONGWRITER)
   - Extended REM metadata generation:
     * Genre, date, disc ID, composer
     * ReplayGain values for normalization
     * Multi-disc numbering
     * Custom application signatures
   - Track flag and gap support (FLAGS, PREGAP, POSTGAP)
   - Multiple output formats (standard, DJ software variants, Kodi-compatible)
   - Frame-accurate timestamp generation (MM:SS:FF format, 75 fps)
   - Command ordering compliance (proper hierarchy and sequence)
   - Custom field support via REM commands

3. **CUE File Editor/Updater**
   - In-place editing capabilities
   - Track addition/removal/reordering
   - INDEX adjustment (including INDEX 00 for pregaps)
   - Metadata updates (TITLE, PERFORMER, SONGWRITER)
   - ISRC and CATALOG code management
   - Format conversion between CUE variants
   - Multi-file CUE sheet handling

4. **Integration Features**
   - Audio file validation and format verification (WAVE, MP3, BINARY)
   - Timestamp verification against audio duration
   - Frame-accurate time calculations (75 frames per second)
   - CD-Text file integration support
   - Batch processing capabilities
   - CUE sheet merging and splitting for multi-file references

### Technical Considerations

#### Specification Sources & Discrepancies
**Reference Standards:**
- Wikipedia CUE sheet specification (primary)
- Hydrogenaudio Knowledgebase specification
- GNU ccd2cue specification
- Kodi implementation notes

**Key Discrepancies to Handle:**
1. **Frame Range**: Wikipedia states 75 frames/second; some specs show 0-74 range (75 total frames)
2. **File Type Mapping**: FLAC files commonly use WAVE type (not in Wikipedia spec)
3. **Comment Syntax**: Alternative comment prefixes (";" and "//") in addition to REM
4. **Command Repetition**: FILE, TRACK, INDEX, and REM can appear multiple times
5. **Character Encoding**: Extended ASCII support in addition to UTF-8
6. **Application-Specific Extensions**: Different software adds proprietary REM fields

#### CUE Command Specifications (Combined Standards)
**Essential Commands:**
- **FILE**: Specifies data file name and format (WAVE, MP3, BINARY, MOTOROLA, AIFF)
- **TRACK**: Defines track number (01-99) and data type (AUDIO, CDG, MODE1/2048, MODE1/2352, MODE2/2336, MODE2/2352, CDI/2336, CDI/2352)
- **INDEX**: Specifies track position (INDEX 01 required, INDEX 00 for pregap)

**Metadata Commands:**
- **REM**: Comments and non-standard extensions
- **TITLE**: Disc or track title (max 80 characters for CD-Text)
- **PERFORMER**: Disc or track performer (max 80 characters)
- **SONGWRITER**: Track songwriter information
- **CDTEXTFILE**: References external CD-Text file

**Disc-Level Commands:**
- **CATALOG**: 13-digit UPC/EAN code (Media Catalog Number)

**Track-Level Commands:**
- **FLAGS**: Subcode flags (DCP, 4CH, PRE, SCMS)
- **ISRC**: International Standard Recording Code (12 characters)
- **PREGAP**: Length of track pregap
- **POSTGAP**: Length of track postgap

**Extended REM Commands (Non-Standard but Widely Supported):**
- **REM GENRE "<value>"**: Music genre classification
- **REM DATE "<value>"**: Release date (year or full date)
- **REM DISCID "<value>"**: Disc identification (e.g., FreeDB/CDDB ID)
- **REM COMMENT "<value>"**: General comments or tool signatures
- **REM DISCNUMBER <n>**: Disc number in multi-disc sets (1-15 for Kodi)
- **REM COMPOSER "<value>"**: Composer information
- **REM REPLAYGAIN_ALBUM_GAIN <value>**: Album-level replay gain in dB
- **REM REPLAYGAIN_ALBUM_PEAK <value>**: Album peak level (0.0-1.0)
- **REM REPLAYGAIN_TRACK_GAIN <value>**: Track-level replay gain in dB
- **REM REPLAYGAIN_TRACK_PEAK <value>**: Track peak level (0.0-1.0)

#### Time Format Handling (Wikipedia Specification)
- **Format**: MM:SS:FF (minute:second:frame)
- **Frame Rate**: 75 frames per second of audio
- **Calculation**: 1 second = 75 frames
- **INDEX Types**:
  - INDEX 00: Start of pregap (optional)
  - INDEX 01: Start of track data (required)
  - INDEX 02-99: Additional subdivision points (optional)
- **Cumulative Timing**: All times are cumulative from start of file
- **Frame Precision**: Maintain exact frame accuracy for CD compatibility

#### File Format Support
- **Audio Formats**: WAVE, MP3, AIFF
- **Lossless Formats**: FLAC (typically uses WAVE type designation)
- **Binary Formats**: BINARY (little-endian), MOTOROLA (big-endian)
- **Data Track Modes**: MODE1/2048, MODE1/2352, MODE2/2336, MODE2/2352, MODE2/2324 (XA form-2), CDI/2336, CDI/2352
- **Multiple Files**: Support for CUE sheets referencing multiple data files
- **Mixed-Mode Support**: Audio and data tracks in same CUE sheet
- **File Path Handling**: Relative and absolute paths, quoted filenames (optional but recommended)

#### Character Encoding
- Plain text format with .cue extension
- UTF-8 as default encoding
- Extended ASCII support for legacy files
- Auto-detection for various encodings (UTF-8, Latin-1, Windows-1252)
- Alternative comment syntax support: REM, ";" prefix, "//" prefix
- Proper handling of quoted strings and special characters
- Whitespace flexibility (spaces and tabs for readability)

### User Stories

#### Story 5.1: Parse Existing CUE Files
**As a** DJ with existing CUE files
**I want** the system to read and interpret my CUE files correctly
**So that** I can import my existing mix information

**Acceptance Criteria:**
- Parse all standard CUE commands successfully
- Handle multiple CUE format variants
- Extract complete track listing with timestamps
- Preserve all metadata and custom fields
- Report parsing errors with helpful messages

#### Story 5.2: Generate CUE Files from Tracklists
**As a** user who has identified tracks in a mix
**I want** to generate a properly formatted CUE file
**So that** I can use it in my DJ software

**Acceptance Criteria:**
- Generate valid CUE files accepted by major DJ software
- Include all required metadata fields
- Calculate accurate INDEX positions
- Support custom REM fields for extended data
- Option to choose output format variant

#### Story 5.3: Edit and Update CUE Files
**As a** DJ refining my mix markers
**I want** to edit CUE files programmatically
**So that** I can correct timestamps and update track information

**Acceptance Criteria:**
- Add/remove/reorder tracks
- Adjust timestamps while maintaining validity
- Update metadata without corruption
- Preserve format-specific extensions
- Create backup before modifications

#### Story 5.4: Validate CUE File Integrity
**As a** user working with CUE files
**I want** validation of CUE file correctness
**So that** I can ensure they work properly

**Acceptance Criteria:**
- Verify syntax correctness
- Validate referenced audio files exist
- Check timestamp consistency and boundaries
- Ensure total time matches audio duration
- Report all issues with clear descriptions

#### Story 5.5: Convert Between CUE Formats
**As a** DJ using multiple software platforms
**I want** to convert CUE files between different formats
**So that** I can use them across all my tools

**Acceptance Criteria:**
- Convert between standard and CDJ formats
- Preserve all possible metadata
- Handle format-specific limitations gracefully
- Provide conversion reports
- Batch conversion support

## Implementation Approach

### Phase 1: Parser Implementation (Week 1)
1. Implement core CUE parser
2. Add support for all CUE commands
3. Handle multiple encodings
4. Create comprehensive test suite
5. Document parsing rules

### Phase 2: Generator Development (Week 1-2)
1. Design generation API
2. Implement standard CUE generation
3. Add format variant support
4. Create validation framework
5. Test with major DJ software

### Phase 3: Editor and Utilities (Week 2-3)
1. Implement editing capabilities
2. Add conversion utilities
3. Build validation tools
4. Create batch processing
5. Performance optimization

## API Specification (Draft)

### Python API
```python
class CueSheet:
    def __init__(self, file_path=None)
    def parse(self, content: str, encoding: str = 'auto')  # Auto-detect encoding
    def add_file(self, filename: str, file_type: str)  # WAVE, MP3, FLAC->WAVE, BINARY, etc.
    def add_track(self, track: Track)
    def remove_track(self, index: int)
    def update_track(self, index: int, track: Track)
    def set_catalog(self, catalog: str)  # 13-digit UPC/EAN
    def set_cdtextfile(self, filename: str)
    def set_title(self, title: str)  # Disc-level title
    def set_performer(self, performer: str)  # Disc-level performer
    def add_rem(self, key: str, value: str)  # Extended REM metadata
    def set_genre(self, genre: str)  # REM GENRE
    def set_date(self, date: str)  # REM DATE
    def set_discid(self, discid: str)  # REM DISCID
    def set_comment(self, comment: str)  # REM COMMENT
    def set_discnumber(self, number: int)  # REM DISCNUMBER (1-15)
    def set_composer(self, composer: str)  # REM COMPOSER
    def set_replaygain(self, gain_type: str, value: float)  # ReplayGain values
    def generate(self, format: CueFormat) -> str
    def validate(self) -> List[ValidationError]
    def validate_command_order(self) -> List[ValidationError]  # Check proper command sequence
    def convert_to(self, format: CueFormat) -> CueSheet

class Track:
    def __init__(self, number: int, data_type: str = "AUDIO")  # AUDIO, CDG, MODE1/2048, MODE2/2324, CDI/2336, etc.
    def add_index(self, number: int, time: CueTime)  # INDEX 00, 01, 02...
    def set_title(self, title: str)  # Max 80 chars for CD-Text
    def set_performer(self, performer: str)  # Max 80 chars
    def set_songwriter(self, songwriter: str)
    def set_isrc(self, isrc: str)  # 12-character ISRC
    def set_flags(self, flags: List[str])  # DCP, 4CH, PRE, SCMS
    def set_pregap(self, time: CueTime)
    def set_postgap(self, time: CueTime)
    def add_rem(self, key: str, value: str)  # Track-specific REM metadata
    def set_replaygain(self, gain: float, peak: float)  # Track ReplayGain values

class CueTime:
    def __init__(self, minutes: int, seconds: int, frames: int)
    def to_frames(self) -> int  # Total frames (75 fps)
    def to_milliseconds(self) -> int
    @classmethod
    def from_frames(cls, frames: int) -> CueTime
    @classmethod
    def from_milliseconds(cls, ms: int) -> CueTime
    def __str__(self) -> str  # Returns "MM:SS:FF" format
```

### REST API
```
POST /api/v1/cue/parse
  - Body: CUE file content
  - Returns: Parsed structure

POST /api/v1/cue/generate
  - Body: Track list with timestamps
  - Returns: Generated CUE content

PUT /api/v1/cue/update
  - Body: CUE content + modifications
  - Returns: Updated CUE content

POST /api/v1/cue/validate
  - Body: CUE content
  - Returns: Validation results

POST /api/v1/cue/convert
  - Body: CUE content + target format
  - Returns: Converted CUE content
```

## Success Metrics
- Parse 100% of valid CUE files successfully
- Generated CUEs accepted by all major DJ software
- Validation catches 100% of format errors
- Processing time <100ms for typical CUE files
- Zero data loss during conversions
- API response time <500ms

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Format specification ambiguity | High | Research multiple implementations (Wikipedia, Hydrogenaudio, GNU), extensive testing |
| Character encoding issues | Medium | Robust detection, support UTF-8/ASCII/Latin-1, extensive testing |
| DJ software compatibility | High | Test with all major platforms (CDJ, Traktor, Serato, Rekordbox), maintain test suite |
| Application-specific REM fields | Medium | Support common extensions, provide flexible REM handling API |
| Frame calculation discrepancies | Medium | Consistent 75 fps standard, handle 0-74 range notation correctly |
| FLAC file type mapping | Low | Auto-detect FLAC and map to WAVE type for compatibility |
| Alternative comment syntax | Low | Support REM, ";", and "//" prefixes for maximum compatibility |
| Command ordering violations | Medium | Implement validation and auto-correction where possible |
| Performance with large CUEs | Low | Optimize parsing, implement streaming if needed |
| Timestamp precision loss | Medium | Use high-precision internal representation |

## Testing Strategy
- Unit tests for all parser components including REM extensions
- Integration tests with real CUE files from various sources
- Compatibility testing with DJ software (CDJ, Traktor, Serato, Rekordbox)
- Media player testing (Kodi, foobar2000, VLC)
- Character encoding tests (UTF-8, ASCII, Latin-1, Windows-1252)
- Command ordering validation tests
- FLAC file type mapping tests
- Alternative comment syntax tests (REM, ";", "//")
- ReplayGain metadata handling tests
- Multi-disc numbering tests
- Fuzz testing for parser robustness
- Performance benchmarks
- Round-trip testing (parse → generate → parse)

## Dependencies
- Epic 1: Core service infrastructure
- Libraries: Python CUE parsing libraries (evaluate existing options)
- Testing: Sample CUE files from various sources

## Definition of Done
- [ ] All user stories completed and accepted
- [ ] Complete CUE specification support
- [ ] Unit test coverage >95% for parser/generator
- [ ] Integration tests with real files
- [ ] Compatibility verified with major DJ software
- [ ] Performance benchmarks met
- [ ] API documentation complete
- [ ] Error handling comprehensive
- [ ] Code reviewed and approved
- [ ] Deployed to staging environment
