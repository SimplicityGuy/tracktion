# Epic 12: MP3 Naming Convention Implementation

## Epic Overview
**Epic ID:** EPIC-12
**Epic Name:** MP3 Naming Convention Implementation
**Priority:** High
**Dependencies:** Epic 2 (Renaming Service), Epic 6 (Media Discovery)
**Estimated Effort:** 2 weeks
**Reference Documentation:** Official MP3 Release Rules 4.1 (2021-MP3COUNCIL)

## Business Value
Implementing industry-standard MP3 naming conventions ensures:
- Professional compliance with scene release standards
- Consistent and predictable file naming patterns
- Improved compatibility with music management software
- Enhanced file organization and searchability
- Reduced confusion and errors in music library management
- Industry-recognized naming patterns for releases

## Technical Scope

### Core Requirements
1. **MP3 Release Rules 4.1 Compliance**
   - Implement official MP3 scene naming conventions
   - Support proper dirname structure and formatting
   - Enforce filename conventions per standards
   - Handle various source types and special cases

2. **Dirname Structure Implementation**
   - Artist Name - Title - Published Year - Group/Source format
   - Support for VA (Various Artists) releases
   - Multi-disc source tagging (-2CD-, -3DVD-, etc.)
   - Special tags: -BOOTLEG-, -MAG-, -OST-, -SINGLE-
   - Source tags: -WEB-, -CD-, -VINYL-, -FM-, -SAT-, etc.

3. **Filename Convention Enforcement**
   - Track Number - Artist Name - Song Title format
   - Proper handling of VA compilations
   - Featured artist and remix notation
   - Character limit compliance (255 total)

4. **Character Restrictions**
   - Only allow: a-z A-Z 0-9 _ . - ()
   - Replace invalid characters appropriately
   - Handle foreign character conversion
   - Maintain filesystem compatibility

### Technical Considerations

#### MP3 Release Rules Integration
Based on Official MP3 Release Rules 4.1 (2021):
- Dirname MUST contain: Artist - Title - Year - Group
- Filenames MUST contain: Track Number - Artist - Title
- Maximum dirname+filename: 255 characters
- Special source tags for different media types
- Proper VA (Various Artists) handling
- Multi-disc release notation

#### Source Type Support
Physical Media:
- CD/CDS/CDM (pressed CDDA)
- VINYL/VLS/LP (vinyl releases)
- DVD/DVDS (video DVD audio)
- TAPE (analog tape)
- FLASH (SlotMusic, CompactStick)

Digital Sources:
- WEB (digital downloads)
- MP3CD (MP3 files on data CD)
- STREAM (webstream - internal only)

Live Sources:
- FM/SAT/CABLE (broadcast)
- SBD (soundboard recording)
- AUD (audience recording)
- LINE (direct line recording)

### User Stories

#### Story 12.1: Core Naming Convention Engine
**As a** music organizer
**I want** files named according to MP3 scene standards
**So that** my library follows professional conventions

**Acceptance Criteria:**
- Implement dirname structure parser and generator
- Support all required tags and formats
- Handle character restrictions properly
- Validate against MP3 Rules 4.1
- Generate compliant names from metadata
- Handle edge cases and exceptions

#### Story 12.2: Source Type Detection and Tagging
**As a** release manager
**I want** proper source tags applied automatically
**So that** releases are correctly identified

**Acceptance Criteria:**
- Detect source type from metadata/context
- Apply appropriate source tags
- Support multi-disc notation
- Handle special cases (bootleg, promo, etc.)
- Validate source tag combinations
- Provide manual override options

#### Story 12.3: VA and Compilation Support
**As a** DJ/compiler
**I want** Various Artists releases handled correctly
**So that** compilations are properly organized

**Acceptance Criteria:**
- Detect VA compilations automatically
- Use "VA" as artist in dirname
- Include track artists in filenames
- Handle mixed artist albums
- Support DJ mixes and compilations
- Maintain proper track ordering

#### Story 12.4: Character and Length Compliance
**As a** system administrator
**I want** filesystem-safe naming
**So that** files work across all platforms

**Acceptance Criteria:**
- Enforce character restrictions
- Handle 255 character limit properly
- Convert invalid characters safely
- Truncate intelligently when needed
- Preserve essential information
- Test cross-platform compatibility

#### Story 12.5: Integration with Existing Services
**As a** developer
**I want** naming conventions integrated system-wide
**So that** all services use consistent naming

**Acceptance Criteria:**
- Update renaming_service with new conventions
- Integrate with media_discovery_service
- Update file organization patterns
- Maintain backward compatibility
- Provide migration tools
- Document integration points

## Implementation Approach

### Phase 1: Analysis and Design (Days 1-3)
1. Deep dive into MP3 Rules 4.1 documentation
2. Map existing renaming service capabilities
3. Identify gaps and required changes
4. Design naming convention engine
5. Create comprehensive test cases

### Phase 2: Core Engine Development (Days 4-7)
1. Implement dirname structure generator
2. Build filename convention handler
3. Add source type detection logic
4. Implement character restriction enforcement
5. Create length compliance manager
6. Unit test all components

### Phase 3: Special Cases (Days 8-10)
1. VA/compilation handling
2. Multi-disc support
3. Bootleg/promo detection
4. Live source formatting
5. OST and special releases
6. Edge case handling

### Phase 4: Integration (Days 11-12)
1. Update renaming_service
2. Integrate with media_discovery
3. Update file organization logic
4. Create migration utilities
5. Update documentation

### Phase 5: Testing and Validation (Days 13-14)
1. Comprehensive testing against MP3 Rules
2. Cross-platform compatibility testing
3. Performance validation
4. User acceptance testing
5. Documentation review

## Success Metrics
- 100% compliance with MP3 Rules 4.1
- All source types properly supported
- Character restrictions enforced
- Length limits properly handled
- VA releases correctly formatted
- Multi-disc notation working
- Integration with existing services complete
- Cross-platform compatibility verified

## Naming Pattern Examples

### Standard Album Release
```
Artist_Name-Album_Title-CD-2021-GROUP
01-artist_name-song_title.mp3
02-artist_name-another_song.mp3
```

### Various Artists Compilation
```
VA-Compilation_Title-2CD-2021-GROUP
01-first_artist-track_title.mp3
02-second_artist-different_track.mp3
```

### Web Single Release
```
Artist_Name-Song_Title-SINGLE-WEB-2021-GROUP
01-artist_name-song_title.mp3
```

### Live Broadcast
```
Artist_Name-Live_at_Venue-FM-2021-GROUP
01-artist_name-intro.mp3
02-artist_name-song_one_live.mp3
```

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Complex rules interpretation | High | Thorough documentation review, test cases |
| Breaking existing functionality | Medium | Maintain backward compatibility mode |
| Character encoding issues | Medium | Comprehensive unicode handling |
| Performance impact | Low | Optimize regex patterns, cache results |
| User confusion | Medium | Clear documentation, migration guide |

## Definition of Done
- [ ] MP3 Rules 4.1 fully implemented
- [ ] All source types supported
- [ ] Character restrictions enforced
- [ ] Length compliance validated
- [ ] VA/compilation support complete
- [ ] Multi-disc notation working
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Migration tools provided
- [ ] Performance benchmarks met
