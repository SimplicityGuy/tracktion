# Epic 6: Tracklist Management

## Epic Overview
**Epic ID:** EPIC-6
**Epic Name:** Tracklist Management
**Priority:** High
**Dependencies:** Epic 1 (Foundation), Epic 2 (Metadata), Epic 4 (1001tracklists API), Epic 5 (CUE Handler)
**Estimated Effort:** 3-4 weeks

*Note: This epic was previously Epic 3 but has been renumbered to Epic 6 to accommodate new prerequisite epics.*

## Business Value
Tracklist management is central to DJ workflow optimization. This epic will:
- Automatically identify and catalog tracks within DJ mixes
- Generate accurate CUE files for seamless mix navigation
- Enable track discovery and library building from favorite mixes
- Support both automatic and manual tracklist curation
- Provide a foundation for advanced mix analysis features

## Technical Scope

### Core Requirements
1. **Tracklist Data Management**
   - Store tracklist information in catalog
   - Link tracks to audio files
   - Support multiple tracklists per file
   - Version control for tracklist edits
   - Import/export capabilities

2. **1001tracklists Integration**
   - Query API for tracklist data
   - Match local files to online mixes
   - Import complete tracklists
   - Handle partial matches
   - Update existing tracklists

3. **CUE File Generation**
   - Generate CUE from tracklist data
   - Support multiple CUE formats
   - Embed additional metadata
   - Validate against audio file
   - Batch generation capabilities

4. **Tracklist Editing Interface**
   - API for tracklist CRUD operations
   - Track reordering and timing adjustment
   - Metadata correction
   - Manual track identification
   - Collaborative editing support

### Technical Considerations

#### Data Model
```python
class Tracklist:
    id: UUID
    audio_file_id: UUID
    source: str  # manual, 1001tracklists, auto-detected
    created_at: datetime
    updated_at: datetime
    tracks: List[TrackEntry]
    cue_file_id: Optional[UUID]
    confidence_score: float

class TrackEntry:
    position: int
    start_time: timedelta
    end_time: Optional[timedelta]
    artist: str
    title: str
    remix: Optional[str]
    label: Optional[str]
    catalog_track_id: Optional[UUID]  # Link to catalog
    confidence: float
    transition_type: Optional[str]
```

#### Integration Architecture
- **Service Communication**: RabbitMQ for async processing
- **Data Flow**: File → Analysis → Tracklist → CUE Generation
- **Caching**: Redis for API responses and processing state
- **Storage**: PostgreSQL for tracklist data, S3 for CUE files

### User Stories

#### Story 6.1: Import Tracklist from 1001tracklists
**As a** DJ with recorded mixes
**I want** to import tracklists from 1001tracklists
**So that** I can generate CUE files for my recordings

**Acceptance Criteria:**
- Search and retrieve tracklists via API
- Match tracklist to local audio file
- Import all track information
- Handle timing adjustments
- Generate CUE file automatically

#### Story 6.2: Manual Tracklist Creation
**As a** user with unidentified mixes
**I want** to manually create tracklists
**So that** I can document my mixes

**Acceptance Criteria:**
- Add tracks with timestamps
- Search catalog for track matches
- Adjust timing while listening
- Save drafts and versions
- Export to CUE format

#### Story 6.3: Automatic Tracklist Detection
**As a** user with many mixes
**I want** automatic tracklist detection attempts
**So that** I can quickly catalog my collection

**Acceptance Criteria:**
- Audio fingerprinting attempts
- Fuzzy matching against known tracks
- Confidence scoring for matches
- Manual verification workflow
- Batch processing support

#### Story 6.4: CUE File Generation
**As a** DJ using CDJs/software
**I want** CUE files generated from tracklists
**So that** I can navigate mixes easily

**Acceptance Criteria:**
- Generate valid CUE files
- Support multiple formats
- Include all metadata
- Validate against audio duration
- Store in appropriate location

#### Story 6.5: Tracklist Synchronization
**As a** user with evolving tracklists
**I want** to keep tracklists synchronized
**So that** updates are reflected everywhere

**Acceptance Criteria:**
- Update CUE when tracklist changes
- Sync with 1001tracklists updates
- Version history maintained
- Conflict resolution options
- Audit trail of changes

#### Story 6.6: Track Library Building
**As a** DJ discovering new music
**I want** to identify and catalog individual tracks from mixes
**So that** I can build my library

**Acceptance Criteria:**
- Extract track information from tracklists
- Search for individual tracks online
- Add to want-list or catalog
- Track source mix information
- Discovery statistics

## Implementation Approach

### Phase 1: Data Foundation (Week 1)
1. Design tracklist data models
2. Create database schemas
3. Build CRUD APIs
4. Implement data validation
5. Set up testing framework

### Phase 2: 1001tracklists Integration (Week 2)
1. Integrate with Epic 4 API
2. Build matching algorithms
3. Implement import workflows
4. Add confidence scoring
5. Handle edge cases

### Phase 3: CUE Generation (Week 2-3)
1. Integrate with Epic 5 handler
2. Build generation pipeline
3. Add format selection
4. Implement validation
5. Set up batch processing

### Phase 4: Advanced Features (Week 3-4)
1. Manual editing interface
2. Audio fingerprinting integration
3. Synchronization services
4. Track discovery features
5. Performance optimization

## API Specification

### REST Endpoints
```
# Tracklist Management
GET /api/v1/tracklists
  - Query params: audio_file_id, source, date_range

POST /api/v1/tracklists
  - Body: Tracklist data

PUT /api/v1/tracklists/{id}
  - Body: Updated tracklist

DELETE /api/v1/tracklists/{id}

# Import Operations
POST /api/v1/tracklists/import/1001tracklists
  - Body: { url: "...", audio_file_id: "..." }

POST /api/v1/tracklists/detect
  - Body: { audio_file_id: "..." }

# CUE Operations
POST /api/v1/tracklists/{id}/generate-cue
  - Query params: format

GET /api/v1/tracklists/{id}/cue
  - Returns: CUE file content

# Track Discovery
GET /api/v1/tracklists/{id}/tracks/discover
  - Returns: Available sources for tracks

POST /api/v1/tracklists/{id}/tracks/{position}/identify
  - Body: { artist: "...", title: "..." }
```

## Success Metrics
- Successful import rate >90% for 1001tracklists
- CUE generation success rate 100%
- Manual editing response time <500ms
- Track identification accuracy >80%
- User satisfaction score >4/5
- Processing time <30s per mix

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Inaccurate timing data | High | Manual adjustment tools, validation checks |
| API changes/unavailability | High | Caching, fallback options, manual entry |
| Large tracklist performance | Medium | Pagination, lazy loading, indexing |
| Conflicting tracklist versions | Medium | Version control, merge tools |
| Audio fingerprinting accuracy | Medium | Multiple algorithms, manual verification |

## Integration Points
- **Epic 1**: Core catalog and file management
- **Epic 2**: Metadata extraction and analysis
- **Epic 4**: 1001tracklists API for data import
- **Epic 5**: CUE file handler for generation
- **Epic 7**: Will benefit from async refactor

## Dependencies
- Epic 1: Foundation infrastructure
- Epic 2: Metadata capabilities
- Epic 4: 1001tracklists API
- Epic 5: CUE file handler
- External: Audio fingerprinting service (optional)

## Definition of Done
- [ ] All user stories completed and accepted
- [ ] API endpoints implemented and documented
- [ ] Integration with Epic 4 & 5 complete
- [ ] Unit test coverage >90%
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] User documentation complete
- [ ] Error handling comprehensive
- [ ] Code reviewed and approved
- [ ] Deployed to staging environment
- [ ] End-to-end workflow tested
