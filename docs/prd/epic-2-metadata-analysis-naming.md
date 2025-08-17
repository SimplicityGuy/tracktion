# Epic 2: Metadata Analysis & Naming

This epic builds upon the foundation established in Epic 1 to deliver the core audio analysis capabilities and file management features. It focuses on extracting meaningful metadata from audio files, storing it in the graph database for relationship analysis, and implementing intelligent file renaming based on user-defined patterns. The stories are designed to build incrementally, starting with basic metadata extraction and progressing to advanced analysis and file management.

## Story 2.1: Analysis Service Setup & Basic Metadata Extraction

**As a** music-loving developer,
**I want** a functional analysis service that can extract basic metadata from audio files,
**so that** I can understand the technical properties and embedded information of my music collection.

### Acceptance Criteria
1. The analysis_service is created as a containerized Python service with proper project structure.
2. The service subscribes to RabbitMQ queues to receive file processing requests from the cataloging service.
3. Basic metadata extraction is implemented using mutagen or similar library (title, artist, album, duration, bitrate, sample rate, file format).
4. Extracted metadata is stored in both PostgreSQL (via Metadata model) and Neo4j (as graph nodes and relationships).
5. The service handles common audio formats (MP3, FLAC, WAV, M4A at minimum).
6. Error handling is implemented for corrupted or unsupported files.
7. Unit tests cover metadata extraction logic with sample audio files.
8. Integration tests verify end-to-end flow from message consumption to database storage.

## Story 2.2: Research Spike - Audio Analysis Libraries and Techniques

**As a** technical lead,
**I want** to research and validate the best approaches for BPM detection and mood/genre analysis,
**so that** we can implement Stories 2.3 and 2.4 with confidence in our technical choices.

### Acceptance Criteria
1. Evaluate Essentia, librosa, madmom, and other relevant libraries for:
   - BPM detection accuracy and performance
   - Mood/genre classification capabilities
   - Key detection algorithms
   - Memory and CPU/GPU requirements
2. Prototype BPM detection with temporal analysis:
   - Implement BPM detection per time window (e.g., every 10 seconds)
   - Calculate start/end BPM for tracks with tempo changes
   - Determine average BPM with confidence scores
3. Validate Essentia's pre-trained models:
   - Test mood classification models on sample dataset
   - Evaluate genre detection accuracy
   - Assess model loading time and memory footprint
4. Research musical key detection approaches:
   - Compare available algorithms and libraries
   - Test accuracy on various musical genres
5. Document findings with:
   - Recommended library choices with justification
   - Performance benchmarks
   - Implementation guidelines
   - Resource requirements
   - Sample code for each approach
6. Create decision matrix comparing all options
7. Provide clear recommendation for production implementation

### Research Deliverables
- Technical report with benchmarks and recommendations
- Prototype code for BPM temporal analysis
- Validated approach for mood/genre detection
- Key detection algorithm selection
- Updated technical requirements for Stories 2.3 and 2.4

## Story 2.3: Advanced Audio Analysis - BPM Detection

**As a** DJ or music enthusiast,
**I want** automatic BPM (beats per minute) detection for my audio files with temporal analysis,
**so that** I can organize my music by tempo and identify tracks with tempo changes.

### Acceptance Criteria
1. BPM detection algorithm is integrated using the library selected in Story 2.2 research (likely Essentia or librosa).
2. Temporal BPM analysis is implemented:
   - BPM calculated for configurable time windows (default: 10-second intervals)
   - Start BPM (first 30 seconds) and End BPM (last 30 seconds) are captured
   - Average BPM across entire track is calculated
   - Tempo stability score indicates if BPM is constant or variable
3. The analysis service can process audio files with accuracy within ±2 BPM tolerance for constant-tempo tracks.
4. BPM values and temporal data are stored as structured metadata in both PostgreSQL and Neo4j:
   - Average BPM as primary value
   - Start/End BPM for tempo changes
   - Temporal BPM array for detailed analysis (optional storage based on configuration)
   - Confidence scores for all measurements
5. Processing is optimized to handle large files efficiently (streaming/chunking for files >100MB).
6. Caching mechanism using Redis prevents re-analysis of already processed files.
7. Performance benchmarks show processing time <30 seconds for typical 5-minute tracks.
8. Unit tests validate BPM detection accuracy using reference tracks with known BPMs.
9. Special handling for:
   - Tracks with multiple tempo changes (DJ mixes, live recordings)
   - Beatless or ambient tracks (return null or very low confidence)
   - Variable tempo music (classical, jazz)

### Technical Implementation Notes
- Implementation based on Story 2.2 research findings
- Consider parallel processing for temporal analysis windows
- Store detailed tempo map only when variance exceeds threshold

## Story 2.4: Musical Key and Mood Detection

**As a** music curator,
**I want** automatic detection of musical key and mood characteristics,
**so that** I can create harmonically compatible playlists and understand the emotional content of my music.

### Prerequisites
- Story 2.2 (Research Spike) must be complete with validated approach for mood/genre detection and key detection

### Acceptance Criteria
1. Musical key detection is implemented using appropriate audio analysis library (essentia preferred).
2. Key detection provides both the root note and scale type (major/minor) with confidence scores.
3. Advanced mood analysis is implemented using Essentia's pre-trained TensorFlow models providing:
   - Multiple mood dimensions (acoustic, electronic, aggressive, relaxed, happy, sad, party)
   - Danceability scores
   - Additional attributes (gender, tonality, voice/instrumental classification)
   - Genre classification using Discogs EffNet models
4. Confidence scores are provided for all predictions using ensemble model approach.
5. All detected features are stored as metadata in both databases with appropriate relationships in Neo4j.
6. The analysis pipeline can process multiple features in parallel for efficiency.
7. Results are validated against a test set of tracks with known keys (accuracy >80%).
8. Integration with existing metadata extraction ensures all analysis happens in a single pass when possible.
9. API endpoint or message format is defined for querying analysis results.

### Technical Implementation Notes
- Implementation approach determined by Story 2.2 research findings
- Use Essentia with pre-trained TensorFlow models if validated in research
- Model management strategy as determined in research spike
- **Mood Detection**: Use Essentia with pre-trained TensorFlow models (MusiCNN, VGGish variants)
- **Genre Detection**: Use Discogs EffNet models for genre/style classification
- **Model Management**: Download and cache models from Essentia's model repository
- **Ensemble Approach**: Average predictions from multiple models per attribute for robustness
- **Research Required**: The provided prototype code is a starting point, but requires:
  - Investigation of latest Essentia API changes and best practices
  - Evaluation of model accuracy for the specific music collection
  - Optimization for batch processing and memory management
  - Integration strategy for model downloads and updates

## Story 2.5: File Renaming Service Implementation

**As a** music collector,
**I want** automatic file renaming based on extracted metadata and configurable patterns,
**so that** my music files have consistent, meaningful names that help me organize my collection.

### Acceptance Criteria
1. A new file_renaming_service is created as a separate microservice or module within analysis_service.
2. Configurable naming patterns are supported using template syntax (e.g., "{artist} - {title} - {bpm}BPM").
3. The service validates new filenames for filesystem compatibility (removes invalid characters, handles length limits).
4. A dry-run mode allows preview of rename operations without actual file changes.
5. Batch renaming operations are supported with transaction-like behavior (all-or-nothing).
6. Original filenames are preserved in the database for rollback capability.
7. File rename events trigger catalog updates to maintain database consistency.
8. Configuration allows for different patterns based on file type or metadata conditions.
9. Safety checks prevent overwrites and handle naming conflicts (auto-increment or timestamp).
10. Unit tests cover various naming patterns and edge cases.

## Story 2.6: Analysis Pipeline Optimization & Monitoring

**As a** system administrator,
**I want** an optimized and observable analysis pipeline,
**so that** I can process large music collections efficiently and troubleshoot issues quickly.

### Acceptance Criteria
1. Queue-based priority system allows high-priority files to be analyzed first.
2. Batch processing mode can analyze multiple files in parallel (configurable concurrency).
3. Progress tracking provides real-time status of analysis queue and individual file processing.
4. Comprehensive logging with structured format enables debugging and monitoring.
5. Metrics are exposed for processing time, success/failure rates, and queue depths.
6. Circuit breaker pattern prevents cascade failures when external services are unavailable.
7. Graceful shutdown ensures no data loss when service is stopped.
8. Health check endpoints report service status and dependency availability.
9. Performance tests demonstrate ability to process 1000+ files per hour.
10. Documentation includes tuning guide for different hardware configurations.

## Epic Success Criteria
- Research spikes provide clear technical direction for audio analysis implementation
- All audio files in the watched directories have basic metadata extracted and stored
- BPM detection works reliably with temporal analysis for electronic/dance music (primary use case)
- Musical key detection provides usable results for harmonic mixing
- File renaming follows consistent patterns without data loss
- The analysis pipeline can handle the user's entire music collection without manual intervention
- System performance meets or exceeds requirements for a 10,000+ file collection

## Technical Considerations
- Analysis operations should be idempotent - reprocessing a file produces the same results
- Consider implementing a plugin architecture for future analysis extensions
- Memory management is critical for processing large audio files and TensorFlow models
- GPU acceleration could be beneficial for large-scale BPM/key detection and neural network inference
- Results should be versioned to allow for algorithm improvements without data loss
- **Model Management**:
  - Pre-trained models need to be downloaded and cached (approximately 2-3GB total)
  - Model versioning strategy required for updates
  - Consider containerizing models with the service
- **TensorFlow Integration**:
  - Requires TensorFlow 2.x for model inference
  - CPU vs GPU inference trade-offs need evaluation
  - Model loading optimization for service startup time

## Dependencies
- Epic 1 must be complete (infrastructure, databases, message queue)
- Audio analysis libraries:
  - Essentia (with TensorFlow support) for mood/genre detection
  - Additional libraries for key detection (requires research)
- Pre-trained models from Essentia model repository (see prototype code for URLs)
- TensorFlow 2.x for neural network inference
- Sample audio files needed for testing (various formats, genres, and qualities)

## Risks & Mitigations
- **Risk**: Audio analysis libraries may have conflicting dependencies
  - *Mitigation*: Use separate virtual environments or containers for each analysis type
- **Risk**: Processing large FLAC/WAV files may consume excessive memory
  - *Mitigation*: Implement streaming processing and file chunking
- **Risk**: BPM/key detection accuracy may vary by genre
  - *Mitigation*: Allow manual override and implement confidence thresholds
