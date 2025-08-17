# BPM Temporal Analysis Requirements

## Story 2.3: Advanced BPM Detection Requirements

### Temporal Analysis Specifications

#### 1. Time Window Analysis
- **Default window**: 10 seconds
- **Configurable range**: 5-30 seconds
- **Overlap**: 50% for smooth transition detection
- **Minimum track length**: 30 seconds for meaningful analysis

#### 2. Key Metrics

##### Primary Metrics
- **Average BPM**: Weighted average of all windows
- **Start BPM**: Average of first 30 seconds
- **End BPM**: Average of last 30 seconds
- **Stability Score**: 0-1 indicating tempo consistency

##### Secondary Metrics
- **Variance**: Standard deviation of window BPMs
- **Min/Max BPM**: Extreme values across the track
- **Tempo Changes**: Count of significant BPM changes (>5% difference)
- **Confidence Score**: Overall reliability of detection

#### 3. Storage Strategy

##### Always Store
- Average BPM (float)
- Start BPM (float)
- End BPM (float) 
- Stability Score (float, 0-1)
- Confidence Score (float, 0-1)

##### Conditionally Store
- Full temporal array if variance > threshold (configurable, default: 5 BPM)
- Tempo change points with timestamps
- Compressed temporal data for long tracks (>10 minutes)

##### Storage Format
```json
{
  "bpm_average": 128.5,
  "bpm_start": 127.0,
  "bpm_end": 130.0,
  "stability_score": 0.85,
  "confidence_score": 0.92,
  "variance": 3.2,
  "tempo_changes": 2,
  "temporal_data": [
    {"time": 0, "bpm": 127.0, "confidence": 0.95},
    {"time": 10, "bpm": 127.5, "confidence": 0.93},
    {"time": 20, "bpm": 128.0, "confidence": 0.94},
    // ... additional windows
  ]
}
```

### 4. Use Cases

#### DJ Mixing
- **Requirement**: High accuracy for constant-tempo electronic music
- **Focus**: Stable BPM detection for beatmatching
- **Output**: Clear indication of mix-friendly tracks (stability_score > 0.9)

#### Transition Detection
- **Requirement**: Identify tempo change points in DJ sets
- **Focus**: Precise timestamps of BPM changes
- **Output**: Array of transition points with before/after BPMs

#### Classical/Jazz
- **Requirement**: Handle variable tempo gracefully
- **Focus**: Capture overall tempo feel without false precision
- **Output**: Low stability score with average tempo range

#### Electronic Music
- **Requirement**: ±1 BPM accuracy for constant tempo
- **Focus**: Precise BPM for mixing and analysis
- **Output**: High confidence scores for reliable values

### 5. Special Cases

#### Beatless/Ambient Tracks
- Return null BPM with explanation
- Set confidence score to 0
- Store genre hint for future reference

#### Multiple Tempo Changes
- DJ mixes: Store transition points
- Live recordings: Capture tempo evolution
- Mashups: Identify dominant BPM sections

#### Edge Cases
- Tracks < 30 seconds: Single BPM value only
- Silence detection: Skip silent sections
- Half/Double time: Detect and note both interpretations

### 6. Performance Requirements

- **Processing Time**: <30 seconds for 5-minute track
- **Memory Usage**: <500MB for analysis
- **Batch Processing**: Support for parallel analysis of multiple tracks
- **Caching**: Use Redis to store results keyed by file hash

### 7. Research Questions for Story 2.2

1. **Library Comparison**: Essentia vs librosa vs madmom for BPM detection
2. **Window Size Optimization**: Best window size for different genres
3. **Confidence Calculation**: How to determine reliability of BPM detection
4. **GPU Acceleration**: Benefits for large-scale processing
5. **Streaming vs Full Load**: Memory/accuracy trade-offs

### 8. Integration Points

- **Metadata Service**: Store BPM data alongside other metadata
- **Neo4j**: Create tempo-based relationships between tracks
- **File Renaming**: Include BPM in filename patterns
- **Caching**: Redis key structure for temporal data

### 9. Testing Requirements

- **Reference Dataset**: 100+ tracks with known BPMs across genres
- **Accuracy Target**: ±2 BPM for electronic, ±5 BPM for acoustic
- **Performance Benchmark**: 1000 tracks/hour on standard hardware
- **Edge Case Coverage**: Beatless, variable tempo, short tracks

### 10. Future Enhancements

- **Beat Grid Export**: Generate beat grids for DJ software
- **Tempo Curve Visualization**: Visual representation of tempo changes
- **ML Model Training**: Improve detection using user corrections
- **Cross-fade Detection**: Identify mixing points in DJ sets