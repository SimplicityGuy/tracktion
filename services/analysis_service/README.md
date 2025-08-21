# Analysis Service

Comprehensive audio analysis service for the Tracktion project, providing BPM detection, musical key detection, mood analysis, genre classification, and metadata extraction.

## Features

### Audio Analysis Capabilities
- **BPM Detection**: Multi-algorithm tempo detection with temporal analysis
- **Musical Key Detection**: Dual-algorithm validation with harmonic analysis
- **Mood Analysis**: Deep learning-based mood dimension scoring
- **Genre Classification**: Discogs EffNet model-based classification
- **Metadata Extraction**: Title, artist, album, duration, bitrate, sample rate, format

### Infrastructure Features
- **Audio Format Support**: MP3, FLAC, WAV, M4A, OGG Vorbis
- **Dual Storage**: PostgreSQL for relational data, Neo4j for graph relationships
- **Message Queue Integration**: RabbitMQ for asynchronous processing
- **Redis Caching**: Intelligent caching with confidence-based TTL
- **Performance Optimization**: Parallel analysis and lazy model loading
- **Error Handling**: Retry logic with exponential backoff
- **Health Checks**: Service health monitoring

## Architecture

The service follows a microservices architecture:

```
RabbitMQ Message → Consumer → Metadata Extractor → Storage Handler → Databases
```

## Project Structure

```
analysis_service/
├── src/
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # Service entry point
│   ├── message_consumer.py     # RabbitMQ message consumer
│   ├── metadata_extractor.py   # Audio metadata extraction
│   ├── storage_handler.py      # Database storage logic
│   ├── audio_cache.py          # Redis caching layer
│   ├── bpm_detector.py         # BPM detection module
│   ├── key_detector.py         # Musical key detection
│   ├── mood_analyzer.py        # Mood and genre analysis
│   ├── model_manager.py        # TensorFlow model management
│   ├── temporal_analyzer.py    # Temporal BPM analysis
│   ├── analysis_query.py       # Query interface for results
│   └── exceptions.py           # Custom exceptions
├── models/                      # Pre-trained TensorFlow models
├── Dockerfile                   # Container definition
├── pyproject.toml              # Python dependencies
└── README.md                   # This file
```

## Configuration

The service is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `NEO4J_URI` | Neo4j Bolt connection URI | Required |
| `NEO4J_USER` | Neo4j username | Required |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `RABBITMQ_URL` | RabbitMQ connection URL | Required |
| `REDIS_HOST` | Redis server hostname | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `ANALYSIS_QUEUE` | Queue name for messages | `analysis_queue` |
| `EXCHANGE_NAME` | RabbitMQ exchange name | `tracktion_exchange` |
| `ANALYSIS_ROUTING_KEY` | Routing key for messages | `file.analyze` |
| `ENABLE_CACHE` | Enable Redis caching | `true` |
| `ENABLE_TEMPORAL_ANALYSIS` | Enable temporal BPM analysis | `true` |
| `ENABLE_KEY_DETECTION` | Enable musical key detection | `true` |
| `ENABLE_MOOD_ANALYSIS` | Enable mood/genre analysis | `true` |
| `MODELS_DIR` | Directory for TensorFlow models | `./models` |
| `AUTO_DOWNLOAD_MODELS` | Auto-download missing models | `true` |
| `MAX_RETRIES` | Maximum retry attempts | `3` |
| `RETRY_DELAY` | Base retry delay in seconds | `5.0` |

## Message Format

The service expects messages in the following JSON format:

```json
{
  "recording_id": "uuid-string",
  "file_path": "/path/to/audio/file.mp3",
  "timestamp": 1234567890.123
}
```

## Running Locally

1. Install dependencies:
```bash
cd services/analysis_service
uv pip install -e .
```

2. Set environment variables:
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/tracktion"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
```

3. Run the service:
```bash
uv run python -m src.main
```

## Running with Docker

The service is included in the main docker-compose configuration:

```bash
docker-compose up analysis-service
```

## Testing

Run unit tests:
```bash
uv run pytest tests/unit/services/analysis/
```

Run integration tests:
```bash
uv run pytest tests/integration/test_analysis_service.py
```

## Extracted Metadata

The service extracts the following metadata fields:

### Basic Metadata
- `title` - Track title
- `artist` - Track artist
- `album` - Album name
- `date` - Release date
- `genre` - Music genre
- `track` - Track number

### Technical Metadata
- `duration` - Track duration in seconds
- `bitrate` - Audio bitrate in bps
- `sample_rate` - Sample rate in Hz
- `channels` - Number of audio channels
- `format` - File format

### Format-Specific Metadata
- MP3: `version`, `layer`
- FLAC: `bits_per_sample`, `albumartist`
- MP4/M4A: `codec`, `codec_description`
- OGG Vorbis: Comprehensive Vorbis comment extraction including:
  - **Standard fields**: `title`, `version`, `album`, `artist`, `performer`, `copyright`, `license`, `organization`, `description`, `genre`, `date`, `location`, `contact`, `isrc`
  - **Extended fields**: `albumartist`, `composer`, `conductor`, `discnumber`, `disctotal`, `totaltracks`, `publisher`, `label`, `compilation`, `lyrics`, `language`, `mood`, `bpm`, `key`
  - **ReplayGain tags**: `replaygain_track_gain`, `replaygain_track_peak`, `replaygain_album_gain`, `replaygain_album_peak`
  - **Technical info**: `bitrate_nominal`, `bitrate_lower`, `bitrate_upper`, `bitrate_mode` (VBR/CBR detection), `file_size`
  - **Custom tags**: All non-standard tags preserved with original case in `custom_tags` field (JSON format)
  - **Multiple values**: Multiple values per tag are joined with semicolons

## Database Schema

### PostgreSQL (Metadata table)
- `id` - UUID primary key
- `recording_id` - Foreign key to Recording
- `key` - Metadata field name
- `value` - Metadata value

### Neo4j Graph Structure
- Nodes: `Recording`, `Metadata`, `Artist`, `Album`, `Genre`
- Relationships: `HAS_METADATA`, `PERFORMED_BY`, `PART_OF`, `HAS_GENRE`

## Error Handling

The service implements comprehensive error handling:

1. **Invalid Audio Files**: Marked as "invalid" status
2. **Extraction Errors**: Retried with exponential backoff
3. **Storage Errors**: Retried and logged
4. **Connection Errors**: Automatic reconnection attempts

## Troubleshooting

### Service won't start
- Check all environment variables are set
- Verify database connections
- Ensure RabbitMQ is running

### Files not processing
- Check RabbitMQ queue for messages
- Verify file paths are accessible
- Check service logs for errors

### Metadata not extracted
- Ensure file format is supported
- Check file isn't corrupted
- Verify mutagen library is installed

## Advanced Audio Analysis

### BPM Detection
The service provides multi-algorithm BPM detection:
- **Multifeature**: Default algorithm with high accuracy
- **Percival**: Alternative for electronic music
- **Degara**: Optimized for classical music
- **Temporal Analysis**: Detects tempo changes throughout the track

### Musical Key Detection
Dual-algorithm approach for accuracy:
- **Primary**: Essentia KeyExtractor
- **Validation**: HPCP-based harmonic analysis
- **Agreement Check**: Higher confidence when algorithms agree
- **Harmonic Compatibility**: Identifies compatible keys for mixing

### Mood and Genre Analysis
Deep learning-based analysis using TensorFlow:
- **Mood Dimensions**: Happy, sad, aggressive, relaxed, acoustic, electronic
- **Genre Classification**: Using Discogs EffNet models
- **Danceability**: Score from 0-1
- **Energy Level**: Calculated from multiple features
- **Valence/Arousal**: Musical positivity and intensity

## Performance Benchmarks

Typical processing times for a 30-second audio file:
- BPM Detection: ~500ms
- Key Detection: ~300ms
- Mood Analysis: ~800ms (with models loaded)
- Full Pipeline: ~1.5s (parallel) / ~2.5s (sequential)

Cache performance:
- Read operations: >1000 ops/sec
- Write operations: >800 ops/sec

## API Usage Examples

### Query Analysis Results
```python
from services.analysis_service.src.analysis_query import AnalysisQuery

query = AnalysisQuery(storage_handler=storage)

# Get results for a recording
result = query.get_analysis_result(recording_id)
print(f"BPM: {result.bpm_data['bpm']}")
print(f"Key: {result.key_data['key']} {result.key_data['scale']}")
print(f"Genre: {result.mood_data['primary_genre']}")

# Find compatible tracks
compatible = query.get_compatible_recordings(
    recording_id,
    compatibility_type="harmonic"
)
```

## Future Enhancements

- Support for additional audio formats (AAC, OPUS, WebM)
- Real-time analysis streaming
- Batch processing optimization
- Advanced ML models for genre classification
- Web API for direct queries
- Automatic playlist generation based on analysis
