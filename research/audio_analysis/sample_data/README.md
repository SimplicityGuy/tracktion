# Sample Data for Audio Analysis Research

## Overview

This directory contains sample audio files and ground truth data for evaluating audio analysis libraries.

## Test Files Required

For comprehensive testing, we need diverse audio samples across different:
- **Genres**: Electronic, Classical, Jazz, Rock, Pop, Hip-Hop, Folk
- **Tempos**: Slow (60-90 BPM), Medium (90-120 BPM), Fast (120-180 BPM), Variable
- **Keys**: Major and minor keys across the circle of fifths
- **Formats**: MP3, FLAC, WAV, M4A

## Ground Truth Data

The `ground_truth.csv` file contains verified metadata for test files:

| Column | Description |
|--------|-------------|
| filename | Audio file name |
| bpm | Verified tempo in beats per minute |
| key | Musical key (e.g., C major, A minor) |
| genre | Primary genre classification |
| mood | Mood descriptors (e.g., energetic, melancholic) |
| duration | Track duration in seconds |

## Obtaining Test Files

### Option 1: Free Music Archive
Download royalty-free tracks from:
- https://freemusicarchive.org/
- https://www.jamendo.com/
- https://incompetech.com/

### Option 2: Generate Test Signals
Use the included `generate_test_signals.py` script to create synthetic test files with known properties.

### Option 3: Personal Collection
Use your own music files, ensuring to document their properties in `ground_truth.csv`.

## File Naming Convention

```
{genre}_{bpm}bpm_{key}_{title}.{format}
```

Example: `electronic_128bpm_Aminor_testtrack.mp3`

## Legal Notice

Ensure all audio files used for testing comply with copyright laws. Use only:
- Royalty-free music
- Creative Commons licensed tracks
- Generated test signals
- Personal recordings with proper rights
