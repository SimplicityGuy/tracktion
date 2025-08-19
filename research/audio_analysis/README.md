# Audio Analysis Research

This directory contains research prototypes and evaluations for audio analysis libraries to inform the implementation of Stories 2.3 and 2.4.

## Overview

This research spike evaluates various audio analysis libraries for:
- BPM (tempo) detection with temporal analysis
- Mood and genre classification
- Musical key detection
- Performance benchmarking

## Setup

1. Create a virtual environment:
```bash
cd research/audio_analysis
uv venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
uv pip install -e .
uv pip install -e ".[dev]"  # For development dependencies
```

## Directory Structure

- `bpm_detection/` - BPM detection comparison and temporal analysis
- `mood_genre/` - Mood and genre classification experiments
- `key_detection/` - Musical key detection algorithms
- `benchmarks/` - Performance benchmarking suite
- `reports/` - Technical reports and findings
- `sample_data/` - Test audio files and ground truth data

## Libraries Evaluated

1. **Essentia** - Comprehensive audio analysis with pre-trained models
2. **librosa** - Python library for music and audio analysis
3. **madmom** - Audio signal processing with neural networks
4. **aubio** - Lightweight audio analysis library

## Running Tests

Each subdirectory contains specific test scripts:

```bash
# BPM Detection
uv run python bpm_detection/compare_bpm.py

# Mood/Genre Classification
uv run python mood_genre/essentia_models.py

# Key Detection
uv run python key_detection/compare_keys.py

# Performance Benchmarks
uv run python benchmarks/benchmark_suite.py
```

## Hardware Requirements

- **CPU**: Multi-core processor recommended for parallel processing
- **RAM**: Minimum 8GB, 16GB+ recommended for large model loading
- **Storage**: ~5GB for models and test data
- **GPU**: Optional but beneficial for TensorFlow-based models

## Notes

- All Python commands must use `uv` package manager
- Maximum line length: 120 characters
- Python version: 3.13
- See `reports/technical_report.md` for findings and recommendations
