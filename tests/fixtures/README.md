# Test Fixtures for Audio Analysis

This directory contains test fixtures for the BPM detection and audio analysis components.

## Audio Files

### Reference BPM Tracks
Place test audio files in this directory with known BPM values for accuracy testing:

- `test_120bpm_rock.mp3` - Rock track at 120 BPM (4/4 time)
- `test_128bpm_electronic.mp3` - Electronic track at 128 BPM (4/4 time)
- `test_85bpm_hiphop.mp3` - Hip-hop track at 85 BPM
- `test_140bpm_dnb.mp3` - Drum & Bass track at 140 BPM
- `test_variable_classical.mp3` - Classical piece with variable tempo
- `test_beatless_ambient.mp3` - Ambient track with no clear beat
- `test_dj_mix.mp3` - DJ mix with multiple tempo changes

### File Format Tests
- `test_audio.wav` - WAV format test
- `test_audio.flac` - FLAC format test
- `test_audio.m4a` - M4A/AAC format test
- `test_audio.ogg` - OGG Vorbis format test

### Edge Cases
- `test_silence.mp3` - Silent audio file
- `test_noise.mp3` - White noise
- `test_corrupted.mp3` - Corrupted/invalid file
- `test_very_short.mp3` - Very short audio (<1 second)
- `test_very_long.mp3` - Very long audio (>10 minutes)

## Note on Test Files

Due to copyright and repository size constraints, actual audio files are not included
in the repository. Instead:

1. **For Development**: Download royalty-free samples from sites like:
   - Freesound.org
   - Sample Focus
   - Looperman

2. **For CI/CD**: Use generated test signals:
   - Sine waves at specific frequencies
   - Click tracks at known BPMs
   - Synthesized drum patterns

3. **For Production Testing**: Use a separate test data repository with
   properly licensed audio files.

## Generating Test Audio

Use the `generate_test_audio.py` script to create synthetic test files:

```bash
python tests/fixtures/generate_test_audio.py
```

This will create basic test files with known characteristics for unit testing.
