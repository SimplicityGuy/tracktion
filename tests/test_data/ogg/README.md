# OGG Vorbis Test Files

This directory contains sample OGG Vorbis files for testing the audio analysis pipeline.

## File Structure

- `sample.ogg` - A valid OGG Vorbis file with complete metadata
- `minimal.ogg` - A minimal OGG file with limited metadata
- `corrupted.ogg` - A corrupted OGG file for error handling tests

## Creating Test Files

To create valid OGG test files, you can use ffmpeg:

```bash
# Convert from another format to OGG
ffmpeg -i input.mp3 -c:a libvorbis -q:a 5 output.ogg

# Create a test tone OGG file
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -c:a libvorbis test_tone.ogg
```

## Metadata Structure

OGG Vorbis files use Vorbis Comments for metadata, which are similar to FLAC:
- title
- artist
- album
- date
- genre
- tracknumber
- albumartist
- comment
- encoder
- organization

## Testing Notes

- The file watcher service should detect `.ogg` and `.oga` extensions
- The metadata extractor uses mutagen.oggvorbis for extraction
- Files are cataloged with format identifier "ogg" or "oga"
