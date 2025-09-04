# Common Tasks - Code Examples

This guide provides practical code examples for the most common Tracktion operations. Each example includes working code, expected output, and troubleshooting tips.

## Table of Contents

- [Basic Audio Analysis](#basic-audio-analysis)
- [Batch File Processing](#batch-file-processing)
- [Playlist Management](#playlist-management)
- [Database Operations](#database-operations)
- [File Organization](#file-organization)
- [API Integration](#api-integration)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)

---

## Basic Audio Analysis

### Analyze Single Audio File

```python
"""Basic audio analysis example."""

import asyncio
from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer
from pathlib import Path

async def analyze_single_file(file_path: str):
    """Analyze a single audio file for BPM, key, and mood."""

    # Initialize analyzers
    bpm_detector = BPMDetector(confidence_threshold=0.8)
    key_detector = KeyDetector(algorithm='hpcp')
    mood_analyzer = MoodAnalyzer()

    print(f"🎵 Analyzing: {Path(file_path).name}")

    try:
        # Run analysis concurrently for better performance
        bpm_result, key_result, mood_result = await asyncio.gather(
            bpm_detector.analyze(file_path),
            key_detector.analyze(file_path),
            mood_analyzer.analyze(file_path)
        )

        # Display results
        print(f"📊 Analysis Results:")
        print(f"  🥁 BPM: {bpm_result.bpm:.1f} (confidence: {bpm_result.confidence:.2f})")
        print(f"  🎹 Key: {key_result.key} (confidence: {key_result.confidence:.2f})")
        print(f"  😊 Mood: {', '.join(mood_result.moods[:3])}")
        print(f"  ⚡ Energy: {mood_result.energy:.2f}")
        print(f"  💃 Danceability: {mood_result.danceability:.2f}")

        return {
            'bpm': bpm_result.bpm,
            'key': key_result.key,
            'moods': mood_result.moods,
            'energy': mood_result.energy,
            'danceability': mood_result.danceability
        }

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

# Usage
if __name__ == "__main__":
    file_path = "/path/to/your/audio.mp3"
    result = asyncio.run(analyze_single_file(file_path))
```

**Expected Output:**
```
🎵 Analyzing: example_song.mp3
📊 Analysis Results:
  🥁 BPM: 128.5 (confidence: 0.94)
  🎹 Key: C major (confidence: 0.87)
  😊 Mood: energetic, happy, uplifting
  ⚡ Energy: 0.82
  💃 Danceability: 0.91
```

### Quick BPM Detection

```python
"""Quick BPM detection for multiple files."""

from tracktion.analysis import BPMDetector
import asyncio
from pathlib import Path

async def quick_bpm_detection(directory_path: str):
    """Quickly detect BPM for all audio files in a directory."""

    detector = BPMDetector(
        confidence_threshold=0.7,
        quick_mode=True  # Faster but less accurate
    )

    # Find all audio files
    directory = Path(directory_path)
    audio_files = []
    for ext in ['.mp3', '.flac', '.wav', '.m4a']:
        audio_files.extend(directory.glob(f"**/*{ext}"))

    print(f"🔍 Found {len(audio_files)} audio files")

    results = []
    for file_path in audio_files:
        try:
            bpm_result = await detector.analyze(str(file_path))
            results.append({
                'file': file_path.name,
                'bpm': bpm_result.bpm,
                'confidence': bpm_result.confidence
            })
            print(f"✅ {file_path.name}: {bpm_result.bpm:.1f} BPM")

        except Exception as e:
            print(f"❌ {file_path.name}: Analysis failed - {e}")

    # Sort by BPM
    results.sort(key=lambda x: x['bpm'])

    print(f"\n📈 BPM Range: {results[0]['bpm']:.1f} - {results[-1]['bpm']:.1f}")
    return results

# Usage
bpm_results = asyncio.run(quick_bpm_detection("/path/to/music/folder"))
```

---

## Batch File Processing

### Process Entire Music Library

```python
"""Batch process entire music library with progress tracking."""

import asyncio
import time
from pathlib import Path
from tracktion.analysis import BPMDetector, KeyDetector
from tracktion.utils import ProgressTracker
import json

class LibraryProcessor:
    def __init__(self, library_path: str, output_file: str = "library_analysis.json"):
        self.library_path = Path(library_path)
        self.output_file = output_file
        self.bpm_detector = BPMDetector()
        self.key_detector = KeyDetector()
        self.progress = ProgressTracker()

    async def process_library(self, max_workers: int = 4):
        """Process entire library with concurrent workers."""

        # Find all audio files
        audio_files = self._find_audio_files()
        total_files = len(audio_files)

        print(f"🎵 Processing {total_files} audio files with {max_workers} workers...")

        # Initialize progress tracking
        self.progress.start(total_files)

        # Process in batches
        batch_size = max_workers
        results = []

        for i in range(0, total_files, batch_size):
            batch = audio_files[i:i + batch_size]

            # Process batch concurrently
            batch_tasks = [self._process_single_file(file_path) for file_path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle results
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"⚠️ Processing error: {result}")
                    results.append({'error': str(result)})
                else:
                    results.append(result)

                self.progress.update(1)

        # Save results
        self._save_results(results)
        self.progress.finish()

        # Print summary
        successful = len([r for r in results if 'error' not in r])
        failed = total_files - successful

        print(f"\n📊 Processing Summary:")
        print(f"  ✅ Successfully processed: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  💾 Results saved to: {self.output_file}")

        return results

    def _find_audio_files(self):
        """Find all supported audio files."""
        audio_extensions = ['.mp3', '.flac', '.wav', '.m4a', '.aiff']
        files = []

        for ext in audio_extensions:
            files.extend(self.library_path.glob(f"**/*{ext}"))

        return files

    async def _process_single_file(self, file_path: Path):
        """Process a single audio file."""
        try:
            # Run analyzers concurrently
            bpm_result, key_result = await asyncio.gather(
                self.bpm_detector.analyze(str(file_path)),
                self.key_detector.analyze(str(file_path))
            )

            return {
                'file_path': str(file_path),
                'filename': file_path.name,
                'bpm': bpm_result.bpm,
                'bpm_confidence': bpm_result.confidence,
                'key': key_result.key,
                'key_confidence': key_result.confidence,
                'file_size': file_path.stat().st_size,
                'processed_at': time.time()
            }

        except Exception as e:
            return {
                'file_path': str(file_path),
                'filename': file_path.name,
                'error': str(e),
                'processed_at': time.time()
            }

    def _save_results(self, results):
        """Save processing results to JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

# Usage
async def process_my_library():
    processor = LibraryProcessor("/path/to/music/library")
    results = await processor.process_library(max_workers=6)
    return results

# Run processing
library_results = asyncio.run(process_my_library())
```

### Resume Interrupted Processing

```python
"""Resume batch processing from where it left off."""

import json
from pathlib import Path

class ResumableProcessor(LibraryProcessor):
    def __init__(self, library_path: str, output_file: str = "library_analysis.json"):
        super().__init__(library_path, output_file)
        self.processed_files = self._load_existing_results()

    def _load_existing_results(self):
        """Load previously processed files."""
        if Path(self.output_file).exists():
            with open(self.output_file, 'r') as f:
                existing_results = json.load(f)

            processed_files = set()
            for result in existing_results:
                if 'file_path' in result and 'error' not in result:
                    processed_files.add(result['file_path'])

            print(f"📂 Found {len(processed_files)} previously processed files")
            return processed_files

        return set()

    async def process_library(self, max_workers: int = 4):
        """Process library, skipping already processed files."""

        # Find all audio files
        all_files = self._find_audio_files()

        # Filter out already processed files
        remaining_files = [
            f for f in all_files
            if str(f) not in self.processed_files
        ]

        print(f"🎵 {len(remaining_files)} files remaining to process...")

        if not remaining_files:
            print("✅ All files already processed!")
            return []

        # Process remaining files using parent method logic
        return await super()._process_files_subset(remaining_files, max_workers)

# Usage
resumable_processor = ResumableProcessor("/path/to/music/library")
remaining_results = asyncio.run(resumable_processor.process_library())
```

---

## Playlist Management

### Create Smart Playlists

```python
"""Create smart playlists based on audio features."""

from tracktion.tracklist import PlaylistGenerator, TrackMatcher
from tracktion.database import TrackDatabase
import asyncio

class SmartPlaylistCreator:
    def __init__(self, database_url: str):
        self.db = TrackDatabase(database_url)
        self.generator = PlaylistGenerator()
        self.matcher = TrackMatcher()

    async def create_workout_playlist(self, duration_minutes: int = 45):
        """Create high-energy workout playlist."""

        criteria = {
            'bpm_range': (120, 160),
            'energy_min': 0.7,
            'danceability_min': 0.6,
            'duration_minutes': duration_minutes,
            'avoid_explicit': False,
            'genre_preferences': ['electronic', 'hip-hop', 'rock', 'pop']
        }

        print(f"🏋️ Creating {duration_minutes}-minute workout playlist...")

        # Get tracks matching criteria
        candidate_tracks = await self.db.find_tracks_by_criteria({
            'bpm': {'$gte': criteria['bpm_range'][0], '$lte': criteria['bpm_range'][1]},
            'energy': {'$gte': criteria['energy_min']},
            'danceability': {'$gte': criteria['danceability_min']}
        })

        # Generate playlist with energy progression
        playlist = await self.generator.create_energy_progression_playlist(
            tracks=candidate_tracks,
            duration_minutes=duration_minutes,
            energy_curve='build_and_sustain'  # Start moderate, build to high, sustain
        )

        # Save playlist
        playlist_data = {
            'name': f'Workout Mix - {duration_minutes} min',
            'description': 'High-energy tracks for workout sessions',
            'tracks': playlist,
            'total_duration': sum(track.duration for track in playlist),
            'created_at': datetime.now().isoformat()
        }

        await self.db.save_playlist(playlist_data)
        print(f"💪 Created workout playlist with {len(playlist)} tracks")

        return playlist_data

    async def create_chill_playlist(self, mood: str = 'relaxed', duration_minutes: int = 30):
        """Create chill/ambient playlist for relaxation."""

        criteria = {
            'bpm_range': (60, 100),
            'energy_max': 0.5,
            'valence_range': (0.3, 0.8),  # Not too sad, not too happy
            'mood_tags': [mood, 'calm', 'ambient', 'peaceful'],
            'duration_minutes': duration_minutes
        }

        print(f"😌 Creating {duration_minutes}-minute chill playlist ({mood} mood)...")

        # Find matching tracks
        candidate_tracks = await self.db.find_tracks_by_criteria({
            'bpm': {'$gte': criteria['bpm_range'][0], '$lte': criteria['bpm_range'][1]},
            'energy': {'$lte': criteria['energy_max']},
            'mood_tags': {'$in': criteria['mood_tags']}
        })

        # Create smooth transitions
        playlist = await self.generator.create_smooth_flow_playlist(
            tracks=candidate_tracks,
            duration_minutes=duration_minutes,
            transition_style='seamless'
        )

        playlist_data = {
            'name': f'Chill Vibes - {mood.title()}',
            'description': f'Relaxing {mood} music for unwinding',
            'tracks': playlist,
            'mood': mood,
            'created_at': datetime.now().isoformat()
        }

        await self.db.save_playlist(playlist_data)
        print(f"🧘 Created chill playlist with {len(playlist)} tracks")

        return playlist_data

    async def create_discovery_playlist(self, user_history: list, count: int = 20):
        """Create discovery playlist based on user listening history."""

        print(f"🔍 Creating discovery playlist with {count} new tracks...")

        # Analyze user preferences from history
        user_preferences = await self._analyze_user_preferences(user_history)

        # Find similar tracks user hasn't heard
        discovery_tracks = await self.matcher.find_similar_tracks(
            reference_tracks=user_preferences['favorite_tracks'],
            exclude_tracks=user_preferences['known_tracks'],
            similarity_threshold=0.7,
            max_results=count * 2  # Get extra to filter from
        )

        # Diversify selection
        diverse_selection = await self.generator.diversify_selection(
            tracks=discovery_tracks,
            target_count=count,
            diversification_factors=['artist', 'genre', 'year', 'energy']
        )

        playlist_data = {
            'name': 'Discover Weekly',
            'description': 'New music based on your listening history',
            'tracks': diverse_selection,
            'created_at': datetime.now().isoformat(),
            'user_preferences': user_preferences
        }

        await self.db.save_playlist(playlist_data)
        print(f"✨ Created discovery playlist with {len(diverse_selection)} tracks")

        return playlist_data

    async def _analyze_user_preferences(self, history: list):
        """Analyze user preferences from listening history."""

        # Get track details for listened tracks
        track_ids = [item['track_id'] for item in history if item.get('liked', False)]
        favorite_tracks = await self.db.get_tracks_by_ids(track_ids)

        # Calculate preference averages
        if favorite_tracks:
            avg_bpm = sum(track.bpm for track in favorite_tracks) / len(favorite_tracks)
            avg_energy = sum(track.energy for track in favorite_tracks) / len(favorite_tracks)
            common_genres = self._get_common_genres(favorite_tracks)

            return {
                'favorite_tracks': favorite_tracks,
                'known_tracks': [item['track_id'] for item in history],
                'avg_bpm': avg_bpm,
                'avg_energy': avg_energy,
                'preferred_genres': common_genres,
                'listening_count': len(history)
            }

        return {'favorite_tracks': [], 'known_tracks': []}

# Usage
async def create_playlists_example():
    creator = SmartPlaylistCreator("postgresql://localhost/tracktion")

    # Create different types of playlists
    workout_playlist = await creator.create_workout_playlist(45)
    chill_playlist = await creator.create_chill_playlist('peaceful', 30)

    # Discovery playlist based on mock user history
    user_history = [
        {'track_id': 'track1', 'liked': True, 'play_count': 15},
        {'track_id': 'track2', 'liked': True, 'play_count': 8},
        # ... more history data
    ]
    discovery_playlist = await creator.create_discovery_playlist(user_history, 25)

    print("🎉 All playlists created successfully!")

# Run example
asyncio.run(create_playlists_example())
```

### Playlist Export and Import

```python
"""Export and import playlists in various formats."""

from tracktion.utils import PlaylistExporter, PlaylistImporter
import json
import xml.etree.ElementTree as ET

class PlaylistManager:
    def __init__(self):
        self.exporter = PlaylistExporter()
        self.importer = PlaylistImporter()

    async def export_to_m3u(self, playlist_data: dict, output_path: str):
        """Export playlist to M3U format."""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            f.write(f"#PLAYLIST:{playlist_data['name']}\n")

            for track in playlist_data['tracks']:
                duration = int(track.get('duration', 0))
                artist = track.get('artist', 'Unknown Artist')
                title = track.get('title', 'Unknown Title')
                file_path = track.get('file_path', '')

                f.write(f"#EXTINF:{duration},{artist} - {title}\n")
                f.write(f"{file_path}\n")

        print(f"📁 Exported to M3U: {output_path}")

    async def export_to_spotify_csv(self, playlist_data: dict, output_path: str):
        """Export playlist in Spotify-compatible CSV format."""

        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Artist', 'Album', 'Track Name', 'Duration (ms)'])

            for track in playlist_data['tracks']:
                writer.writerow([
                    track.get('artist', ''),
                    track.get('album', ''),
                    track.get('title', ''),
                    track.get('duration_ms', 0)
                ])

        print(f"🎵 Exported Spotify CSV: {output_path}")

    async def import_from_m3u(self, file_path: str):
        """Import playlist from M3U file."""

        playlist_tracks = []
        current_track_info = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('#EXTINF:'):
                    # Parse track info
                    parts = line[8:].split(',', 1)
                    if len(parts) == 2:
                        duration = parts[0]
                        title_info = parts[1]

                        if ' - ' in title_info:
                            artist, title = title_info.split(' - ', 1)
                            current_track_info = {
                                'artist': artist.strip(),
                                'title': title.strip(),
                                'duration': int(duration) if duration.isdigit() else 0
                            }

                elif line and not line.startswith('#'):
                    # File path line
                    current_track_info['file_path'] = line
                    playlist_tracks.append(current_track_info.copy())
                    current_track_info = {}

        playlist_data = {
            'name': Path(file_path).stem,
            'source': 'M3U Import',
            'tracks': playlist_tracks,
            'imported_at': datetime.now().isoformat()
        }

        print(f"📂 Imported {len(playlist_tracks)} tracks from M3U")
        return playlist_data

# Usage example
async def playlist_management_example():
    manager = PlaylistManager()

    # Example playlist data
    playlist = {
        'name': 'My Awesome Playlist',
        'tracks': [
            {
                'artist': 'Artist 1',
                'title': 'Song 1',
                'album': 'Album 1',
                'duration': 240,
                'duration_ms': 240000,
                'file_path': '/music/artist1/song1.mp3'
            },
            # ... more tracks
        ]
    }

    # Export to different formats
    await manager.export_to_m3u(playlist, 'my_playlist.m3u')
    await manager.export_to_spotify_csv(playlist, 'my_playlist.csv')

    # Import from M3U
    imported_playlist = await manager.import_from_m3u('imported_playlist.m3u')

    print("✅ Playlist management operations completed")

asyncio.run(playlist_management_example())
```

---

## Database Operations

### Basic Database Queries

```python
"""Common database operations for music tracks."""

from tracktion.database import TrackDatabase
import asyncio
from datetime import datetime, timedelta

class TrackDatabaseManager:
    def __init__(self, database_url: str):
        self.db = TrackDatabase(database_url)

    async def add_track(self, track_data: dict):
        """Add new track to database."""

        # Validate required fields
        required_fields = ['file_path', 'title', 'artist']
        for field in required_fields:
            if field not in track_data:
                raise ValueError(f"Missing required field: {field}")

        # Add metadata
        track_data.update({
            'added_at': datetime.now(),
            'file_size': Path(track_data['file_path']).stat().st_size,
            'file_format': Path(track_data['file_path']).suffix.lower()
        })

        # Insert into database
        track_id = await self.db.insert_track(track_data)
        print(f"✅ Added track: {track_data['title']} (ID: {track_id})")

        return track_id

    async def find_tracks_by_bpm_range(self, min_bpm: float, max_bpm: float):
        """Find tracks within BPM range."""

        query = {
            'bpm': {'$gte': min_bpm, '$lte': max_bpm},
            'bpm_confidence': {'$gte': 0.7}  # Only confident BPM detections
        }

        tracks = await self.db.find_tracks(query)
        print(f"🎵 Found {len(tracks)} tracks between {min_bpm}-{max_bpm} BPM")

        return tracks

    async def find_similar_tracks(self, reference_track_id: str, limit: int = 10):
        """Find tracks similar to reference track."""

        # Get reference track
        reference_track = await self.db.get_track_by_id(reference_track_id)
        if not reference_track:
            raise ValueError(f"Track not found: {reference_track_id}")

        # Define similarity criteria
        bpm_tolerance = 10
        energy_tolerance = 0.2

        query = {
            'id': {'$ne': reference_track_id},  # Exclude reference track
            'bpm': {
                '$gte': reference_track.bpm - bpm_tolerance,
                '$lte': reference_track.bpm + bpm_tolerance
            },
            'energy': {
                '$gte': reference_track.energy - energy_tolerance,
                '$lte': reference_track.energy + energy_tolerance
            }
        }

        # Add key compatibility if available
        if reference_track.key:
            compatible_keys = self._get_compatible_keys(reference_track.key)
            query['key'] = {'$in': compatible_keys}

        similar_tracks = await self.db.find_tracks(query, limit=limit)
        print(f"🔍 Found {len(similar_tracks)} similar tracks to {reference_track.title}")

        return similar_tracks

    async def get_library_statistics(self):
        """Get comprehensive library statistics."""

        stats = {}

        # Basic counts
        stats['total_tracks'] = await self.db.count_tracks()
        stats['total_artists'] = await self.db.count_unique_artists()
        stats['total_albums'] = await self.db.count_unique_albums()

        # Analysis coverage
        stats['tracks_with_bpm'] = await self.db.count_tracks({'bpm': {'$ne': None}})
        stats['tracks_with_key'] = await self.db.count_tracks({'key': {'$ne': None}})
        stats['tracks_with_mood'] = await self.db.count_tracks({'mood_tags': {'$ne': []}})

        # Calculate percentages
        if stats['total_tracks'] > 0:
            stats['bpm_coverage'] = stats['tracks_with_bpm'] / stats['total_tracks'] * 100
            stats['key_coverage'] = stats['tracks_with_key'] / stats['total_tracks'] * 100
            stats['mood_coverage'] = stats['tracks_with_mood'] / stats['total_tracks'] * 100

        # BPM distribution
        bpm_ranges = [
            (0, 80, 'Very Slow'),
            (80, 100, 'Slow'),
            (100, 120, 'Moderate'),
            (120, 140, 'Fast'),
            (140, 180, 'Very Fast'),
            (180, 999, 'Extremely Fast')
        ]

        stats['bpm_distribution'] = {}
        for min_bpm, max_bpm, label in bpm_ranges:
            count = await self.db.count_tracks({
                'bpm': {'$gte': min_bpm, '$lt': max_bpm}
            })
            stats['bpm_distribution'][label] = count

        # Top genres
        stats['top_genres'] = await self.db.get_top_genres(limit=10)

        print("📊 Library Statistics:")
        print(f"  📁 Total Tracks: {stats['total_tracks']}")
        print(f"  👥 Artists: {stats['total_artists']}")
        print(f"  💿 Albums: {stats['total_albums']}")
        print(f"  🥁 BPM Coverage: {stats['bmp_coverage']:.1f}%")
        print(f"  🎹 Key Coverage: {stats['key_coverage']:.1f}%")
        print(f"  😊 Mood Coverage: {stats['mood_coverage']:.1f}%")

        return stats

    async def cleanup_orphaned_records(self):
        """Clean up tracks that no longer exist on filesystem."""

        print("🧹 Cleaning up orphaned records...")

        all_tracks = await self.db.find_tracks({})
        orphaned_count = 0

        for track in all_tracks:
            if not Path(track.file_path).exists():
                await self.db.delete_track(track.id)
                orphaned_count += 1
                print(f"🗑️ Removed orphaned track: {track.title}")

        print(f"✅ Cleanup complete. Removed {orphaned_count} orphaned records.")
        return orphaned_count

    def _get_compatible_keys(self, reference_key: str) -> list:
        """Get musically compatible keys."""
        # This is a simplified example - real implementation would use
        # Circle of Fifths and Camelot Wheel relationships
        key_relationships = {
            'C major': ['C major', 'A minor', 'G major', 'F major'],
            'A minor': ['A minor', 'C major', 'E minor', 'D minor'],
            # ... add more key relationships
        }

        return key_relationships.get(reference_key, [reference_key])

# Usage examples
async def database_operations_example():
    db_manager = TrackDatabaseManager("postgresql://localhost/tracktion")

    # Add new track
    new_track = {
        'file_path': '/music/artist/song.mp3',
        'title': 'Example Song',
        'artist': 'Example Artist',
        'album': 'Example Album',
        'bpm': 128.5,
        'key': 'C major',
        'energy': 0.8,
        'mood_tags': ['energetic', 'happy']
    }

    track_id = await db_manager.add_track(new_track)

    # Find tracks by BPM
    dance_tracks = await db_manager.find_tracks_by_bpm_range(120, 140)

    # Find similar tracks
    similar = await db_manager.find_similar_tracks(track_id, limit=5)

    # Get library stats
    stats = await db_manager.get_library_statistics()

    # Cleanup
    orphaned = await db_manager.cleanup_orphaned_records()

    print("✅ Database operations completed successfully!")

# Run example
asyncio.run(database_operations_example())
```

### Advanced Database Queries

```python
"""Advanced database queries and aggregations."""

async def advanced_queries_example():
    """Examples of advanced database queries."""

    db = TrackDatabase("postgresql://localhost/tracktion")

    # Complex aggregation query
    energy_by_genre = await db.aggregate([
        {"$group": {
            "_id": "$genre",
            "avg_energy": {"$avg": "$energy"},
            "avg_bpm": {"$avg": "$bpm"},
            "track_count": {"$sum": 1}
        }},
        {"$sort": {"avg_energy": -1}},
        {"$limit": 10}
    ])

    print("🎭 Energy by Genre:")
    for genre_data in energy_by_genre:
        print(f"  {genre_data['_id']}: {genre_data['avg_energy']:.2f} energy, "
              f"{genre_data['avg_bpm']:.1f} BPM, {genre_data['track_count']} tracks")

    # Temporal analysis
    tracks_by_month = await db.aggregate([
        {"$group": {
            "_id": {
                "year": {"$year": "$added_at"},
                "month": {"$month": "$added_at"}
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ])

    print("\n📅 Tracks Added by Month:")
    for period in tracks_by_month[-6:]:  # Last 6 months
        year, month = period['_id']['year'], period['_id']['month']
        count = period['count']
        print(f"  {year}-{month:02d}: {count} tracks")

asyncio.run(advanced_queries_example())
```

---

## File Organization

### Automatic File Organization

```python
"""Automatically organize music files based on metadata and analysis."""

from tracktion.file_organization import FileOrganizer
from tracktion.utils import MetadataExtractor
import shutil
from pathlib import Path

class MusicLibraryOrganizer:
    def __init__(self, source_path: str, organized_path: str):
        self.source_path = Path(source_path)
        self.organized_path = Path(organized_path)
        self.organizer = FileOrganizer()
        self.metadata_extractor = MetadataExtractor()

        # Create organized library structure
        self.organized_path.mkdir(exist_ok=True)

    async def organize_by_genre_and_year(self):
        """Organize files into Genre/Year/Artist - Album structure."""

        print(f"📁 Organizing files from {self.source_path} to {self.organized_path}")

        # Find all music files
        music_files = []
        for ext in ['.mp3', '.flac', '.wav', '.m4a']:
            music_files.extend(self.source_path.glob(f"**/*{ext}"))

        print(f"🎵 Found {len(music_files)} music files to organize")

        organized_count = 0
        error_count = 0

        for file_path in music_files:
            try:
                # Extract metadata
                metadata = await self.metadata_extractor.extract(str(file_path))

                # Determine organization path
                genre = self._clean_filename(metadata.get('genre', 'Unknown Genre'))
                year = metadata.get('year', 'Unknown Year')
                artist = self._clean_filename(metadata.get('artist', 'Unknown Artist'))
                album = self._clean_filename(metadata.get('album', 'Unknown Album'))
                title = self._clean_filename(metadata.get('title', file_path.stem))

                # Create directory structure: Genre/Year/Artist - Album/
                target_dir = self.organized_path / genre / str(year) / f"{artist} - {album}"
                target_dir.mkdir(parents=True, exist_ok=True)

                # Generate target filename
                track_num = metadata.get('track_number', '')
                if track_num:
                    filename = f"{track_num:02d}. {artist} - {title}{file_path.suffix}"
                else:
                    filename = f"{artist} - {title}{file_path.suffix}"

                target_path = target_dir / filename

                # Move file (with duplicate handling)
                target_path = self._handle_duplicates(target_path)
                shutil.move(str(file_path), str(target_path))

                print(f"📂 Moved: {file_path.name} → {target_path.relative_to(self.organized_path)}")
                organized_count += 1

            except Exception as e:
                print(f"❌ Error organizing {file_path.name}: {e}")
                error_count += 1

        print(f"\n📊 Organization Summary:")
        print(f"  ✅ Successfully organized: {organized_count}")
        print(f"  ❌ Errors: {error_count}")

        return organized_count, error_count

    async def organize_by_bpm_ranges(self):
        """Organize files by BPM ranges for DJ use."""

        from tracktion.analysis import BPMDetector

        bmp_detector = BPMDetector()
        bpm_ranges = [
            (0, 90, "Slow (0-90 BPM)"),
            (90, 110, "Moderate (90-110 BPM)"),
            (110, 130, "Dance (110-130 BPM)"),
            (130, 150, "Fast (130-150 BPM)"),
            (150, 999, "Very Fast (150+ BPM)")
        ]

        print("🥁 Organizing by BPM ranges...")

        # Create BPM directories
        for min_bpm, max_bpm, label in bpm_ranges:
            (self.organized_path / label).mkdir(exist_ok=True)

        # Process files
        music_files = list(self.source_path.glob("**/*.mp3"))

        for file_path in music_files:
            try:
                # Detect BPM
                bpm_result = await bpm_detector.analyze(str(file_path))

                if bpm_result.confidence > 0.7:  # Only organize confident detections
                    # Find appropriate BPM range
                    target_dir = None
                    for min_bpm, max_bpm, label in bpm_ranges:
                        if min_bpm <= bpm_result.bpm < max_bpm:
                            target_dir = self.organized_path / label
                            break

                    if target_dir:
                        target_path = target_dir / f"{bpm_result.bpm:.0f} - {file_path.name}"
                        shutil.move(str(file_path), str(target_path))
                        print(f"🎵 {file_path.name} → {target_dir.name}")
                else:
                    print(f"⚠️ Low confidence BPM for {file_path.name}, skipping")

            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")

    def _clean_filename(self, text: str) -> str:
        """Clean text for use in filenames."""
        # Remove invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '')

        # Limit length and strip whitespace
        return text.strip()[:100]

    def _handle_duplicates(self, target_path: Path) -> Path:
        """Handle duplicate filenames by adding numbers."""
        if not target_path.exists():
            return target_path

        counter = 1
        while True:
            stem = target_path.stem
            suffix = target_path.suffix
            parent = target_path.parent

            new_path = parent / f"{stem} ({counter}){suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

# Usage
async def organize_music_library():
    organizer = MusicLibraryOrganizer(
        source_path="/path/to/messy/music",
        organized_path="/path/to/organized/music"
    )

    # Organize by genre and year
    await organizer.organize_by_genre_and_year()

    # Alternative: organize by BPM for DJ use
    # await organizer.organize_by_bpm_ranges()

    print("🎉 Music library organization complete!")

asyncio.run(organize_music_library())
```

### Duplicate Detection and Management

```python
"""Detect and manage duplicate music files."""

from tracktion.utils import DuplicateDetector, AudioFingerprinter
import hashlib
from collections import defaultdict

class DuplicateManager:
    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.detector = DuplicateDetector()
        self.fingerprinter = AudioFingerprinter()

    async def find_duplicates(self, method: str = 'fingerprint'):
        """Find duplicate files using various methods."""

        print(f"🔍 Scanning for duplicates using {method} method...")

        if method == 'hash':
            duplicates = await self._find_by_file_hash()
        elif method == 'fingerprint':
            duplicates = await self._find_by_audio_fingerprint()
        elif method == 'metadata':
            duplicates = await self._find_by_metadata()
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"📊 Found {len(duplicates)} sets of duplicates")
        return duplicates

    async def _find_by_file_hash(self):
        """Find duplicates by file hash (exact matches)."""

        hash_to_files = defaultdict(list)

        # Calculate hash for each file
        for file_path in self.library_path.glob("**/*.mp3"):
            file_hash = self._calculate_file_hash(file_path)
            hash_to_files[file_hash].append(file_path)

        # Return groups with more than one file
        duplicates = []
        for file_hash, files in hash_to_files.items():
            if len(files) > 1:
                duplicates.append({
                    'method': 'file_hash',
                    'hash': file_hash,
                    'files': files,
                    'confidence': 1.0  # Exact matches
                })

        return duplicates

    async def _find_by_audio_fingerprint(self):
        """Find duplicates by audio fingerprint (content-based)."""

        fingerprint_to_files = defaultdict(list)

        # Generate fingerprints
        for file_path in self.library_path.glob("**/*.mp3"):
            try:
                fingerprint = await self.fingerprinter.generate(str(file_path))
                fingerprint_to_files[fingerprint].append(file_path)
            except Exception as e:
                print(f"⚠️ Fingerprint failed for {file_path.name}: {e}")

        # Find similar fingerprints
        duplicates = []
        processed_fingerprints = set()

        for fingerprint, files in fingerprint_to_files.items():
            if fingerprint in processed_fingerprints:
                continue

            # Find similar fingerprints
            similar_groups = [files]
            for other_fingerprint, other_files in fingerprint_to_files.items():
                if other_fingerprint != fingerprint and other_fingerprint not in processed_fingerprints:
                    similarity = self.fingerprinter.compare(fingerprint, other_fingerprint)
                    if similarity > 0.9:  # Very similar
                        similar_groups.append(other_files)
                        processed_fingerprints.add(other_fingerprint)

            if len(similar_groups) > 1 or len(files) > 1:
                all_files = [file for group in similar_groups for file in group]
                duplicates.append({
                    'method': 'audio_fingerprint',
                    'files': all_files,
                    'confidence': similarity if 'similarity' in locals() else 1.0
                })

            processed_fingerprints.add(fingerprint)

        return duplicates

    async def _find_by_metadata(self):
        """Find duplicates by metadata (title, artist, duration)."""

        from tracktion.utils import MetadataExtractor

        extractor = MetadataExtractor()
        metadata_to_files = defaultdict(list)

        # Extract metadata for each file
        for file_path in self.library_path.glob("**/*.mp3"):
            try:
                metadata = await extractor.extract(str(file_path))

                # Create metadata key
                key = (
                    metadata.get('title', '').lower().strip(),
                    metadata.get('artist', '').lower().strip(),
                    round(metadata.get('duration', 0))  # Round duration to nearest second
                )

                metadata_to_files[key].append({
                    'file_path': file_path,
                    'metadata': metadata
                })

            except Exception as e:
                print(f"⚠️ Metadata extraction failed for {file_path.name}: {e}")

        # Find groups with matching metadata
        duplicates = []
        for key, file_data in metadata_to_files.items():
            if len(file_data) > 1:
                duplicates.append({
                    'method': 'metadata',
                    'metadata_key': key,
                    'files': [item['file_path'] for item in file_data],
                    'confidence': 0.8  # Lower confidence for metadata matches
                })

        return duplicates

    async def resolve_duplicates(self, duplicates: list, strategy: str = 'keep_best_quality'):
        """Resolve duplicates using specified strategy."""

        print(f"🔧 Resolving duplicates using '{strategy}' strategy...")

        resolved_count = 0

        for duplicate_group in duplicates:
            files = duplicate_group['files']

            if strategy == 'keep_best_quality':
                keeper = await self._select_best_quality_file(files)
            elif strategy == 'keep_newest':
                keeper = max(files, key=lambda f: f.stat().st_mtime)
            elif strategy == 'keep_largest':
                keeper = max(files, key=lambda f: f.stat().st_size)
            elif strategy == 'interactive':
                keeper = await self._interactive_selection(files)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Remove duplicates (keep the selected file)
            for file_path in files:
                if file_path != keeper:
                    print(f"🗑️ Removing duplicate: {file_path.name}")
                    file_path.unlink()  # Delete file
                    resolved_count += 1

        print(f"✅ Resolved {resolved_count} duplicate files")
        return resolved_count

    async def _select_best_quality_file(self, files: list):
        """Select file with best audio quality."""

        best_file = files[0]
        best_score = 0

        for file_path in files:
            try:
                # Calculate quality score based on file size and format
                size_mb = file_path.stat().st_size / (1024 * 1024)
                format_score = {'.flac': 10, '.wav': 9, '.mp3': 5, '.m4a': 4}.get(file_path.suffix.lower(), 1)

                # Prefer larger files and better formats
                quality_score = format_score * 2 + min(size_mb / 10, 5)

                if quality_score > best_score:
                    best_score = quality_score
                    best_file = file_path

            except Exception as e:
                print(f"⚠️ Quality assessment failed for {file_path.name}: {e}")

        print(f"🏆 Selected best quality: {best_file.name}")
        return best_file

    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192):
        """Calculate SHA-256 hash of file."""

        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

# Usage
async def duplicate_management_example():
    manager = DuplicateManager("/path/to/music/library")

    # Find duplicates using different methods
    hash_duplicates = await manager.find_duplicates('hash')
    fingerprint_duplicates = await manager.find_duplicates('fingerprint')

    # Resolve duplicates
    if hash_duplicates:
        await manager.resolve_duplicates(hash_duplicates, 'keep_best_quality')

    print("🧹 Duplicate cleanup completed!")

asyncio.run(duplicate_management_example())
```

This comprehensive guide provides practical, working code examples for the most common Tracktion operations. Each example includes proper error handling, progress tracking, and real-world considerations for production use.

The examples can be adapted and combined based on your specific needs. For more advanced usage patterns, refer to the API documentation and integration examples.

---

**Next:** [Integration Examples](integration-examples.md) | **Previous:** [Use Cases](example-use-cases.md)
