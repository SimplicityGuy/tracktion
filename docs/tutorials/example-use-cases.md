# Example Use Cases

This guide showcases real-world scenarios and practical examples of using Tracktion to manage your music library efficiently.

## Overview

Tracktion is designed to handle various music management workflows, from simple library organization to complex automated processing systems. These examples demonstrate how to use Tracktion's features for different use cases.

## Use Case 1: Personal Music Library Organization

**Scenario:** You have a large collection of digital music files in various formats and want to organize them automatically.

### Setup

```bash
# Clone and set up Tracktion
git clone https://github.com/your-org/tracktion.git
cd tracktion

# Install dependencies
uv sync

# Configure for personal use
cp .env.example .env
```

### Configuration

```bash
# .env configuration for personal library
DATABASE_URL=postgresql://localhost/tracktion_personal
MUSIC_LIBRARY_PATH=/Users/yourname/Music
ANALYSIS_ENABLED=true
AUTO_ORGANIZE=true
DUPLICATE_DETECTION=true

# File organization settings
ORGANIZE_BY_GENRE=true
ORGANIZE_BY_YEAR=true
CREATE_PLAYLISTS=true
```

### Running the System

```python
#!/usr/bin/env python3
"""Personal music library organizer example."""

from tracktion.analysis import BPMDetector, KeyDetector
from tracktion.file_watcher import LibraryWatcher
from tracktion.tracklist import PlaylistGenerator
from pathlib import Path
import asyncio

async def organize_personal_library():
    """Organize personal music library with automated analysis."""

    # Set up library path
    library_path = Path("/Users/yourname/Music")

    # Initialize analyzers
    bpm_detector = BPMDetector(confidence_threshold=0.8)
    key_detector = KeyDetector(algorithm='hpcp')
    playlist_generator = PlaylistGenerator()

    # Set up file watcher for new additions
    watcher = LibraryWatcher(
        watch_path=library_path,
        auto_analyze=True,
        auto_organize=True
    )

    print("🎵 Starting personal library organization...")

    # Process existing files
    music_files = list(library_path.glob("**/*.mp3"))
    music_files.extend(library_path.glob("**/*.flac"))
    music_files.extend(library_path.glob("**/*.wav"))

    for file_path in music_files:
        print(f"📁 Processing: {file_path.name}")

        # Analyze audio properties
        bpm_result = await bpm_detector.analyze(str(file_path))
        key_result = await key_detector.analyze(str(file_path))

        # Update metadata
        metadata = {
            'bpm': bpm_result.bpm,
            'key': key_result.key,
            'confidence': min(bpm_result.confidence, key_result.confidence)
        }

        print(f"  🎶 BPM: {metadata['bpm']}, Key: {metadata['key']}")

        # Organize file based on metadata
        await organize_file(file_path, metadata)

    # Generate playlists based on analysis
    await generate_smart_playlists(playlist_generator, library_path)

    # Start watching for new files
    print("👀 Starting file watcher for new additions...")
    await watcher.start_watching()

async def organize_file(file_path: Path, metadata: dict):
    """Organize file into appropriate directory structure."""
    # Implementation details for file organization
    pass

async def generate_smart_playlists(generator, library_path: Path):
    """Generate smart playlists based on audio analysis."""
    playlists = [
        {"name": "High Energy", "bpm_min": 120, "key_filter": None},
        {"name": "Chill Vibes", "bpm_max": 100, "key_filter": None},
        {"name": "Dance Floor", "bpm_min": 128, "bpm_max": 140, "key_filter": None}
    ]

    for playlist_config in playlists:
        tracks = await generator.create_playlist(
            name=playlist_config["name"],
            criteria=playlist_config,
            library_path=str(library_path)
        )
        print(f"📋 Created playlist '{playlist_config['name']}' with {len(tracks)} tracks")

if __name__ == "__main__":
    asyncio.run(organize_personal_library())
```

### Expected Results

```bash
🎵 Starting personal library organization...
📁 Processing: song1.mp3
  🎶 BPM: 128.5, Key: C major
📁 Processing: song2.flac
  🎶 BPM: 95.2, Key: G minor
📋 Created playlist 'High Energy' with 45 tracks
📋 Created playlist 'Chill Vibes' with 23 tracks
📋 Created playlist 'Dance Floor' with 67 tracks
👀 Starting file watcher for new additions...
```

---

## Use Case 2: DJ Performance Preparation

**Scenario:** A DJ wants to analyze their music collection for BPM and key compatibility to create seamless mix sets.

### Setup

```python
"""DJ set preparation with harmonic mixing support."""

from tracktion.analysis import BPMDetector, KeyDetector, HarmonicAnalyzer
from tracktion.tracklist import DJSetGenerator
from tracktion.utils import CamelotWheel
import pandas as pd

class DJSetPreparation:
    def __init__(self, music_library_path: str):
        self.library_path = music_library_path
        self.bpm_detector = BPMDetector(
            confidence_threshold=0.9,  # Higher confidence for DJ use
            algorithm='multiband'      # More accurate for dance music
        )
        self.key_detector = KeyDetector(
            algorithm='hpcp',
            use_camelot_notation=True  # DJ-friendly key notation
        )
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.set_generator = DJSetGenerator()

    async def analyze_library_for_dj_use(self):
        """Analyze entire library with DJ-specific requirements."""

        # Get all dance music files
        dance_files = await self._get_dance_music_files()
        analysis_results = []

        print(f"🎧 Analyzing {len(dance_files)} tracks for DJ use...")

        for file_path in dance_files:
            try:
                # Perform comprehensive analysis
                bpm_result = await self.bpm_detector.analyze(file_path)
                key_result = await self.key_detector.analyze(file_path)
                harmonic_data = await self.harmonic_analyzer.analyze(file_path)

                # DJ-specific metadata
                track_data = {
                    'file_path': file_path,
                    'bpm': bpm_result.bpm,
                    'bpm_confidence': bpm_result.confidence,
                    'key': key_result.key,
                    'camelot_key': key_result.camelot_notation,
                    'key_confidence': key_result.confidence,
                    'energy_level': harmonic_data.energy_level,
                    'danceability': harmonic_data.danceability,
                    'compatible_keys': CamelotWheel.get_compatible_keys(key_result.camelot_notation)
                }

                analysis_results.append(track_data)
                print(f"  ✅ {Path(file_path).name} - {track_data['bpm']} BPM, {track_data['camelot_key']}")

            except Exception as e:
                print(f"  ❌ Error analyzing {file_path}: {e}")

        # Save results for DJ software import
        df = pd.DataFrame(analysis_results)
        df.to_csv('dj_library_analysis.csv', index=False)
        print(f"💾 Analysis saved to dj_library_analysis.csv")

        return analysis_results

    async def create_harmonic_mix_set(self, target_duration: int = 60, energy_progression: str = "build"):
        """Create a harmonically mixed DJ set."""

        # Load analyzed tracks
        df = pd.read_csv('dj_library_analysis.csv')

        # Filter high-confidence tracks
        quality_tracks = df[
            (df['bpm_confidence'] > 0.8) &
            (df['key_confidence'] > 0.8) &
            (df['danceability'] > 0.7)
        ].copy()

        print(f"🎛️ Creating {target_duration}-minute harmonic mix set...")

        # Generate set based on harmonic mixing rules
        mix_set = await self.set_generator.create_harmonic_set(
            tracks=quality_tracks.to_dict('records'),
            duration_minutes=target_duration,
            energy_progression=energy_progression,
            allow_key_jumps=True,
            max_bpm_change=6  # Maximum BPM change between tracks
        )

        # Export set as playlist
        await self._export_dj_set(mix_set, f"harmonic_set_{energy_progression}_{target_duration}min")

        return mix_set

    async def _get_dance_music_files(self):
        """Get dance music files from library."""
        dance_extensions = ['.mp3', '.flac', '.wav', '.aiff']
        dance_genres = ['house', 'techno', 'trance', 'electronic', 'edm', 'dance']

        # Implementation would scan library for dance music
        # This is a simplified example
        library_path = Path(self.library_path)
        return list(library_path.glob("**/*.mp3"))  # Simplified

    async def _export_dj_set(self, mix_set: list, playlist_name: str):
        """Export DJ set in multiple formats."""

        # Export as M3U playlist
        with open(f"{playlist_name}.m3u", 'w') as f:
            f.write("#EXTM3U\n")
            for track in mix_set:
                f.write(f"#EXTINF:-1,{track['artist']} - {track['title']}\n")
                f.write(f"{track['file_path']}\n")

        # Export as Serato/Rekordbox compatible CSV
        df = pd.DataFrame(mix_set)
        df.to_csv(f"{playlist_name}_dj_import.csv", index=False)

        print(f"📁 Exported DJ set as {playlist_name}.m3u and {playlist_name}_dj_import.csv")

# Usage example
async def prepare_dj_set():
    """Example DJ set preparation workflow."""

    dj_prep = DJSetPreparation("/path/to/dj/music/library")

    # Step 1: Analyze library
    analysis_results = await dj_prep.analyze_library_for_dj_use()

    # Step 2: Create different types of sets
    opening_set = await dj_prep.create_harmonic_mix_set(
        target_duration=30,
        energy_progression="gradual_build"
    )

    peak_time_set = await dj_prep.create_harmonic_mix_set(
        target_duration=45,
        energy_progression="high_energy"
    )

    closing_set = await dj_prep.create_harmonic_mix_set(
        target_duration=30,
        energy_progression="wind_down"
    )

    print("🎉 DJ preparation complete!")
    print(f"📊 Library analyzed: {len(analysis_results)} tracks")
    print("🎵 Created 3 harmonic mix sets ready for performance")

if __name__ == "__main__":
    import asyncio
    asyncio.run(prepare_dj_set())
```

### Key Features for DJs

- **High-accuracy BPM detection** for beatmatching
- **Camelot wheel notation** for harmonic mixing
- **Energy level analysis** for set progression
- **Compatible key suggestions** for smooth transitions
- **Export formats** for popular DJ software

---

## Use Case 3: Music Production Studio Integration

**Scenario:** A music production studio wants to integrate Tracktion into their workflow for sample organization and project management.

### Studio Integration Setup

```python
"""Music production studio integration example."""

from tracktion.analysis import BPMDetector, KeyDetector, SpectrumAnalyzer
from tracktion.cataloging import SampleLibraryManager
from tracktion.tracklist import ProjectManager
from pathlib import Path
import json

class StudioWorkflowIntegration:
    def __init__(self, studio_config: dict):
        self.config = studio_config
        self.sample_manager = SampleLibraryManager()
        self.project_manager = ProjectManager()
        self.analyzer_suite = self._setup_analyzers()

    def _setup_analyzers(self):
        """Set up analysis suite optimized for production use."""
        return {
            'bpm': BPMDetector(
                confidence_threshold=0.95,
                algorithm='percussive_onset',  # Best for samples
                enable_subdivision_detection=True  # Detect half/double time
            ),
            'key': KeyDetector(
                algorithm='hpcp',
                chromagram_resolution=12,  # High resolution for accuracy
                enable_mode_detection=True  # Major/minor detection
            ),
            'spectrum': SpectrumAnalyzer(
                fft_size=4096,
                enable_spectral_features=True,
                detect_dominant_frequencies=True
            )
        }

    async def organize_sample_library(self, library_path: str):
        """Organize and analyze sample library for production use."""

        print("🎚️ Organizing sample library for production...")

        sample_path = Path(library_path)
        categories = {
            'drums': ['kick', 'snare', 'hihat', 'perc', 'drum'],
            'bass': ['bass', 'sub', 'low'],
            'melody': ['lead', 'melody', 'synth', 'keys'],
            'fx': ['fx', 'sweep', 'riser', 'impact'],
            'vocal': ['vocal', 'voice', 'vox', 'acapella']
        }

        organized_samples = {}

        # Process each category
        for category, keywords in categories.items():
            category_samples = []

            # Find samples matching category
            for keyword in keywords:
                matching_files = list(sample_path.glob(f"**/*{keyword}*.wav"))
                matching_files.extend(sample_path.glob(f"**/*{keyword}*.aiff"))

                for sample_file in matching_files:
                    # Analyze sample
                    analysis = await self._analyze_sample(sample_file)

                    sample_data = {
                        'file_path': str(sample_file),
                        'category': category,
                        'keyword': keyword,
                        'bpm': analysis.get('bpm'),
                        'key': analysis.get('key'),
                        'dominant_freq': analysis.get('dominant_frequency'),
                        'spectral_centroid': analysis.get('spectral_centroid'),
                        'suitable_genres': self._suggest_genres(analysis),
                        'production_tags': self._generate_production_tags(analysis)
                    }

                    category_samples.append(sample_data)
                    print(f"  📁 {category.upper()}: {sample_file.name}")

            organized_samples[category] = category_samples

        # Save organized library index
        with open('sample_library_index.json', 'w') as f:
            json.dump(organized_samples, f, indent=2)

        print(f"💾 Sample library organized - {sum(len(cat) for cat in organized_samples.values())} samples indexed")
        return organized_samples

    async def create_project_template(self, project_name: str, genre: str, target_bpm: int):
        """Create a new project with suggested samples and settings."""

        print(f"🎼 Creating project template: {project_name} ({genre} @ {target_bpm} BPM)")

        # Load sample library index
        with open('sample_library_index.json', 'r') as f:
            sample_library = json.load(f)

        # Find compatible samples
        compatible_samples = self._find_compatible_samples(
            sample_library, genre, target_bpm
        )

        # Create project structure
        project_data = {
            'name': project_name,
            'genre': genre,
            'bpm': target_bpm,
            'created_date': str(datetime.now()),
            'suggested_samples': compatible_samples,
            'daw_settings': {
                'tempo': target_bpm,
                'time_signature': '4/4',
                'key_signature': self._suggest_key_for_genre(genre),
                'sample_rate': 44100,
                'bit_depth': 24
            },
            'recommended_effects': self._get_genre_effects(genre),
            'arrangement_template': self._get_arrangement_template(genre)
        }

        # Save project template
        project_dir = Path(f"projects/{project_name}")
        project_dir.mkdir(parents=True, exist_ok=True)

        with open(project_dir / "project_template.json", 'w') as f:
            json.dump(project_data, f, indent=2)

        # Create sample folders
        for category in compatible_samples:
            (project_dir / category).mkdir(exist_ok=True)

        print(f"📂 Project template created at {project_dir}")
        return project_data

    async def _analyze_sample(self, sample_file: Path):
        """Comprehensive sample analysis."""
        try:
            # Run all analyzers
            bpm_result = await self.analyzer_suite['bpm'].analyze(str(sample_file))
            key_result = await self.analyzer_suite['key'].analyze(str(sample_file))
            spectrum_result = await self.analyzer_suite['spectrum'].analyze(str(sample_file))

            return {
                'bpm': bpm_result.bpm if bpm_result.confidence > 0.7 else None,
                'key': key_result.key if key_result.confidence > 0.6 else None,
                'dominant_frequency': spectrum_result.dominant_frequency,
                'spectral_centroid': spectrum_result.spectral_centroid,
                'spectral_features': spectrum_result.features
            }
        except Exception as e:
            print(f"⚠️ Analysis failed for {sample_file}: {e}")
            return {}

    def _find_compatible_samples(self, library: dict, genre: str, target_bpm: int):
        """Find samples compatible with project requirements."""
        compatible = {}
        bpm_tolerance = 10  # ±10 BPM tolerance

        for category, samples in library.items():
            compatible_samples = []

            for sample in samples:
                sample_bpm = sample.get('bpm')

                # BPM compatibility check
                if sample_bpm and abs(sample_bpm - target_bpm) <= bpm_tolerance:
                    compatible_samples.append(sample)
                elif not sample_bpm:  # One-shots without tempo
                    compatible_samples.append(sample)

            compatible[category] = compatible_samples[:10]  # Limit to top 10 per category

        return compatible

    def _suggest_genres(self, analysis: dict) -> list:
        """Suggest suitable genres based on analysis."""
        genres = []

        if analysis.get('bpm'):
            bpm = analysis['bpm']
            if 60 <= bpm <= 90:
                genres.extend(['hip-hop', 'trap', 'lo-fi'])
            elif 90 <= bpm <= 110:
                genres.extend(['house', 'deep-house', 'tech-house'])
            elif 110 <= bpm <= 130:
                genres.extend(['techno', 'progressive'])
            elif 130 <= bpm <= 150:
                genres.extend(['trance', 'hardstyle'])

        return genres[:3]  # Return top 3 suggestions

    def _generate_production_tags(self, analysis: dict) -> list:
        """Generate production-relevant tags."""
        tags = []

        if analysis.get('dominant_frequency'):
            freq = analysis['dominant_frequency']
            if freq < 200:
                tags.append('sub-bass')
            elif freq < 500:
                tags.append('bass')
            elif freq < 2000:
                tags.append('mid-range')
            else:
                tags.append('high-freq')

        if analysis.get('spectral_centroid'):
            centroid = analysis['spectral_centroid']
            if centroid > 3000:
                tags.append('bright')
            else:
                tags.append('warm')

        return tags

# Usage example
async def studio_integration_example():
    """Example studio workflow integration."""

    studio_config = {
        'sample_library_path': '/studio/samples',
        'project_path': '/studio/projects',
        'export_formats': ['wav', 'aiff', 'flac'],
        'default_sample_rate': 44100
    }

    studio = StudioWorkflowIntegration(studio_config)

    # Organize sample library
    library = await studio.organize_sample_library('/studio/samples')

    # Create project templates for different genres
    house_project = await studio.create_project_template(
        "Deep House Track 01", "deep-house", 124
    )

    techno_project = await studio.create_project_template(
        "Techno Banger 01", "techno", 132
    )

    print("🎛️ Studio integration complete!")
    print(f"📊 Sample library organized with {sum(len(cat) for cat in library.values())} samples")
    print("🎼 Project templates created and ready for production")

if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    asyncio.run(studio_integration_example())
```

### Studio Integration Benefits

- **Automated sample organization** by instrument and characteristics
- **BPM and key matching** for harmonic compatibility
- **Project template generation** with suggested samples
- **Genre-specific recommendations** based on analysis
- **DAW integration ready** with standard export formats

---

## Use Case 4: Music Streaming Service Backend

**Scenario:** Building a music streaming service backend that needs to analyze and categorize uploaded content.

### Streaming Service Implementation

```python
"""Music streaming service backend integration."""

from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer
from tracktion.cataloging import ContentCategorizer
from tracktion.utils import AudioQualityChecker
import asyncio
from typing import Dict, List
import uuid

class StreamingServiceBackend:
    def __init__(self):
        self.content_analyzers = self._setup_analyzers()
        self.categorizer = ContentCategorizer()
        self.quality_checker = AudioQualityChecker()

    def _setup_analyzers(self):
        """Setup analyzers optimized for streaming service needs."""
        return {
            'bpm': BPMDetector(
                confidence_threshold=0.8,
                batch_processing=True  # For handling multiple uploads
            ),
            'key': KeyDetector(
                algorithm='chroma_cens',  # Good for various music types
                batch_processing=True
            ),
            'mood': MoodAnalyzer(
                model='streaming_optimized',
                categories=['happy', 'sad', 'energetic', 'calm', 'aggressive', 'romantic']
            )
        }

    async def process_uploaded_track(self, file_path: str, metadata: Dict) -> Dict:
        """Process newly uploaded track for streaming service."""

        track_id = str(uuid.uuid4())
        print(f"🎵 Processing upload: {metadata.get('title', 'Unknown')} (ID: {track_id})")

        try:
            # Step 1: Quality check
            quality_check = await self.quality_checker.analyze(file_path)
            if not quality_check.meets_streaming_standards:
                return {
                    'status': 'rejected',
                    'reason': 'Audio quality below streaming standards',
                    'details': quality_check.issues
                }

            # Step 2: Comprehensive audio analysis
            analysis_results = await self._perform_comprehensive_analysis(file_path)

            # Step 3: Content categorization
            categories = await self.categorizer.categorize_content(
                file_path, analysis_results, metadata
            )

            # Step 4: Generate streaming metadata
            streaming_metadata = {
                'track_id': track_id,
                'original_metadata': metadata,
                'audio_features': {
                    'bpm': analysis_results.get('bpm'),
                    'key': analysis_results.get('key'),
                    'energy_level': analysis_results.get('energy'),
                    'danceability': analysis_results.get('danceability'),
                    'valence': analysis_results.get('valence'),
                    'mood_tags': analysis_results.get('mood_tags', [])
                },
                'categories': categories,
                'quality_metrics': {
                    'bitrate': quality_check.bitrate,
                    'sample_rate': quality_check.sample_rate,
                    'dynamic_range': quality_check.dynamic_range,
                    'loudness_lufs': quality_check.loudness_lufs
                },
                'recommendations': {
                    'similar_tracks': await self._find_similar_tracks(analysis_results),
                    'playlist_suggestions': await self._suggest_playlists(analysis_results, categories),
                    'radio_compatibility': self._assess_radio_compatibility(analysis_results)
                },
                'processing_timestamp': datetime.now().isoformat(),
                'status': 'processed_successfully'
            }

            # Step 5: Store in content database
            await self._store_track_metadata(streaming_metadata)

            print(f"✅ Track processed successfully: {metadata.get('title')}")
            return streaming_metadata

        except Exception as e:
            print(f"❌ Processing failed for {metadata.get('title', 'Unknown')}: {e}")
            return {
                'status': 'processing_failed',
                'error': str(e),
                'track_id': track_id
            }

    async def batch_process_library(self, library_path: str) -> Dict:
        """Process entire music library for streaming service."""

        library_stats = {
            'total_files': 0,
            'processed_successfully': 0,
            'processing_failed': 0,
            'quality_rejected': 0,
            'processing_time': 0
        }

        start_time = time.time()
        music_files = self._get_music_files(library_path)
        library_stats['total_files'] = len(music_files)

        print(f"🚀 Starting batch processing of {len(music_files)} tracks...")

        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, len(music_files), batch_size):
            batch = music_files[i:i + batch_size]
            batch_tasks = []

            for file_path in batch:
                # Extract basic metadata
                metadata = await self._extract_basic_metadata(file_path)
                task = self.process_uploaded_track(file_path, metadata)
                batch_tasks.append(task)

            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Update statistics
            for result in batch_results:
                if isinstance(result, Exception):
                    library_stats['processing_failed'] += 1
                elif result.get('status') == 'rejected':
                    library_stats['quality_rejected'] += 1
                elif result.get('status') == 'processed_successfully':
                    library_stats['processed_successfully'] += 1
                else:
                    library_stats['processing_failed'] += 1

        library_stats['processing_time'] = time.time() - start_time

        print(f"📊 Batch processing complete:")
        print(f"  ✅ Successfully processed: {library_stats['processed_successfully']}")
        print(f"  ❌ Processing failed: {library_stats['processing_failed']}")
        print(f"  🚫 Quality rejected: {library_stats['quality_rejected']}")
        print(f"  ⏱️ Total time: {library_stats['processing_time']:.2f} seconds")

        return library_stats

    async def generate_personalized_recommendations(self, user_id: str, listening_history: List[Dict]) -> Dict:
        """Generate personalized music recommendations."""

        print(f"🎯 Generating recommendations for user {user_id}...")

        # Analyze user's listening patterns
        user_preferences = await self._analyze_user_preferences(listening_history)

        # Find similar tracks based on audio features
        similar_tracks = await self._find_tracks_by_audio_features(user_preferences['audio_features'])

        # Generate different types of recommendations
        recommendations = {
            'discover_weekly': await self._generate_discovery_playlist(user_preferences),
            'similar_artists': await self._find_similar_artists(user_preferences['favorite_artists']),
            'mood_based': await self._generate_mood_playlists(user_preferences['mood_preferences']),
            'genre_exploration': await self._suggest_genre_expansion(user_preferences['genres']),
            'release_radar': await self._get_new_releases_for_user(user_preferences)
        }

        print(f"📝 Generated {sum(len(rec) for rec in recommendations.values())} recommendations")
        return recommendations

    async def _perform_comprehensive_analysis(self, file_path: str) -> Dict:
        """Perform comprehensive audio analysis."""

        # Run all analyzers concurrently
        analysis_tasks = {
            'bpm': self.content_analyzers['bpm'].analyze(file_path),
            'key': self.content_analyzers['key'].analyze(file_path),
            'mood': self.content_analyzers['mood'].analyze(file_path)
        }

        results = await asyncio.gather(*analysis_tasks.values(), return_exceptions=True)
        analysis_data = {}

        for analyzer_name, result in zip(analysis_tasks.keys(), results):
            if isinstance(result, Exception):
                print(f"⚠️ {analyzer_name} analysis failed: {result}")
                continue

            if analyzer_name == 'bpm':
                analysis_data.update({
                    'bpm': result.bpm,
                    'bpm_confidence': result.confidence,
                    'tempo_stability': result.stability
                })
            elif analyzer_name == 'key':
                analysis_data.update({
                    'key': result.key,
                    'key_confidence': result.confidence,
                    'mode': result.mode
                })
            elif analyzer_name == 'mood':
                analysis_data.update({
                    'mood_tags': result.moods,
                    'energy': result.energy,
                    'valence': result.valence,
                    'danceability': result.danceability
                })

        return analysis_data

# Usage example for streaming service
async def streaming_service_example():
    """Example streaming service backend usage."""

    backend = StreamingServiceBackend()

    # Process single uploaded track
    track_metadata = {
        'title': 'Summer Vibes',
        'artist': 'Electronic Artist',
        'album': 'Chill Collection',
        'genre': 'Electronic'
    }

    result = await backend.process_uploaded_track(
        '/uploads/summer_vibes.mp3',
        track_metadata
    )
    print(f"Single track processing result: {result['status']}")

    # Batch process music library
    batch_stats = await backend.batch_process_library('/music_library')

    # Generate user recommendations
    user_history = [
        {'track_id': 'track1', 'play_count': 15, 'liked': True},
        {'track_id': 'track2', 'play_count': 8, 'liked': True},
        # ... more listening history
    ]

    recommendations = await backend.generate_personalized_recommendations(
        'user123', user_history
    )

    print("🎉 Streaming service backend processing complete!")

if __name__ == "__main__":
    import asyncio
    import time
    from datetime import datetime
    asyncio.run(streaming_service_example())
```

### Streaming Service Features

- **Automated content analysis** for all uploads
- **Quality validation** before accepting tracks
- **Personalized recommendations** based on audio features
- **Mood and genre categorization** for discovery
- **Batch processing** for large libraries

---

## Use Case 5: Radio Station Automation

**Scenario:** A radio station wants to automate playlist generation while maintaining smooth transitions and energy flow.

### Radio Automation Setup

```python
"""Radio station automation system."""

from tracktion.analysis import BPMDetector, KeyDetector, EnergyAnalyzer
from tracktion.tracklist import RadioPlaylistGenerator
from datetime import datetime, timedelta
import asyncio

class RadioStationAutomation:
    def __init__(self, station_config: Dict):
        self.config = station_config
        self.playlist_generator = RadioPlaylistGenerator()
        self.energy_analyzer = EnergyAnalyzer()

    async def generate_daily_programming(self, date: str) -> Dict:
        """Generate complete daily radio programming."""

        print(f"📻 Generating programming for {date}")

        programming_schedule = {
            'morning_show': await self._create_morning_show_playlist(6, 10),  # 6-10 AM
            'midday_mix': await self._create_midday_playlist(10, 14),        # 10 AM-2 PM
            'afternoon_drive': await self._create_drive_time_playlist(14, 18), # 2-6 PM
            'evening_chill': await self._create_evening_playlist(18, 22),    # 6-10 PM
            'late_night': await self._create_late_night_playlist(22, 2)      # 10 PM-2 AM
        }

        # Add transition analysis between shows
        for i, (show_name, playlist) in enumerate(programming_schedule.items()):
            if i < len(programming_schedule) - 1:
                next_show = list(programming_schedule.values())[i + 1]
                transition = await self._analyze_show_transition(playlist, next_show)
                playlist['transition_to_next'] = transition

        return programming_schedule

    async def _create_morning_show_playlist(self, start_hour: int, end_hour: int) -> Dict:
        """Create energizing morning show playlist."""

        duration_hours = end_hour - start_hour
        target_tracks = duration_hours * 12  # ~12 tracks per hour

        # Morning show characteristics
        criteria = {
            'energy_progression': 'gradual_increase',
            'start_energy': 0.6,  # Moderate start
            'peak_energy': 0.85,  # High energy peak
            'bpm_range': (100, 140),
            'mood_tags': ['uplifting', 'energetic', 'positive'],
            'avoid_explicit': True,  # Family-friendly morning hours
            'include_news_breaks': True,
            'ad_break_intervals': 15  # minutes
        }

        playlist = await self.playlist_generator.create_time_slot_playlist(
            start_time=f"{start_hour:02d}:00",
            duration_minutes=duration_hours * 60,
            criteria=criteria
        )

        return {
            'name': 'Morning Show',
            'time_slot': f"{start_hour:02d}:00-{end_hour:02d}:00",
            'tracks': playlist,
            'total_tracks': len(playlist),
            'programming_notes': 'High energy, positive vibes, news-friendly'
        }

    async def _create_drive_time_playlist(self, start_hour: int, end_hour: int) -> Dict:
        """Create engaging drive-time playlist."""

        duration_hours = end_hour - start_hour

        # Drive time characteristics - keep listeners engaged
        criteria = {
            'energy_progression': 'sustained_high',
            'target_energy': 0.8,
            'bpm_range': (110, 150),
            'include_classics': True,  # Mix of hits and new music
            'avoid_slow_songs': True,  # Keep energy up for commuters
            'genre_variety': True,
            'include_traffic_breaks': True,
            'ad_break_intervals': 12
        }

        playlist = await self.playlist_generator.create_time_slot_playlist(
            start_time=f"{start_hour:02d}:00",
            duration_minutes=duration_hours * 60,
            criteria=criteria
        )

        return {
            'name': 'Afternoon Drive',
            'time_slot': f"{start_hour:02d}:00-{end_hour:02d}:00",
            'tracks': playlist,
            'total_tracks': len(playlist),
            'programming_notes': 'High energy, hit-focused, commuter-friendly'
        }

# Real-time automation features
async def radio_automation_example():
    """Example radio station automation."""

    station_config = {
        'call_sign': 'KFUN',
        'format': 'Contemporary Hit Radio',
        'target_demo': '18-34',
        'music_library': '/radio/music_library'
    }

    radio = RadioStationAutomation(station_config)

    # Generate today's programming
    today = datetime.now().strftime('%Y-%m-%d')
    daily_programming = await radio.generate_daily_programming(today)

    print("📻 Daily programming generated:")
    for show_name, show_data in daily_programming.items():
        print(f"  🎵 {show_data['name']}: {show_data['total_tracks']} tracks ({show_data['time_slot']})")

if __name__ == "__main__":
    asyncio.run(radio_automation_example())
```

---

## Common Integration Patterns

### Environment Configuration

```bash
# .env template for different use cases
# Copy and modify based on your specific needs

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/tracktion
REDIS_URL=redis://localhost:6379/0

# Analysis Settings
BPM_CONFIDENCE_THRESHOLD=0.8
KEY_CONFIDENCE_THRESHOLD=0.7
ENABLE_MOOD_ANALYSIS=true
ENABLE_GENRE_CLASSIFICATION=true

# File Processing
MAX_FILE_SIZE_MB=100
SUPPORTED_FORMATS=mp3,flac,wav,aiff,m4a
AUTO_BACKUP_ANALYZED_FILES=true

# Performance Settings
CONCURRENT_ANALYSIS_WORKERS=4
BATCH_PROCESSING_SIZE=10
ENABLE_CACHING=true
CACHE_TTL_HOURS=24

# Integration Settings
WEBHOOK_URL=https://your-app.com/tracktion-webhook
API_KEY=your_secure_api_key_here
ENABLE_EXTERNAL_API=true
```

### Error Handling and Monitoring

```python
"""Common error handling and monitoring patterns."""

import logging
from typing import Optional
import asyncio
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class TrackionMonitor:
    def __init__(self):
        self.logger = self._setup_logging()
        self.metrics = {
            'files_processed': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }

    def _setup_logging(self):
        """Set up comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tracktion.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('tracktion')

    async def process_with_monitoring(self, process_func, *args, **kwargs) -> ProcessingResult:
        """Wrapper for processing functions with monitoring."""
        start_time = time.time()

        try:
            result = await process_func(*args, **kwargs)
            processing_time = time.time() - start_time

            self.metrics['files_processed'] += 1
            self._update_average_processing_time(processing_time)

            self.logger.info(f"Processing successful in {processing_time:.2f}s")
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics['processing_errors'] += 1

            self.logger.error(f"Processing failed after {processing_time:.2f}s: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    def get_health_status(self) -> dict:
        """Get system health status."""
        error_rate = (
            self.metrics['processing_errors'] /
            max(self.metrics['files_processed'], 1)
        )

        return {
            'status': 'healthy' if error_rate < 0.1 else 'degraded',
            'metrics': self.metrics,
            'error_rate': error_rate
        }
```

## Next Steps

These examples demonstrate various ways to integrate Tracktion into different workflows. For your specific use case:

1. **Choose the closest example** to your requirements
2. **Adapt the configuration** for your environment
3. **Test with a small dataset** first
4. **Monitor performance** and adjust settings
5. **Scale up** gradually based on results

For more detailed implementation guides, see:
- [API Integration Tutorial](api-integration.md)
- [Performance Optimization Guide](performance-optimization.md)
- [Production Deployment](../operations/deployment-procedures.md)

---

**Need help with your specific use case?** Create an issue in the repository or reach out to the community for support.
