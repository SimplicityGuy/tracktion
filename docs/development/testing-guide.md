# Tracktion Testing Guide

## Overview

This guide outlines the comprehensive testing strategy used across all Tracktion services. The project uses pytest as the primary testing framework with extensive use of mocking, fixtures, and specialized testing utilities.

## Testing Philosophy

### Core Principles
1. **Test Pyramid**: Emphasize unit tests, support with integration tests, supplement with E2E tests
2. **Fast Feedback**: Unit tests should run in milliseconds, integration tests in seconds
3. **Isolated Tests**: Each test should be independent and repeatable
4. **Comprehensive Coverage**: Aim for >90% code coverage with meaningful tests
5. **Realistic Mocking**: Mock external dependencies while maintaining realistic behavior
6. **Error Path Testing**: Test both success and failure scenarios extensively

### Testing Strategy
- **Unit Tests**: 80% of test coverage - fast, isolated, comprehensive
- **Integration Tests**: 15% of test coverage - test service boundaries and external dependencies
- **Performance Tests**: 5% of test coverage - benchmarks and load testing

## Test Framework and Tools

### Core Testing Stack
```python
# Primary testing dependencies from pyproject.toml
pytest>=8.3.4                    # Primary test framework
pytest-asyncio>=0.25.2          # Async test support
pytest-benchmark>=5.1.0         # Performance benchmarking
pytest-cov>=6.0.0               # Code coverage reporting
pytest-mock>=3.12.0             # Enhanced mocking capabilities
```

### pytest Configuration
```toml
# pyproject.toml - pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = """
    -ra
    --strict-markers
    --ignore=docs
    --ignore=setup.py
    --ignore=ci
    --ignore=.eggs
    --tb=short
    --maxfail=1
"""

# Test markers for categorization
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests that require external dependencies",
    "unit: marks tests as unit tests",
    "benchmark: marks tests as benchmark tests",
    "requires_db: marks tests that require database connections (PostgreSQL/Neo4j)",
    "requires_redis: marks tests that require Redis server",
]
```

## Test Organization Structure

### Directory Structure
```
tracktion/
├── tests/
│   ├── unit/                          # Unit tests (isolated, mocked)
│   │   ├── analysis_service/          # Service-specific unit tests
│   │   │   ├── test_bpm_detector_mock.py
│   │   │   ├── test_key_detector.py
│   │   │   ├── test_mood_analyzer.py
│   │   │   ├── api/
│   │   │   │   ├── test_app.py
│   │   │   │   └── test_websocket.py
│   │   │   ├── test_cue_handler/
│   │   │   │   ├── test_models.py
│   │   │   │   ├── test_converter.py
│   │   │   │   └── test_generator.py
│   │   │   └── test_pipeline_optimization/
│   │   │       ├── test_batch_processor.py
│   │   │       ├── test_circuit_breaker.py
│   │   │       └── test_graceful_shutdown.py
│   │   ├── tracklist_service/          # Tracklist service tests
│   │   │   ├── test_matching_service.py
│   │   │   ├── test_resilient_extractor.py
│   │   │   ├── test_error_handling.py
│   │   │   └── test_retry_manager.py
│   │   ├── file_watcher/               # File watcher tests
│   │   │   ├── test_file_scanner.py
│   │   │   ├── test_dual_hashing.py
│   │   │   └── test_message_publisher.py
│   │   └── shared/                     # Shared component tests
│   │       ├── test_neo4j_repository.py
│   │       └── test_async_repositories.py
│   ├── integration/                    # Integration tests
│   │   ├── analysis_service/
│   │   │   ├── test_full_pipeline.py
│   │   │   └── test_async_api_integration.py
│   │   ├── test_ogg_metadata_integration.py
│   │   └── test_docker_compose.py
│   ├── performance/                    # Performance and benchmark tests
│   │   ├── test_pipeline_performance.py
│   │   └── test_async_database_performance.py
│   └── research/                       # Research and accuracy validation
│       └── audio_analysis/
│           └── bpm_detection/
│               └── test_bpm_accuracy.py
```

### Service-Specific Test Structure
```
services/
├── analysis_service/
│   └── tests/                          # Service-internal tests
└── file_watcher/
    └── tests/
        └── unit/
            ├── test_performance_benchmarks.py
            └── test_edge_cases.py
```

## Unit Testing Patterns

### Test Class Organization
```python
"""Unit tests for BPM detection functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from services.analysis_service.src.bpm_detector import BPMDetector, BPMDetectionResult
from services.analysis_service.src.exceptions import InvalidAudioFileError


class TestBPMDetectorMocked:
    """Test suite for BPMDetector class with complete mocking."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock external dependencies
        with (
            patch("services.analysis_service.src.bmp_detector.es.RhythmExtractor2013") as mock_rhythm,
            patch("services.analysis_service.src.bmp_detector.es.PercivalBmpEstimator") as mock_percival,
        ):
            self.mock_rhythm_instance = Mock()
            self.mock_percival_instance = Mock()
            mock_rhythm.return_value = self.mock_rhythm_instance
            mock_percival.return_value = self.mock_percival_instance

            self.detector = BPMDetector(
                confidence_threshold=0.7,
                agreement_tolerance=5.0
            )

    def teardown_method(self):
        """Clean up after each test method."""
        # Cleanup any resources if needed
        pass
```

### Fixture Patterns
```python
@pytest.fixture
def sample_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create synthetic audio data
        sample_rate = 44100
        duration = 5  # seconds
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)

        # Write audio data (simplified)
        f.write(audio.astype(np.float32).tobytes())
        temp_path = f.name

    yield temp_path

    # Cleanup
    with contextlib.suppress(Exception):
        Path(temp_path).unlink()

@pytest.fixture
def mock_essentia_components():
    """Mock all Essentia components for consistent testing."""
    with (
        patch("services.analysis_service.src.bmp_detector.es.MonoLoader") as mock_loader,
        patch("services.analysis_service.src.bmp_detector.es.RhythmExtractor2013") as mock_rhythm,
        patch("services.analysis_service.src.bmp_detector.es.PercivalBmpEstimator") as mock_percival,
    ):
        # Configure mock behavior
        mock_audio = np.ones(44100, dtype=np.float32)
        mock_loader.return_value = Mock(return_value=mock_audio)

        yield {
            'loader': mock_loader,
            'rhythm': mock_rhythm,
            'percival': mock_percival,
            'audio': mock_audio
        }

@pytest.fixture
def detector_config():
    """Provide detector configuration for tests."""
    return {
        'confidence_threshold': 0.7,
        'agreement_tolerance': 5.0,
        'sample_rate': 44100
    }
```

### Parametrized Testing
```python
@pytest.mark.parametrize("confidence,expected_algorithm", [
    (0.95, "primary"),      # High confidence - use primary
    (0.85, "primary"),      # Good confidence - use primary
    (0.65, "consensus"),    # Low confidence, algorithms agree
    (0.45, "fallback"),     # Very low confidence - use fallback
])
def test_algorithm_selection_based_on_confidence(detector, confidence, expected_algorithm):
    """Test that correct algorithm is selected based on confidence levels."""
    with patch.object(detector, '_extract_rhythm') as mock_rhythm:
        mock_rhythm.return_value = (128.0, np.array([]), confidence, None, np.array([]))

        result = detector.detect_bpm("/path/to/test.mp3")

        assert result["algorithm"] == expected_algorithm
        assert result["confidence"] >= 0.0

@pytest.mark.parametrize("bmp_primary,bmp_fallback,tolerance,should_agree", [
    (128.0, 130.0, 5.0, True),    # Within tolerance - should agree
    (128.0, 140.0, 5.0, False),   # Outside tolerance - should disagree
    (120.0, 119.0, 2.0, True),    # Close values - should agree
    (120.0, 110.0, 2.0, False),   # Far values - should disagree
])
def test_algorithm_agreement_logic(detector, bmp_primary, bmp_fallback, tolerance, should_agree):
    """Test algorithm agreement detection with various BPM differences."""
    detector.agreement_tolerance = tolerance

    # Test the agreement logic
    agreement = abs(bmp_primary - bmp_fallback) < detector.agreement_tolerance
    assert agreement == should_agree
```

### Mocking External Dependencies
```python
class TestAudioAnalysisWithMocks:
    """Test audio analysis with comprehensive mocking."""

    @patch("services.analysis_service.src.storage_handler.Neo4jRepository")
    @patch("services.analysis_service.src.mood_analyzer.MoodAnalyzer")
    @patch("services.analysis_service.src.key_detector.KeyDetector")
    @patch("services.analysis_service.src.bmp_detector.BPMDetector")
    def test_full_analysis_pipeline(self, mock_bmp, mock_key, mock_mood, mock_storage):
        """Test complete analysis pipeline with all components mocked."""
        # Configure mock behaviors
        mock_bmp_instance = mock_bmp.return_value
        mock_bmp_instance.detect_bmp.return_value = {
            'bmp': 128.0,
            'confidence': 0.85,
            'algorithm': 'primary'
        }

        mock_key_instance = mock_key.return_value
        mock_key_instance.detect_key.return_value = Mock(
            key='C',
            scale='major',
            confidence=0.9
        )

        mock_mood_instance = mock_mood.return_value
        mock_mood_instance.analyze.return_value = {
            'valence': 0.7,
            'arousal': 0.6,
            'dominant_mood': 'happy'
        }

        mock_storage_instance = mock_storage.return_value
        mock_storage_instance.store_analysis.return_value = True

        # Test the pipeline
        analyzer = AudioAnalyzer()
        result = analyzer.analyze_file("/path/to/test.mp3")

        # Verify all components were called
        mock_bmp_instance.detect_bmp.assert_called_once()
        mock_key_instance.detect_key.assert_called_once()
        mock_mood_instance.analyze.assert_called_once()
        mock_storage_instance.store_analysis.assert_called_once()

        # Verify result structure
        assert 'bmp' in result
        assert 'key' in result
        assert 'mood' in result
        assert result['confidence'] > 0.0
```

### Error Handling Tests
```python
class TestErrorHandling:
    """Test error handling scenarios."""

    def test_file_not_found_error(self, detector):
        """Test handling of missing audio files."""
        with pytest.raises(FileNotFoundError):
            detector.detect_bmp("/nonexistent/file.mp3")

    def test_unsupported_format_error(self, detector):
        """Test handling of unsupported audio formats."""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            detector.detect_bmp("/path/to/file.xyz")

        assert "Unsupported format" in str(exc_info.value)
        assert exc_info.value.error_code == "UNSUPPORTED_FORMAT"

    @patch("services.analysis_service.src.bmp_detector.es.MonoLoader")
    def test_audio_loading_failure(self, mock_loader, detector):
        """Test handling of audio loading failures."""
        # Simulate loading failure
        mock_loader.side_effect = RuntimeError("Failed to load audio")

        with pytest.raises(RuntimeError) as exc_info:
            detector.detect_bmp("/path/to/corrupted.mp3")

        assert "BPM detection failed" in str(exc_info.value)

    @patch("services.analysis_service.src.bmp_detector.es.RhythmExtractor2013")
    def test_algorithm_failure_with_fallback(self, mock_rhythm, detector):
        """Test graceful fallback when primary algorithm fails."""
        # Primary algorithm fails
        mock_rhythm_instance = mock_rhythm.return_value
        mock_rhythm_instance.side_effect = RuntimeError("Algorithm failed")

        # Fallback should still work
        with patch.object(detector, '_estimate_bmp_percival') as mock_fallback:
            mock_fallback.return_value = 120.0

            result = detector.detect_bmp("/path/to/test.mp3")

            assert result['bmp'] == 120.0
            assert result['algorithm'] == 'fallback'
            assert result['needs_review'] is True
```

## Integration Testing Patterns

### Service Integration Tests
```python
class TestFullPipeline:
    """Integration tests for the complete analysis pipeline."""

    @pytest.mark.integration
    @pytest.mark.requires_db
    def test_end_to_end_analysis_workflow(self, real_audio_file, neo4j_connection):
        """Test complete analysis workflow with real database."""
        # This test uses real external dependencies
        analyzer = AudioAnalyzer(
            storage=Neo4jRepository(neo4j_connection),
            cache=RedisCache()  # Could use real Redis or fakeredis
        )

        result = analyzer.analyze_file(real_audio_file)

        # Verify analysis completed
        assert result['success'] is True
        assert 'bmp' in result
        assert 'key' in result

        # Verify data was stored
        stored_data = analyzer.storage.get_analysis(result['analysis_id'])
        assert stored_data is not None
        assert stored_data['bmp'] == result['bmp']

    @pytest.mark.integration
    @pytest.mark.requires_redis
    def test_caching_integration(self, sample_audio_file, redis_connection):
        """Test that analysis results are properly cached."""
        analyzer = AudioAnalyzer(cache=RedisCache(redis_connection))

        # First analysis - should calculate and cache
        start_time = time.time()
        result1 = analyzer.analyze_file(sample_audio_file)
        first_duration = time.time() - start_time

        # Second analysis - should use cache
        start_time = time.time()
        result2 = analyzer.analyze_file(sample_audio_file)
        second_duration = time.time() - start_time

        # Results should be identical
        assert result1['bmp'] == result2['bmp']
        assert result1['key'] == result2['key']

        # Second call should be much faster (cached)
        assert second_duration < first_duration / 2
```

### Database Integration Tests
```python
@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseIntegration:
    """Test database operations with real database connections."""

    @pytest.fixture
    def neo4j_session(self):
        """Provide real Neo4j session for testing."""
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "testpassword")
        )

        with driver.session() as session:
            # Clean up test data before test
            session.run("MATCH (n:TestNode) DELETE n")
            yield session
            # Clean up test data after test
            session.run("MATCH (n:TestNode) DELETE n")

        driver.close()

    def test_store_analysis_results(self, neo4j_session):
        """Test storing analysis results in Neo4j."""
        storage = Neo4jRepository(neo4j_session)

        analysis_data = {
            'file_path': '/test/audio.mp3',
            'bmp': 128.0,
            'key': 'C major',
            'confidence': 0.85
        }

        # Store data
        analysis_id = storage.store_analysis(analysis_data)
        assert analysis_id is not None

        # Retrieve data
        retrieved = storage.get_analysis(analysis_id)
        assert retrieved['bmp'] == 128.0
        assert retrieved['key'] == 'C major'
```

### API Integration Tests
```python
@pytest.mark.integration
class TestAPIIntegration:
    """Test API endpoints with real HTTP calls."""

    @pytest.fixture
    def api_client(self):
        """Create test client for API testing."""
        from fastapi.testclient import TestClient
        from services.analysis_service.src.api.app import app

        return TestClient(app)

    def test_health_endpoint(self, api_client):
        """Test API health check endpoint."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data

    def test_analysis_endpoint_with_mock(self, api_client):
        """Test analysis endpoint with mocked processing."""
        with patch('services.analysis_service.src.api.endpoints.AudioAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.analyze_file.return_value = {
                'bmp': 128.0,
                'confidence': 0.85,
                'key': 'C major'
            }

            response = api_client.post(
                "/analyze",
                json={'file_path': '/test/audio.mp3'}
            )

            assert response.status_code == 200
            data = response.json()
            assert data['bmp'] == 128.0
            assert data['key'] == 'C major'
```

## Performance and Benchmark Testing

### Benchmark Tests
```python
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_bmp_detection_performance(self, benchmark, sample_audio_file):
        """Benchmark BPM detection performance."""
        detector = BPMDetector()

        # Benchmark the detection
        result = benchmark(detector.detect_bmp, sample_audio_file)

        # Verify result is valid
        assert 'bmp' in result
        assert result['bmp'] > 0

        # Performance assertions
        assert benchmark.stats.mean < 2.0  # Should complete in under 2 seconds

    @pytest.mark.parametrize("file_size_mb", [1, 5, 10, 25])
    def test_memory_usage_scaling(self, file_size_mb, benchmark, memory_monitor):
        """Test memory usage with different file sizes."""
        # Generate audio file of specified size
        audio_file = generate_test_audio(size_mb=file_size_mb)

        detector = BPMDetector()

        with memory_monitor:
            result = detector.detect_bmp(audio_file)

        # Memory usage should scale reasonably
        peak_memory_mb = memory_monitor.peak / (1024 * 1024)
        assert peak_memory_mb < file_size_mb * 3  # Should not exceed 3x file size
```

### Load Testing
```python
@pytest.mark.slow
@pytest.mark.benchmark
class TestLoadBehavior:
    """Test system behavior under load."""

    def test_concurrent_analysis_performance(self):
        """Test performance with multiple concurrent analyses."""
        import concurrent.futures

        detector = BPMDetector()
        audio_files = [f"/test/audio_{i}.mp3" for i in range(10)]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(detector.detect_bmp, audio_file)
                for audio_file in audio_files
            ]

            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # All analyses should complete
        assert len(results) == 10
        assert all('bmp' in result for result in results)

        # Should benefit from concurrency
        assert total_time < 20.0  # Should complete in reasonable time
```

## Async Testing Patterns

### Async Test Setup
```python
@pytest.mark.asyncio
class TestAsyncAudioProcessing:
    """Test asynchronous audio processing functionality."""

    async def test_async_batch_processing(self):
        """Test processing multiple files asynchronously."""
        processor = AsyncAudioProcessor()
        file_paths = [f"/test/audio_{i}.mp3" for i in range(5)]

        results = await processor.process_batch(file_paths, max_concurrent=3)

        assert len(results) == 5
        assert all('bmp' in result for result in results)

    async def test_async_error_handling(self):
        """Test async error handling and recovery."""
        processor = AsyncAudioProcessor()

        with patch.object(processor, '_process_single_file') as mock_process:
            # Make some files fail
            mock_process.side_effect = [
                {'bmp': 128.0},  # Success
                RuntimeError("Processing failed"),  # Failure
                {'bmp': 120.0},  # Success
            ]

            results = await processor.process_batch([
                "/test/audio_1.mp3",
                "/test/audio_2.mp3",
                "/test/audio_3.mp3"
            ])

            # Should handle failures gracefully
            assert len(results) == 2  # Only successful results
            assert all('bmp' in result for result in results)
```

### Async Context Managers
```python
@pytest.mark.asyncio
async def test_async_resource_management():
    """Test async resource management with context managers."""
    async with AsyncAudioProcessor() as processor:
        result = await processor.process_file("/test/audio.mp3")
        assert 'bmp' in result

    # Processor should be properly cleaned up
    assert processor.is_closed is True
```

## Test Data Management

### Test Fixtures and Data
```python
@pytest.fixture(scope="session")
def test_audio_samples():
    """Provide test audio samples for consistent testing."""
    samples = {}

    # Generate different types of test audio
    samples['sine_440hz'] = generate_sine_wave(frequency=440, duration=5)
    samples['sine_880hz'] = generate_sine_wave(frequency=880, duration=5)
    samples['noise'] = generate_white_noise(duration=3)
    samples['drums_120bmp'] = generate_drum_pattern(bmp=120, duration=10)
    samples['drums_140bmp'] = generate_drum_pattern(bmp=140, duration=10)

    return samples

@pytest.fixture
def mock_audio_metadata():
    """Provide realistic audio metadata for testing."""
    return {
        'title': 'Test Track',
        'artist': 'Test Artist',
        'album': 'Test Album',
        'duration_seconds': 180,
        'sample_rate': 44100,
        'bitrate': 320000,
        'format': 'mp3'
    }
```

### Database Test Data
```python
@pytest.fixture
def seed_test_data(neo4j_session):
    """Seed database with test data."""
    test_data = [
        {
            'file_path': '/test/track1.mp3',
            'bmp': 128.0,
            'key': 'C major',
            'mood': 'energetic'
        },
        {
            'file_path': '/test/track2.mp3',
            'bmp': 140.0,
            'key': 'G minor',
            'mood': 'dark'
        }
    ]

    for data in test_data:
        neo4j_session.run(
            "CREATE (a:AudioTrack {file_path: $file_path, bmp: $bmp, key: $key, mood: $mood})",
            **data
        )

    return test_data
```

## Test Execution and CI/CD

### Running Tests Locally
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m "unit"                    # Only unit tests
uv run pytest -m "not slow"                # Skip slow tests
uv run pytest -m "integration"             # Only integration tests
uv run pytest -m "requires_db"             # Tests requiring database

# Run with coverage
uv run pytest --cov=services --cov-report=html --cov-report=term

# Run performance benchmarks
uv run pytest -m "benchmark" --benchmark-only

# Run tests in parallel
uv run pytest -n auto                      # Use all CPU cores

# Verbose output with logging
uv run pytest -v -s --log-cli-level=DEBUG
```

### Test Selection Examples
```bash
# Run specific test file
uv run pytest tests/unit/analysis_service/test_bmp_detector_mock.py

# Run specific test method
uv run pytest tests/unit/analysis_service/test_bmp_detector_mock.py::TestBPMDetectorMocked::test_detect_bmp_high_confidence

# Run tests matching pattern
uv run pytest -k "bmp_detection"           # All tests with "bmp_detection" in name

# Run only failed tests from last run
uv run pytest --lf                         # Last failed
uv run pytest --ff                         # Failed first
```

### Coverage Configuration
```toml
# pyproject.toml - coverage settings
[tool.coverage.run]
branch = true
source = ["tracktion", "services"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/migrations/*",
    "*/venv/*",
    "*/.venv/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]

# Coverage targets
fail_under = 90                    # Fail if coverage below 90%
show_missing = true                # Show missing lines
skip_covered = false              # Don't skip covered files in report
```

## Test Maintenance and Best Practices

### Writing Effective Tests
```python
# Good test - descriptive name, single responsibility, clear assertions
def test_fuzzy_match_identical_strings_returns_perfect_score():
    """Test that fuzzy matching identical strings returns score of 1.0."""
    matcher = FuzzyMatcher()

    result = matcher.match("test string", "test string")

    assert result == 1.0

# Good test - covers edge case with clear documentation
def test_fuzzy_match_empty_strings_returns_zero():
    """Test that fuzzy matching empty strings returns score of 0.0."""
    matcher = FuzzyMatcher()

    result = matcher.match("", "")

    assert result == 0.0

# Good test - parameterized for comprehensive coverage
@pytest.mark.parametrize("input_string,expected_normalized", [
    ("Test String", "test string"),
    ("UPPERCASE", "uppercase"),
    ("Mixed_Case-123", "mixed case 123"),
    ("", ""),
])
def test_string_normalization_handles_various_cases(input_string, expected_normalized):
    """Test string normalization with various input cases."""
    normalizer = StringNormalizer()

    result = normalizer.normalize(input_string)

    assert result == expected_normalized
```

### Test Maintenance Guidelines
1. **Keep tests simple and focused**: One concept per test
2. **Use descriptive test names**: Name should describe what is being tested
3. **Maintain test independence**: Tests should not depend on each other
4. **Update tests with code changes**: Keep tests in sync with implementation
5. **Review test coverage regularly**: Ensure critical paths are covered
6. **Clean up test data**: Use fixtures and teardown methods
7. **Mock external dependencies**: Keep tests fast and reliable
8. **Test both success and failure paths**: Error handling is crucial

### Common Testing Anti-Patterns to Avoid
```python
# BAD: Test too complex and testing multiple things
def test_everything_at_once():
    # Tests BPM detection, key detection, storage, and API in one test
    pass

# BAD: No clear assertion or expected behavior
def test_something():
    detector.detect_bmp("file.mp3")
    # No assertions!

# BAD: Brittle test that breaks with unrelated changes
def test_detector_internal_implementation():
    assert detector.internal_counter == 5  # Testing implementation details

# BAD: Test depends on external state or other tests
def test_depends_on_previous_test():
    # Assumes test_setup_data() ran first
    pass

# GOOD: Clear, focused, independent test
def test_detect_bmp_returns_float_between_valid_range():
    """Test that BPM detection returns value in valid BPM range."""
    detector = BPMDetector()

    result = detector.detect_bmp("/path/to/test.mp3")

    assert isinstance(result['bmp'], float)
    assert 60.0 <= result['bmp'] <= 200.0  # Reasonable BPM range
```

This comprehensive testing guide ensures consistent, reliable, and maintainable tests across all Tracktion services while supporting both development velocity and code quality.
