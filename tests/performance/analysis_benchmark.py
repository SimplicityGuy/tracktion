"""Performance benchmarks for the analysis service."""

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np


class AnalysisPerformanceBenchmark:
    """Benchmark suite for analysis service performance."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: list[dict[str, Any]] = []

    def benchmark_bpm_detection(self, iterations: int = 10) -> dict[str, float]:
        """Benchmark BPM detection performance.

        Args:
            iterations: Number of iterations to run

        Returns:
            Performance metrics
        """
        from services.analysis_service.src.bpm_detector import BPMDetector

        # Mock essentia to avoid actual audio processing
        with patch("services.analysis_service.src.bpm_detector.es") as mock_es:
            # Setup mocks
            mock_es.MonoLoader.return_value.return_value = np.random.randn(44100 * 30)  # 30 seconds
            mock_es.RhythmExtractor2013.return_value.return_value = (120.0, [120] * 100, [0.9] * 100, [], 0.9)

            detector = BPMDetector()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = detector.detect_bpm("/fake/audio.mp3")
                end = time.perf_counter()
                times.append(end - start)

            return {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "iterations": iterations,
            }

    def benchmark_key_detection(self, iterations: int = 10) -> dict[str, float]:
        """Benchmark key detection performance.

        Args:
            iterations: Number of iterations to run

        Returns:
            Performance metrics
        """
        from services.analysis_service.src.key_detector import KeyDetector

        with patch("services.analysis_service.src.key_detector.es") as mock_es:
            # Setup mocks
            mock_audio = np.random.randn(44100 * 30)
            mock_es.MonoLoader.return_value.return_value = mock_audio
            mock_es.KeyExtractor.return_value.return_value = ("C", "major", 0.85)
            mock_es.Windowing.return_value.return_value = mock_audio[:2048]
            mock_es.Spectrum.return_value.return_value = np.random.rand(1025)
            mock_es.SpectralPeaks.return_value.return_value = (np.array([100, 200]), np.array([0.5, 0.3]))
            mock_es.HPCP.return_value.return_value = np.random.rand(12)
            mock_es.Key.return_value.return_value = ("C", "major", 0.85, 0.9)

            detector = KeyDetector()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = detector.detect_key("/fake/audio.mp3")
                end = time.perf_counter()
                times.append(end - start)

            return {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "iterations": iterations,
            }

    def benchmark_mood_analysis(self, iterations: int = 10) -> dict[str, float]:
        """Benchmark mood analysis performance.

        Args:
            iterations: Number of iterations to run

        Returns:
            Performance metrics
        """
        from services.analysis_service.src.model_manager import ModelManager
        from services.analysis_service.src.mood_analyzer import MoodAnalyzer

        with patch("services.analysis_service.src.mood_analyzer.es") as mock_es:
            mock_es.MonoLoader.return_value.return_value = np.random.randn(16000 * 30)

            # Mock model manager
            mock_manager = Mock(spec=ModelManager)
            mock_manager.get_all_models.return_value = ["mood_happy", "genre_discogs_effnet"]

            def mock_load_model(model_id):
                mock_model = Mock()
                mock_model.return_value = Mock(numpy=lambda: np.array([0.7]))
                return mock_model

            mock_manager.load_model.side_effect = mock_load_model

            analyzer = MoodAnalyzer(model_manager=mock_manager)

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = analyzer.analyze_mood("/fake/audio.mp3")
                end = time.perf_counter()
                times.append(end - start)

            return {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "iterations": iterations,
            }

    def benchmark_cache_operations(self, iterations: int = 100) -> dict[str, Any]:
        """Benchmark cache operations.

        Args:
            iterations: Number of iterations to run

        Returns:
            Performance metrics for cache operations
        """
        from services.analysis_service.src.audio_cache import AudioCache

        with patch("redis.Redis") as MockRedis:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            MockRedis.return_value = mock_redis

            cache = AudioCache(enabled=True)

            # Benchmark cache writes
            write_times = []
            test_data = {"bpm": 120, "confidence": 0.9, "algorithm": "multifeature"}

            for i in range(iterations):
                start = time.perf_counter()
                cache.set_bpm_results(f"/fake/audio_{i}.mp3", test_data)
                end = time.perf_counter()
                write_times.append(end - start)

            # Benchmark cache reads
            mock_redis.get.return_value = json.dumps(test_data)
            read_times = []

            for i in range(iterations):
                start = time.perf_counter()
                _ = cache.get_bpm_results(f"/fake/audio_{i}.mp3")
                end = time.perf_counter()
                read_times.append(end - start)

            return {
                "write": {
                    "mean_time": np.mean(write_times),
                    "std_time": np.std(write_times),
                    "min_time": np.min(write_times),
                    "max_time": np.max(write_times),
                    "total_time": np.sum(write_times),
                    "operations_per_second": iterations / np.sum(write_times),
                },
                "read": {
                    "mean_time": np.mean(read_times),
                    "std_time": np.std(read_times),
                    "min_time": np.min(read_times),
                    "max_time": np.max(read_times),
                    "total_time": np.sum(read_times),
                    "operations_per_second": iterations / np.sum(read_times),
                },
                "iterations": iterations,
            }

    def benchmark_full_pipeline(self, iterations: int = 5) -> dict[str, Any]:
        """Benchmark the complete analysis pipeline.

        Args:
            iterations: Number of iterations to run

        Returns:
            Performance metrics for full pipeline
        """
        from services.analysis_service.src.message_consumer import MessageConsumer

        # Setup mocks for all components
        with (
            patch("services.analysis_service.src.message_consumer.BPMDetector") as MockBPM,
            patch("services.analysis_service.src.message_consumer.KeyDetector") as MockKey,
            patch("services.analysis_service.src.message_consumer.MoodAnalyzer") as MockMood,
            patch("services.analysis_service.src.message_consumer.AudioCache") as MockCache,
            patch("services.analysis_service.src.message_consumer.ModelManager") as MockManager,
        ):
            # Setup mock responses
            mock_bpm = Mock()
            mock_bpm.detect_bpm.return_value = {"bpm": 120, "confidence": 0.9}
            MockBPM.return_value = mock_bpm

            mock_key = Mock()
            key_result = Mock()
            key_result.key = "C"
            key_result.scale = "major"
            key_result.confidence = 0.85
            key_result.agreement = True
            key_result.needs_review = False
            key_result.alternative_key = None
            key_result.alternative_scale = None
            mock_key.detect_key.return_value = key_result
            MockKey.return_value = mock_key

            mock_mood = Mock()
            mood_result = Mock()
            mood_result.mood_scores = {"happy": 0.8}
            mood_result.primary_genre = "Pop"
            mood_result.genre_confidence = 0.9
            mood_result.genres = []
            mood_result.danceability = 0.75
            mood_result.energy = 0.8
            mood_result.valence = 0.7
            mood_result.arousal = 0.6
            mood_result.voice_instrumental = "voice"
            mood_result.overall_confidence = 0.85
            mood_result.needs_review = False
            mock_mood.analyze_mood.return_value = mood_result
            MockMood.return_value = mock_mood

            mock_cache = Mock()
            mock_cache.get_bpm_results.return_value = None
            mock_cache.get_key_results.return_value = None
            mock_cache.get_mood_results.return_value = None
            mock_cache.set_bpm_results.return_value = True
            mock_cache.set_key_results.return_value = True
            mock_cache.set_mood_results.return_value = True
            MockCache.return_value = mock_cache

            MockManager.return_value = Mock()

            consumer = MessageConsumer(
                rabbitmq_url="amqp://localhost",
                enable_cache=True,
                enable_temporal_analysis=False,
                enable_key_detection=True,
                enable_mood_analysis=True,
            )

            # Create fake audio file
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True

                times = []

                for i in range(iterations):
                    # Track component times
                    bpm_start = time.perf_counter()
                    overall_start = bpm_start

                    _ = consumer.process_audio_file(f"/fake/audio_{i}.mp3", f"recording_{i}")

                    overall_end = time.perf_counter()
                    times.append(overall_end - overall_start)

                return {
                    "total": {
                        "mean_time": np.mean(times),
                        "std_time": np.std(times),
                        "min_time": np.min(times),
                        "max_time": np.max(times),
                        "total_time": np.sum(times),
                    },
                    "iterations": iterations,
                }

    def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all benchmarks and return results.

        Returns:
            Complete benchmark results
        """
        print("Running performance benchmarks...")

        results = {}

        print("  - BPM Detection...")
        results["bpm_detection"] = self.benchmark_bpm_detection()

        print("  - Key Detection...")
        results["key_detection"] = self.benchmark_key_detection()

        print("  - Mood Analysis...")
        results["mood_analysis"] = self.benchmark_mood_analysis()

        print("  - Cache Operations...")
        results["cache_operations"] = self.benchmark_cache_operations()

        print("  - Full Pipeline...")
        results["full_pipeline"] = self.benchmark_full_pipeline()

        # Calculate summary statistics
        results["summary"] = {
            "total_components_mean_time": (
                results["bpm_detection"]["mean_time"]
                + results["key_detection"]["mean_time"]
                + results["mood_analysis"]["mean_time"]
            ),
            "cache_read_ops_per_second": results["cache_operations"]["read"]["operations_per_second"],  # type: ignore[index]
            "cache_write_ops_per_second": results["cache_operations"]["write"]["operations_per_second"],  # type: ignore[index]
            "full_pipeline_mean_time": results["full_pipeline"]["total"]["mean_time"],  # type: ignore[index]
        }

        return results

    def print_results(self, results: dict[str, Any]) -> None:
        """Print benchmark results in a readable format.

        Args:
            results: Benchmark results to print
        """
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        print("\nComponent Performance:")
        print("-" * 40)

        components = ["bpm_detection", "key_detection", "mood_analysis"]
        for component in components:
            if component in results:
                data = results[component]
                print(f"\n{component.replace('_', ' ').title()}:")
                print(f"  Mean time: {data['mean_time'] * 1000:.2f} ms")
                print(f"  Std dev:   {data['std_time'] * 1000:.2f} ms")
                print(f"  Min time:  {data['min_time'] * 1000:.2f} ms")
                print(f"  Max time:  {data['max_time'] * 1000:.2f} ms")

        print("\n\nCache Performance:")
        print("-" * 40)
        if "cache_operations" in results:
            cache = results["cache_operations"]
            print("\nWrite Operations:")
            print(f"  Mean time: {cache['write']['mean_time'] * 1000:.3f} ms")
            print(f"  Ops/sec:   {cache['write']['operations_per_second']:.0f}")
            print("\nRead Operations:")
            print(f"  Mean time: {cache['read']['mean_time'] * 1000:.3f} ms")
            print(f"  Ops/sec:   {cache['read']['operations_per_second']:.0f}")

        print("\n\nFull Pipeline Performance:")
        print("-" * 40)
        if "full_pipeline" in results:
            pipeline = results["full_pipeline"]["total"]
            print(f"  Mean time: {pipeline['mean_time'] * 1000:.2f} ms")
            print(f"  Std dev:   {pipeline['std_time'] * 1000:.2f} ms")
            print(f"  Min time:  {pipeline['min_time'] * 1000:.2f} ms")
            print(f"  Max time:  {pipeline['max_time'] * 1000:.2f} ms")

        print("\n\nSummary:")
        print("-" * 40)
        if "summary" in results:
            summary = results["summary"]
            print(f"  Total components mean: {summary['total_components_mean_time'] * 1000:.2f} ms")
            print(f"  Pipeline mean time:    {summary['full_pipeline_mean_time'] * 1000:.2f} ms")
            print(f"  Cache read ops/sec:    {summary['cache_read_ops_per_second']:.0f}")
            print(f"  Cache write ops/sec:   {summary['cache_write_ops_per_second']:.0f}")

        print("\n" + "=" * 60)


def main():
    """Run benchmarks from command line."""
    benchmark = AnalysisPerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.print_results(results)

    # Optionally save results to file
    output_file = Path("benchmark_results.json")
    with output_file.open("w") as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32 | np.float64):
                return float(obj)
            elif isinstance(obj, np.int32 | np.int64):
                return int(obj)
            return obj

        json_results = json.dumps(results, default=convert_numpy, indent=2)
        f.write(json_results)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
