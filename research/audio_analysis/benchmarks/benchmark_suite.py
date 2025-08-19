#!/usr/bin/env python
"""
Performance Benchmarking Suite
Comprehensive performance testing for audio analysis libraries.
"""

import gc
import os
import platform
import psutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import essentia.standard as es
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Import our detectors
import sys

sys.path.append(str(Path(__file__).parent.parent))
from bpm_detection.compare_bpm import EssentiaBPMDetector, LibrosaBPMDetector
from key_detection.compare_keys import EssentiaKeyDetector, LibrosaKeyDetector


class PerformanceBenchmark:
    """Performance benchmarking for audio analysis libraries."""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.results = []

    def _get_system_info(self) -> Dict:
        """Get system hardware and software information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version(),
        }

    def benchmark_file_loading(self, audio_files: List[str]) -> pd.DataFrame:
        """Benchmark file loading performance."""
        results = []

        for audio_path in tqdm(audio_files, desc="Benchmarking file loading"):
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

            # Benchmark Librosa loading
            gc.collect()
            start_time = time.perf_counter()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)

            try:
                y, sr = librosa.load(audio_path, sr=None)
                librosa_time = time.perf_counter() - start_time
                librosa_mem = psutil.Process().memory_info().rss / (1024 * 1024) - start_mem
                librosa_samples = len(y)
                librosa_error = None
            except Exception as e:
                librosa_time = time.perf_counter() - start_time
                librosa_mem = 0
                librosa_samples = 0
                librosa_error = str(e)

            # Clear memory
            if "y" in locals():
                del y
            gc.collect()

            # Benchmark Essentia loading
            start_time = time.perf_counter()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)

            try:
                audio = es.MonoLoader(filename=audio_path)()
                essentia_time = time.perf_counter() - start_time
                essentia_mem = psutil.Process().memory_info().rss / (1024 * 1024) - start_mem
                essentia_samples = len(audio)
                essentia_error = None
            except Exception as e:
                essentia_time = time.perf_counter() - start_time
                essentia_mem = 0
                essentia_samples = 0
                essentia_error = str(e)

            # Clear memory
            if "audio" in locals():
                del audio
            gc.collect()

            results.append(
                {
                    "filename": Path(audio_path).name,
                    "file_size_mb": file_size_mb,
                    "librosa_load_time": librosa_time,
                    "librosa_memory_mb": librosa_mem,
                    "librosa_samples": librosa_samples,
                    "librosa_error": librosa_error,
                    "essentia_load_time": essentia_time,
                    "essentia_memory_mb": essentia_mem,
                    "essentia_samples": essentia_samples,
                    "essentia_error": essentia_error,
                }
            )

        return pd.DataFrame(results)

    def benchmark_feature_extraction(self, audio_files: List[str]) -> pd.DataFrame:
        """Benchmark feature extraction performance."""
        results = []

        for audio_path in tqdm(audio_files[:5], desc="Benchmarking features"):  # Test subset
            # Load audio once
            y, sr = librosa.load(audio_path, sr=None)
            audio_essentia = es.MonoLoader(filename=audio_path)()

            result = {
                "filename": Path(audio_path).name,
                "duration_sec": len(y) / sr,
            }

            # Spectral Centroid
            gc.collect()
            start = time.perf_counter()
            _ = librosa.feature.spectral_centroid(y=y, sr=sr)
            result["librosa_spectral_centroid"] = time.perf_counter() - start

            start = time.perf_counter()
            spectrum = es.Spectrum()(audio_essentia)
            _ = es.Centroid()(spectrum)
            result["essentia_spectral_centroid"] = time.perf_counter() - start

            # MFCC
            gc.collect()
            start = time.perf_counter()
            _ = librosa.feature.mfcc(y=y, sr=sr)
            result["librosa_mfcc"] = time.perf_counter() - start

            start = time.perf_counter()
            _ = es.MFCC()(spectrum)
            result["essentia_mfcc"] = time.perf_counter() - start

            # Chroma
            gc.collect()
            start = time.perf_counter()
            _ = librosa.feature.chroma_stft(y=y, sr=sr)
            result["librosa_chroma"] = time.perf_counter() - start

            # Tempo (BPM)
            gc.collect()
            start = time.perf_counter()
            _ = librosa.beat.tempo(y=y, sr=sr)
            result["librosa_tempo"] = time.perf_counter() - start

            start = time.perf_counter()
            _ = es.RhythmExtractor2013()(audio_essentia)
            result["essentia_tempo"] = time.perf_counter() - start

            results.append(result)

            # Clear memory
            del y, audio_essentia, spectrum
            gc.collect()

        return pd.DataFrame(results)

    def benchmark_parallel_processing(self, audio_files: List[str], max_workers: int = 4) -> pd.DataFrame:
        """Benchmark parallel processing capabilities."""
        results = []

        # Single-threaded baseline
        gc.collect()
        start_time = time.perf_counter()

        for audio_path in audio_files[:10]:  # Test subset
            detector = LibrosaBPMDetector()
            _ = detector.detect(audio_path)

        single_time = time.perf_counter() - start_time

        # Multi-threaded
        gc.collect()
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            detector = LibrosaBPMDetector()
            futures = [executor.submit(detector.detect, audio_path) for audio_path in audio_files[:10]]
            _ = [f.result() for f in futures]

        thread_time = time.perf_counter() - start_time

        # Multi-process
        gc.collect()
        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            detector = LibrosaBPMDetector()
            futures = [executor.submit(detector.detect, audio_path) for audio_path in audio_files[:10]]
            _ = [f.result() for f in futures]

        process_time = time.perf_counter() - start_time

        results.append(
            {
                "num_files": 10,
                "single_thread_time": single_time,
                "multi_thread_time": thread_time,
                "multi_process_time": process_time,
                "thread_speedup": single_time / thread_time,
                "process_speedup": single_time / process_time,
                "max_workers": max_workers,
            }
        )

        return pd.DataFrame(results)

    def benchmark_memory_usage(self, audio_files: List[str]) -> pd.DataFrame:
        """Benchmark memory usage patterns."""
        results = []

        for audio_path in tqdm(audio_files[:5], desc="Benchmarking memory"):
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

            # Measure peak memory for different operations
            operations = {
                "bpm_librosa": LibrosaBPMDetector(),
                "bpm_essentia": EssentiaBPMDetector(),
                "key_librosa": LibrosaKeyDetector(),
                "key_essentia": EssentiaKeyDetector(),
            }

            for op_name, detector in operations.items():
                gc.collect()
                process = psutil.Process()

                # Baseline memory
                mem_before = process.memory_info().rss / (1024 * 1024)

                # Run operation
                try:
                    _ = detector.detect(audio_path)
                    error = None
                except Exception as e:
                    error = str(e)

                # Peak memory
                mem_after = process.memory_info().rss / (1024 * 1024)
                mem_increase = mem_after - mem_before

                results.append(
                    {
                        "filename": Path(audio_path).name,
                        "file_size_mb": file_size_mb,
                        "operation": op_name,
                        "memory_before_mb": mem_before,
                        "memory_after_mb": mem_after,
                        "memory_increase_mb": mem_increase,
                        "memory_ratio": mem_increase / file_size_mb if file_size_mb > 0 else 0,
                        "error": error,
                    }
                )

                gc.collect()

        return pd.DataFrame(results)

    def benchmark_format_compatibility(self, sample_dir: Path) -> pd.DataFrame:
        """Benchmark performance across different audio formats."""
        results = []

        formats = ["mp3", "wav", "flac", "m4a"]

        for fmt in formats:
            format_files = list(sample_dir.glob(f"*.{fmt}"))

            if format_files:
                test_file = str(format_files[0])
                file_size_mb = os.path.getsize(test_file) / (1024 * 1024)

                # Test with each library
                for lib_name, detector in [("librosa", LibrosaBPMDetector()), ("essentia", EssentiaBPMDetector())]:
                    gc.collect()
                    start_time = time.perf_counter()

                    result = detector.detect(test_file)

                    processing_time = time.perf_counter() - start_time

                    results.append(
                        {
                            "format": fmt,
                            "library": lib_name,
                            "file_size_mb": file_size_mb,
                            "processing_time": processing_time,
                            "throughput_mb_per_sec": file_size_mb / processing_time if processing_time > 0 else 0,
                            "success": result["error"] is None,
                            "error": result.get("error"),
                        }
                    )

        return pd.DataFrame(results)

    def generate_performance_report(self, output_dir: Path) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("# Audio Analysis Performance Benchmark Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")

        # System Information
        report.append("\n## System Information\n")
        for key, value in self.system_info.items():
            report.append(f"- **{key}**: {value}")

        # Load benchmark results
        benchmark_files = {
            "loading": output_dir / "benchmark_loading.csv",
            "features": output_dir / "benchmark_features.csv",
            "parallel": output_dir / "benchmark_parallel.csv",
            "memory": output_dir / "benchmark_memory.csv",
            "formats": output_dir / "benchmark_formats.csv",
        }

        # File Loading Performance
        if benchmark_files["loading"].exists():
            df = pd.read_csv(benchmark_files["loading"])
            report.append("\n## File Loading Performance\n")
            report.append(f"- **Librosa Avg Load Time**: {df['librosa_load_time'].mean():.3f}s")
            report.append(f"- **Essentia Avg Load Time**: {df['essentia_load_time'].mean():.3f}s")
            report.append(f"- **Librosa Avg Memory**: {df['librosa_memory_mb'].mean():.1f}MB")
            report.append(f"- **Essentia Avg Memory**: {df['essentia_memory_mb'].mean():.1f}MB")

        # Feature Extraction Performance
        if benchmark_files["features"].exists():
            df = pd.read_csv(benchmark_files["features"])
            report.append("\n## Feature Extraction Performance\n")

            feature_cols = [c for c in df.columns if c.startswith("librosa_") or c.startswith("essentia_")]
            for col in feature_cols:
                if col in df:
                    report.append(f"- **{col}**: {df[col].mean():.4f}s")

        # Parallel Processing
        if benchmark_files["parallel"].exists():
            df = pd.read_csv(benchmark_files["parallel"])
            report.append("\n## Parallel Processing Performance\n")
            if not df.empty:
                row = df.iloc[0]
                report.append(f"- **Single Thread**: {row['single_thread_time']:.2f}s")
                report.append(f"- **Multi Thread Speedup**: {row['thread_speedup']:.2f}x")
                report.append(f"- **Multi Process Speedup**: {row['process_speedup']:.2f}x")

        # Memory Usage
        if benchmark_files["memory"].exists():
            df = pd.read_csv(benchmark_files["memory"])
            report.append("\n## Memory Usage Analysis\n")

            for op in df["operation"].unique():
                op_df = df[df["operation"] == op]
                report.append(f"- **{op}**: {op_df['memory_increase_mb'].mean():.1f}MB average")

        # Format Compatibility
        if benchmark_files["formats"].exists():
            df = pd.read_csv(benchmark_files["formats"])
            report.append("\n## Format Compatibility\n")

            pivot = df.pivot_table(values="success", index="format", columns="library", aggfunc="mean")

            report.append("| Format | Librosa | Essentia |")
            report.append("|--------|---------|----------|")

            for fmt in pivot.index:
                librosa_success = pivot.loc[fmt, "librosa"] * 100 if "librosa" in pivot else 0
                essentia_success = pivot.loc[fmt, "essentia"] * 100 if "essentia" in pivot else 0
                report.append(f"| {fmt} | {librosa_success:.0f}% | {essentia_success:.0f}% |")

        # Recommendations
        report.append("\n## Performance Recommendations\n")
        report.append("1. **For Speed**: Librosa generally faster for simple features")
        report.append("2. **For Memory**: Essentia more memory-efficient for large files")
        report.append("3. **For Parallelism**: Use process pool for CPU-bound operations")
        report.append("4. **For Production**: Target <1s processing for 3-minute tracks")

        return "\n".join(report)

    def create_visualizations(self, output_dir: Path):
        """Create performance visualization charts."""
        # Set style
        sns.set_style("whitegrid")

        # Load data
        loading_df = None
        if (output_dir / "benchmark_loading.csv").exists():
            loading_df = pd.read_csv(output_dir / "benchmark_loading.csv")

        memory_df = None
        if (output_dir / "benchmark_memory.csv").exists():
            memory_df = pd.read_csv(output_dir / "benchmark_memory.csv")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loading Time Comparison
        if loading_df is not None and not loading_df.empty:
            ax = axes[0, 0]
            data = pd.DataFrame(
                {"Librosa": loading_df["librosa_load_time"], "Essentia": loading_df["essentia_load_time"]}
            )
            data.boxplot(ax=ax)
            ax.set_title("File Loading Time Comparison")
            ax.set_ylabel("Time (seconds)")

        # Memory Usage Comparison
        if memory_df is not None and not memory_df.empty:
            ax = axes[0, 1]
            memory_pivot = memory_df.pivot_table(values="memory_increase_mb", index="operation", aggfunc="mean")
            memory_pivot.plot(kind="bar", ax=ax)
            ax.set_title("Memory Usage by Operation")
            ax.set_ylabel("Memory (MB)")
            ax.set_xlabel("Operation")
            ax.legend().remove()

        # Processing Time vs File Size
        if loading_df is not None and not loading_df.empty:
            ax = axes[1, 0]
            ax.scatter(loading_df["file_size_mb"], loading_df["librosa_load_time"], alpha=0.5, label="Librosa")
            ax.scatter(loading_df["file_size_mb"], loading_df["essentia_load_time"], alpha=0.5, label="Essentia")
            ax.set_xlabel("File Size (MB)")
            ax.set_ylabel("Load Time (seconds)")
            ax.set_title("Load Time vs File Size")
            ax.legend()

        # Throughput Comparison
        if loading_df is not None and not loading_df.empty:
            ax = axes[1, 1]
            loading_df["librosa_throughput"] = loading_df["file_size_mb"] / loading_df["librosa_load_time"]
            loading_df["essentia_throughput"] = loading_df["file_size_mb"] / loading_df["essentia_load_time"]

            throughput_data = pd.DataFrame(
                {"Librosa": loading_df["librosa_throughput"], "Essentia": loading_df["essentia_throughput"]}
            )
            throughput_data.boxplot(ax=ax)
            ax.set_title("Processing Throughput")
            ax.set_ylabel("MB/second")

        plt.tight_layout()
        plt.savefig(output_dir / "performance_charts.png", dpi=100, bbox_inches="tight")
        plt.close()


def main():
    """Main benchmarking function."""
    sample_dir = Path(__file__).parent.parent / "sample_data"
    output_dir = Path(__file__).parent

    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        print("Please add sample audio files to the sample_data directory.")
        return

    # Get test files
    audio_files = []
    for ext in ["*.mp3", "*.wav", "*.flac", "*.m4a"]:
        audio_files.extend(sample_dir.glob(ext))

    if not audio_files:
        print("No audio files found in sample_data directory.")
        return

    audio_files = [str(f) for f in audio_files]
    print(f"Found {len(audio_files)} audio files for benchmarking")

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    print("\nSystem Information:")
    for key, value in benchmark.system_info.items():
        print(f"  {key}: {value}")

    # Run benchmarks
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    # 1. File Loading Benchmark
    print("\n1. Benchmarking file loading...")
    loading_df = benchmark.benchmark_file_loading(audio_files)
    loading_df.to_csv(output_dir / "benchmark_loading.csv", index=False)

    # 2. Feature Extraction Benchmark
    print("\n2. Benchmarking feature extraction...")
    features_df = benchmark.benchmark_feature_extraction(audio_files)
    features_df.to_csv(output_dir / "benchmark_features.csv", index=False)

    # 3. Parallel Processing Benchmark
    print("\n3. Benchmarking parallel processing...")
    parallel_df = benchmark.benchmark_parallel_processing(audio_files)
    parallel_df.to_csv(output_dir / "benchmark_parallel.csv", index=False)

    # 4. Memory Usage Benchmark
    print("\n4. Benchmarking memory usage...")
    memory_df = benchmark.benchmark_memory_usage(audio_files)
    memory_df.to_csv(output_dir / "benchmark_memory.csv", index=False)

    # 5. Format Compatibility Benchmark
    print("\n5. Benchmarking format compatibility...")
    formats_df = benchmark.benchmark_format_compatibility(sample_dir)
    formats_df.to_csv(output_dir / "benchmark_formats.csv", index=False)

    # Generate report
    print("\n6. Generating performance report...")
    report = benchmark.generate_performance_report(output_dir)

    with open(output_dir / "performance_report.md", "w") as f:
        f.write(report)

    print(f"\nPerformance report saved to {output_dir / 'performance_report.md'}")

    # Create visualizations
    print("\n7. Creating visualizations...")
    benchmark.create_visualizations(output_dir)
    print(f"Charts saved to {output_dir / 'performance_charts.png'}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if not loading_df.empty:
        print("\nAverage Load Times:")
        print(f"  Librosa: {loading_df['librosa_load_time'].mean():.3f}s")
        print(f"  Essentia: {loading_df['essentia_load_time'].mean():.3f}s")

    if not memory_df.empty:
        print("\nAverage Memory Usage:")
        for op in memory_df["operation"].unique():
            op_df = memory_df[memory_df["operation"] == op]
            print(f"  {op}: {op_df['memory_increase_mb'].mean():.1f}MB")

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
