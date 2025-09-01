#!/usr/bin/env python
"""
BPM Detection Comparison Script
Evaluates BPM detection accuracy and performance across multiple libraries.
"""

from __future__ import annotations

import time
from pathlib import Path

import essentia.standard as es
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


class BPMDetector:
    """Base class for BPM detection methods."""

    def __init__(self, name: str):
        self.name = name

    def detect(self, audio_path: str) -> dict:
        """Detect BPM from audio file."""
        raise NotImplementedError


class LibrosaBPMDetector(BPMDetector):
    """BPM detection using librosa."""

    def __init__(self):
        super().__init__("Librosa")

    def detect(self, audio_path: str) -> dict:
        """Detect BPM using librosa's beat tracker."""
        start_time = time.time()

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Get tempo over time windows for variable tempo detection
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

            # Dynamic tempo estimation
            dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None)

            # Calculate statistics
            if len(dtempo) > 0:
                tempo_mean = float(np.mean(dtempo))
                tempo_std = float(np.std(dtempo))
                tempo_min = float(np.min(dtempo))
                tempo_max = float(np.max(dtempo))
            else:
                tempo_mean = float(tempo)
                tempo_std = 0.0
                tempo_min = float(tempo)
                tempo_max = float(tempo)

            processing_time = time.time() - start_time

            return {
                "bpm": float(tempo),
                "bpm_mean": tempo_mean,
                "bpm_std": tempo_std,
                "bpm_min": tempo_min,
                "bpm_max": tempo_max,
                "num_beats": len(beats),
                "processing_time": processing_time,
                "error": None,
            }

        except Exception as e:
            return {
                "bpm": None,
                "bpm_mean": None,
                "bpm_std": None,
                "bpm_min": None,
                "bpm_max": None,
                "num_beats": None,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }


class EssentiaBPMDetector(BPMDetector):
    """BPM detection using Essentia."""

    def __init__(self):
        super().__init__("Essentia")

    def detect(self, audio_path: str) -> dict:
        """Detect BPM using Essentia's RhythmExtractor2013."""
        start_time = time.time()

        try:
            # Load audio
            audio = es.MonoLoader(filename=audio_path)()

            # Use RhythmExtractor2013 for comprehensive rhythm analysis
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

            # Calculate tempo variations
            if len(beats_intervals) > 0:
                # Convert beat intervals to instantaneous BPM
                instantaneous_bpm = 60.0 / beats_intervals
                tempo_mean = float(np.mean(instantaneous_bpm))
                tempo_std = float(np.std(instantaneous_bpm))
                tempo_min = float(np.min(instantaneous_bpm))
                tempo_max = float(np.max(instantaneous_bpm))
            else:
                tempo_mean = float(bpm)
                tempo_std = 0.0
                tempo_min = float(bpm)
                tempo_max = float(bpm)

            processing_time = time.time() - start_time

            return {
                "bpm": float(bpm),
                "bpm_mean": tempo_mean,
                "bpm_std": tempo_std,
                "bpm_min": tempo_min,
                "bpm_max": tempo_max,
                "num_beats": len(beats),
                "confidence": float(beats_confidence),
                "processing_time": processing_time,
                "error": None,
            }

        except Exception as e:
            return {
                "bpm": None,
                "bpm_mean": None,
                "bpm_std": None,
                "bpm_min": None,
                "bpm_max": None,
                "num_beats": None,
                "confidence": None,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }


class EssentiaPercivalDetector(BPMDetector):
    """BPM detection using Essentia's PercivalBpmEstimator."""

    def __init__(self):
        super().__init__("Essentia_Percival")

    def detect(self, audio_path: str) -> dict:
        """Detect BPM using Essentia's PercivalBpmEstimator."""
        start_time = time.time()

        try:
            # Load audio
            audio = es.MonoLoader(filename=audio_path)()

            # Use PercivalBpmEstimator (alternative algorithm)
            bpm_estimator = es.PercivalBpmEstimator()
            bpm = bpm_estimator(audio)

            # Also use BeatTrackerMultiFeature for beat positions
            beat_tracker = es.BeatTrackerMultiFeature()
            beats, confidence = beat_tracker(audio)

            processing_time = time.time() - start_time

            return {
                "bpm": float(bpm),
                "bpm_mean": float(bpm),  # Percival gives single estimate
                "bpm_std": 0.0,
                "bpm_min": float(bpm),
                "bpm_max": float(bpm),
                "num_beats": len(beats),
                "confidence": float(confidence) if confidence else None,
                "processing_time": processing_time,
                "error": None,
            }

        except Exception as e:
            return {
                "bpm": None,
                "bpm_mean": None,
                "bpm_std": None,
                "bpm_min": None,
                "bpm_max": None,
                "num_beats": None,
                "confidence": None,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }


def compare_bpm_detection(audio_files: list[str], ground_truth: dict | None = None) -> pd.DataFrame:
    """
    Compare BPM detection across multiple libraries.

    Args:
        audio_files: List of audio file paths
        ground_truth: Dictionary mapping filenames to actual BPM values

    Returns:
        DataFrame with comparison results
    """
    detectors = [
        LibrosaBPMDetector(),
        EssentiaBPMDetector(),
        EssentiaPercivalDetector(),
    ]

    results = []

    for audio_path in tqdm(audio_files, desc="Processing files"):
        filename = Path(audio_path).name

        for detector in detectors:
            result = detector.detect(audio_path)

            # Add metadata
            result["filename"] = filename
            result["detector"] = detector.name
            result["file_size_mb"] = Path(audio_path).stat().st_size / (1024 * 1024)

            # Add ground truth if available
            if ground_truth and filename in ground_truth:
                actual_bpm = ground_truth[filename]
                result["actual_bpm"] = actual_bpm
                if result["bpm"] is not None:
                    result["error"] = abs(result["bpm"] - actual_bpm)
                    result["error_percent"] = (abs(result["bpm"] - actual_bpm) / actual_bpm) * 100

            results.append(result)

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> dict:
    """Analyze and summarize BPM detection results."""
    summary = {}

    for detector in df["detector"].unique():
        detector_df = df[df["detector"] == detector]

        # Filter out errors
        valid_df = detector_df[detector_df["error"].isna()]

        summary[detector] = {
            "success_rate": len(valid_df) / len(detector_df) * 100,
            "avg_processing_time": valid_df["processing_time"].mean(),
            "std_processing_time": valid_df["processing_time"].std(),
        }

        # If ground truth is available
        if "actual_bpm" in df.columns:
            valid_with_truth = valid_df[valid_df["actual_bpm"].notna()]
            if len(valid_with_truth) > 0:
                summary[detector].update(
                    {
                        "mean_absolute_error": valid_with_truth["error"].mean(),
                        "mean_percent_error": valid_with_truth["error_percent"].mean(),
                        "max_error": valid_with_truth["error"].max(),
                        "within_2_bpm": (valid_with_truth["error"] <= 2).mean() * 100,
                        "within_5_percent": (valid_with_truth["error_percent"] <= 5).mean() * 100,
                    }
                )

    return summary


def main():
    """Main function for testing BPM detection."""
    sample_dir = Path(__file__).parent.parent / "sample_data"

    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        print("Please add sample audio files to the sample_data directory.")
        return

    # Get all audio files
    audio_files: list[Path] = []
    for ext in ["*.mp3", "*.wav", "*.flac", "*.m4a"]:
        audio_files.extend(sample_dir.glob(ext))

    if not audio_files:
        print("No audio files found in sample_data directory.")
        return

    print(f"Found {len(audio_files)} audio files")

    # Load ground truth if available
    ground_truth = {}
    ground_truth_file = sample_dir / "ground_truth.csv"
    if ground_truth_file.exists():
        gt_df = pd.read_csv(ground_truth_file)
        # Handle BPM ranges (e.g., "120-140") by taking the average
        for _, row in gt_df.iterrows():
            bpm_str = str(row["actual_bpm"])
            if "-" in bpm_str:
                # For ranges, take the average
                start, end = map(float, bpm_str.split("-"))
                ground_truth[row["filename"]] = (start + end) / 2
            else:
                ground_truth[row["filename"]] = float(bpm_str)
        print(f"Loaded ground truth for {len(ground_truth)} files")

    # Run comparison
    print("\nComparing BPM detection methods...")
    results_df = compare_bpm_detection([str(f) for f in audio_files], ground_truth)

    # Save results
    output_dir = Path(__file__).parent
    results_df.to_csv(output_dir / "bpm_comparison_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'bpm_comparison_results.csv'}")

    # Analyze and print summary
    summary = analyze_results(results_df)

    print("\n" + "=" * 60)
    print("BPM DETECTION COMPARISON SUMMARY")
    print("=" * 60)

    for detector, stats in summary.items():
        print(f"\n{detector}:")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Avg Processing Time: {stats['avg_processing_time']:.3f}s")

        if "mean_absolute_error" in stats:
            print(f"  Mean Absolute Error: {stats['mean_absolute_error']:.2f} BPM")
            print(f"  Mean Percent Error: {stats['mean_percent_error']:.1f}%")
            print(f"  Within 2 BPM: {stats['within_2_bpm']:.1f}%")
            print(f"  Within 5% Error: {stats['within_5_percent']:.1f}%")


if __name__ == "__main__":
    main()
