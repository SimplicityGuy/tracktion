#!/usr/bin/env python
"""
Temporal BPM Analysis Script
Analyzes BPM changes over time windows to detect tempo variations.
"""

import time
from pathlib import Path
from typing import Dict, List

import essentia.standard as es
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


class TemporalBPMAnalyzer:
    """Analyze BPM changes over time windows."""

    def __init__(self, window_size: float = 10.0, hop_size: float = 5.0):
        """
        Initialize temporal BPM analyzer.

        Args:
            window_size: Window size in seconds for BPM calculation
            hop_size: Hop size in seconds between windows
        """
        self.window_size = window_size
        self.hop_size = hop_size

    def analyze_with_librosa(self, audio_path: str) -> Dict:
        """Analyze temporal BPM using librosa."""
        start_time = time.time()

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr

        # Calculate onset strength
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Window parameters in frames
        window_frames = int(self.window_size * sr / hop_length)
        hop_frames = int(self.hop_size * sr / hop_length)

        # Calculate BPM for each window
        tempos = []
        times = []
        confidences = []

        for i in range(0, len(onset_env) - window_frames, hop_frames):
            window = onset_env[i : i + window_frames]

            # Estimate tempo for this window
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=window, sr=sr, hop_length=hop_length, start_bpm=120.0, units="time"
            )

            # Calculate autocorrelation for confidence
            autocorr = np.correlate(window, window, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(autocorr, height=0.1)
            if len(peaks) > 0:
                confidence = properties["peak_heights"][0]
            else:
                confidence = 0.0

            tempos.append(float(tempo))
            times.append(i * hop_length / sr)
            confidences.append(float(confidence))

        # Calculate statistics
        tempos = np.array(tempos)

        result = {
            "method": "librosa_temporal",
            "duration": duration,
            "window_size": self.window_size,
            "hop_size": self.hop_size,
            "times": times,
            "tempos": tempos.tolist(),
            "confidences": confidences,
            "start_bpm": float(tempos[0]) if len(tempos) > 0 else None,
            "end_bpm": float(tempos[-1]) if len(tempos) > 0 else None,
            "mean_bpm": float(np.mean(tempos)),
            "std_bpm": float(np.std(tempos)),
            "min_bpm": float(np.min(tempos)),
            "max_bpm": float(np.max(tempos)),
            "tempo_variation": float(np.max(tempos) - np.min(tempos)),
            "is_variable": float(np.std(tempos)) > 2.0,  # Consider variable if std > 2 BPM
            "processing_time": time.time() - start_time,
        }

        return result

    def analyze_with_essentia(self, audio_path: str) -> Dict:
        """Analyze temporal BPM using Essentia."""
        start_time = time.time()

        # Load audio
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()
        sr = 44100  # Essentia default
        duration = len(audio) / sr

        # Window parameters in samples
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)

        # Calculate BPM for each window
        tempos = []
        times = []
        confidences = []

        rhythm_extractor = es.RhythmExtractor2013()

        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i : i + window_samples]

            try:
                # Extract rhythm for this window
                bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(window)

                tempos.append(float(bpm))
                times.append(i / sr)
                confidences.append(float(beats_confidence))

            except Exception:
                # Some windows might be too short or problematic
                continue

        # Calculate statistics
        if len(tempos) > 0:
            tempos = np.array(tempos)

            result = {
                "method": "essentia_temporal",
                "duration": duration,
                "window_size": self.window_size,
                "hop_size": self.hop_size,
                "times": times,
                "tempos": tempos.tolist(),
                "confidences": confidences,
                "start_bpm": float(tempos[0]),
                "end_bpm": float(tempos[-1]),
                "mean_bpm": float(np.mean(tempos)),
                "std_bpm": float(np.std(tempos)),
                "min_bpm": float(np.min(tempos)),
                "max_bpm": float(np.max(tempos)),
                "tempo_variation": float(np.max(tempos) - np.min(tempos)),
                "is_variable": float(np.std(tempos)) > 2.0,
                "processing_time": time.time() - start_time,
            }
        else:
            result = {
                "method": "essentia_temporal",
                "duration": duration,
                "error": "No valid windows processed",
                "processing_time": time.time() - start_time,
            }

        return result

    def plot_temporal_analysis(self, results: List[Dict], output_path: str = None):
        """Plot temporal BPM analysis results."""
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))

        if len(results) == 1:
            axes = [axes]

        for idx, result in enumerate(results):
            ax = axes[idx]

            if "error" in result:
                ax.text(0.5, 0.5, f"Error: {result['error']}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(result["method"])
                continue

            times = result["times"]
            tempos = result["tempos"]
            confidences = result.get("confidences", [1.0] * len(tempos))

            # Plot tempo over time
            ax.plot(times, tempos, "b-", label="BPM", linewidth=2)

            # Add confidence as transparency if available
            if confidences and len(confidences) == len(tempos):
                ax2 = ax.twinx()
                ax2.plot(times, confidences, "g--", alpha=0.5, label="Confidence")
                ax2.set_ylabel("Confidence", color="g")
                ax2.tick_params(axis="y", labelcolor="g")

            # Add statistics
            ax.axhline(
                y=result["mean_bpm"], color="r", linestyle="--", alpha=0.5, label=f"Mean: {result['mean_bpm']:.1f}"
            )

            # Shade std deviation
            mean = result["mean_bpm"]
            std = result["std_bpm"]
            ax.fill_between(times, mean - std, mean + std, color="gray", alpha=0.2, label=f"±1 STD: {std:.1f}")

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("BPM")
            ax.set_title(f"{result['method']} - Variable: {result['is_variable']}")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        return fig


def analyze_tempo_changes(audio_path: str, window_sizes: List[float] = None) -> pd.DataFrame:
    """
    Analyze tempo changes with different window sizes.

    Args:
        audio_path: Path to audio file
        window_sizes: List of window sizes in seconds

    Returns:
        DataFrame with analysis results
    """
    if window_sizes is None:
        window_sizes = [5.0, 10.0, 20.0, 30.0]

    results = []

    for window_size in window_sizes:
        analyzer = TemporalBPMAnalyzer(window_size=window_size, hop_size=window_size / 2)

        # Analyze with librosa
        librosa_result = analyzer.analyze_with_librosa(audio_path)
        librosa_result["window_size_sec"] = window_size
        results.append(librosa_result)

        # Analyze with Essentia
        essentia_result = analyzer.analyze_with_essentia(audio_path)
        essentia_result["window_size_sec"] = window_size
        results.append(essentia_result)

    # Convert to DataFrame for easy comparison
    df = pd.DataFrame(results)

    # Select relevant columns for summary
    summary_cols = [
        "method",
        "window_size_sec",
        "mean_bpm",
        "std_bpm",
        "min_bpm",
        "max_bpm",
        "tempo_variation",
        "is_variable",
        "processing_time",
    ]

    return df[summary_cols]


def main():
    """Main function for temporal BPM analysis."""
    sample_dir = Path(__file__).parent.parent / "sample_data"

    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        print("Please add sample audio files to the sample_data directory.")
        return

    # Look for variable tempo test files
    variable_files = (
        list(sample_dir.glob("*variable*.mp3"))
        + list(sample_dir.glob("*variable*.wav"))
        + list(sample_dir.glob("*transition*.mp3"))
    )

    if not variable_files:
        print("No variable tempo files found.")
        print("Looking for any audio file...")
        audio_files = list(sample_dir.glob("*.mp3")) + list(sample_dir.glob("*.wav"))
        if audio_files:
            variable_files = [audio_files[0]]
        else:
            print("No audio files found in sample_data directory.")
            return

    print(f"Analyzing {len(variable_files)} file(s) for tempo variations...")

    for audio_file in variable_files:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {audio_file.name}")
        print("=" * 60)

        # Analyze with different window sizes
        df = analyze_tempo_changes(str(audio_file))
        print("\nTemporal Analysis Results:")
        print(df.to_string(index=False))

        # Create detailed temporal analysis
        analyzer = TemporalBPMAnalyzer(window_size=10.0, hop_size=5.0)

        results = [analyzer.analyze_with_librosa(str(audio_file)), analyzer.analyze_with_essentia(str(audio_file))]

        # Plot results
        output_path = Path(__file__).parent / f"temporal_{audio_file.stem}.png"
        analyzer.plot_temporal_analysis(results, str(output_path))

        # Save detailed results
        detailed_df = pd.DataFrame(results)
        output_csv = Path(__file__).parent / f"temporal_{audio_file.stem}.csv"
        detailed_df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to {output_csv}")


if __name__ == "__main__":
    main()
