#!/usr/bin/env python
"""
Musical Key Detection Comparison Script
Evaluates key detection accuracy across different libraries and algorithms.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import essentia.standard as es
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


class KeyDetector:
    """Base class for key detection methods."""

    # Mapping of keys to Camelot notation and circle of fifths
    KEY_MAPPINGS = {
        "C major": {"camelot": "8B", "fifths": 0},
        "G major": {"camelot": "9B", "fifths": 1},
        "D major": {"camelot": "10B", "fifths": 2},
        "A major": {"camelot": "11B", "fifths": 3},
        "E major": {"camelot": "12B", "fifths": 4},
        "B major": {"camelot": "1B", "fifths": 5},
        "F# major": {"camelot": "2B", "fifths": 6},
        "Db major": {"camelot": "3B", "fifths": -5},
        "Ab major": {"camelot": "4B", "fifths": -4},
        "Eb major": {"camelot": "5B", "fifths": -3},
        "Bb major": {"camelot": "6B", "fifths": -2},
        "F major": {"camelot": "7B", "fifths": -1},
        "A minor": {"camelot": "8A", "fifths": 0},
        "E minor": {"camelot": "9A", "fifths": 1},
        "B minor": {"camelot": "10A", "fifths": 2},
        "F# minor": {"camelot": "11A", "fifths": 3},
        "C# minor": {"camelot": "12A", "fifths": 4},
        "G# minor": {"camelot": "1A", "fifths": 5},
        "Eb minor": {"camelot": "2A", "fifths": 6},
        "Bb minor": {"camelot": "3A", "fifths": -5},
        "F minor": {"camelot": "4A", "fifths": -4},
        "C minor": {"camelot": "5A", "fifths": -3},
        "G minor": {"camelot": "6A", "fifths": -2},
        "D minor": {"camelot": "7A", "fifths": -1},
    }

    def __init__(self, name: str):
        self.name = name

    def detect(self, audio_path: str) -> Dict:
        """Detect key from audio file."""
        raise NotImplementedError

    @staticmethod
    def normalize_key_name(key: str) -> str:
        """Normalize key names to standard format."""
        # Handle various key notations
        key = key.strip()

        # Replace flat/sharp symbols
        key = key.replace("♭", "b").replace("♯", "#")

        # Normalize major/minor notation
        if "maj" in key.lower() and "major" not in key.lower():
            key = key.replace("maj", " major").replace("Maj", " major")
        if "min" in key.lower() and "minor" not in key.lower():
            key = key.replace("min", " minor").replace("Min", " minor")

        # Handle uppercase/lowercase
        parts = key.split()
        if len(parts) >= 2:
            parts[0] = parts[0].capitalize()
            parts[1] = parts[1].lower()
            key = " ".join(parts)

        return key


class EssentiaKeyDetector(KeyDetector):
    """Key detection using Essentia's KeyExtractor."""

    def __init__(self):
        super().__init__("Essentia_KeyExtractor")

    def detect(self, audio_path: str) -> Dict:
        """Detect key using Essentia's KeyExtractor."""
        start_time = time.time()

        try:
            # Load audio
            audio = es.MonoLoader(filename=audio_path)()

            # Extract key using KeyExtractor
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)

            # Format key name
            key_name = f"{key} {scale}"
            key_normalized = self.normalize_key_name(key_name)

            # Alternative: Use HPCP (Harmonic Pitch Class Profile)
            # This provides more detailed harmonic information
            frame_size = 4096
            hop_size = 2048

            windowing = es.Windowing(type="blackmanharris62")
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks(
                orderBy="magnitude", magnitudeThreshold=0.00001, minFrequency=20, maxFrequency=3500, maxPeaks=60
            )
            hpcp = es.HPCP()
            key_from_hpcp = es.Key()

            # Compute HPCP for additional analysis
            hpcps = []
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                windowed = windowing(frame)
                spec = spectrum(windowed)
                freqs, mags = spectral_peaks(spec)
                hpcp_frame = hpcp(freqs, mags)
                hpcps.append(hpcp_frame)

            # Average HPCP across frames
            mean_hpcp = np.mean(hpcps, axis=0) if hpcps else np.zeros(12)

            # Get key from HPCP
            hpcp_key, hpcp_scale, hpcp_strength, _ = key_from_hpcp(mean_hpcp.astype(np.float32))
            hpcp_key_name = f"{hpcp_key} {hpcp_scale}"

            processing_time = time.time() - start_time

            return {
                "key": key_normalized,
                "scale": scale,
                "confidence": float(strength),
                "key_raw": key,
                "hpcp_key": self.normalize_key_name(hpcp_key_name),
                "hpcp_confidence": float(hpcp_strength),
                "chromagram": mean_hpcp.tolist(),
                "processing_time": processing_time,
                "error": None,
            }

        except Exception as e:
            return {
                "key": None,
                "scale": None,
                "confidence": None,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }


class LibrosaKeyDetector(KeyDetector):
    """Key detection using Librosa's chroma features."""

    def __init__(self):
        super().__init__("Librosa_Chroma")

        # Krumhansl-Schmuckler key profiles
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        self.key_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

    def detect(self, audio_path: str) -> Dict:
        """Detect key using Librosa's chroma features."""
        start_time = time.time()

        try:
            # Load audio
            y, sr = librosa.load(audio_path)

            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

            # Average chroma across time
            mean_chroma = np.mean(chroma, axis=1)

            # Correlate with key profiles
            major_corrs = []
            minor_corrs = []

            for shift in range(12):
                # Rotate chroma vector
                rotated_chroma = np.roll(mean_chroma, shift)

                # Correlate with profiles
                major_corr = np.corrcoef(rotated_chroma, self.major_profile)[0, 1]
                minor_corr = np.corrcoef(rotated_chroma, self.minor_profile)[0, 1]

                major_corrs.append(major_corr)
                minor_corrs.append(minor_corr)

            # Find best match
            major_corrs = np.array(major_corrs)
            minor_corrs = np.array(minor_corrs)

            max_major = np.max(major_corrs)
            max_minor = np.max(minor_corrs)

            if max_major > max_minor:
                key_idx = np.argmax(major_corrs)
                scale = "major"
                confidence = max_major
            else:
                key_idx = np.argmax(minor_corrs)
                scale = "minor"
                confidence = max_minor

            key = self.key_names[key_idx]
            key_name = f"{key} {scale}"
            key_normalized = self.normalize_key_name(key_name)

            # Also calculate using different chroma types for comparison
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

            processing_time = time.time() - start_time

            return {
                "key": key_normalized,
                "scale": scale,
                "confidence": float(confidence),
                "key_raw": key,
                "chromagram": mean_chroma.tolist(),
                "chroma_stft": np.mean(chroma_stft, axis=1).tolist(),
                "chroma_cens": np.mean(chroma_cens, axis=1).tolist(),
                "processing_time": processing_time,
                "error": None,
            }

        except Exception as e:
            return {
                "key": None,
                "scale": None,
                "confidence": None,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }


def compare_key_detection(audio_files: List[str], ground_truth: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compare key detection across multiple methods.

    Args:
        audio_files: List of audio file paths
        ground_truth: Dictionary mapping filenames to actual keys

    Returns:
        DataFrame with comparison results
    """
    detectors = [
        EssentiaKeyDetector(),
        LibrosaKeyDetector(),
    ]

    results = []

    for audio_path in tqdm(audio_files, desc="Processing files"):
        filename = Path(audio_path).name

        for detector in detectors:
            result = detector.detect(audio_path)

            # Add metadata
            result["filename"] = filename
            result["detector"] = detector.name

            # Add ground truth if available
            if ground_truth and filename in ground_truth:
                actual_key = ground_truth[filename]
                result["actual_key"] = actual_key

                if result["key"]:
                    # Check exact match
                    result["exact_match"] = result["key"] == actual_key

                    # Check if same tonic (ignoring major/minor)
                    actual_tonic = actual_key.split()[0] if " " in actual_key else actual_key
                    detected_tonic = result["key"].split()[0] if " " in result["key"] else result["key"]
                    result["tonic_match"] = actual_tonic == detected_tonic

                    # Check if relative key (e.g., C major vs A minor)
                    result["is_relative"] = check_relative_keys(result["key"], actual_key)

                    # Check if parallel key (same tonic, different mode)
                    result["is_parallel"] = check_parallel_keys(result["key"], actual_key)

            results.append(result)

    return pd.DataFrame(results)


def check_relative_keys(key1: str, key2: str) -> bool:
    """Check if two keys are relative (e.g., C major and A minor)."""
    relatives = {
        "C major": "A minor",
        "A minor": "C major",
        "G major": "E minor",
        "E minor": "G major",
        "D major": "B minor",
        "B minor": "D major",
        "A major": "F# minor",
        "F# minor": "A major",
        "E major": "C# minor",
        "C# minor": "E major",
        "B major": "G# minor",
        "G# minor": "B major",
        "F# major": "D# minor",
        "D# minor": "F# major",
        "F major": "D minor",
        "D minor": "F major",
        "Bb major": "G minor",
        "G minor": "Bb major",
        "Eb major": "C minor",
        "C minor": "Eb major",
        "Ab major": "F minor",
        "F minor": "Ab major",
        "Db major": "Bb minor",
        "Bb minor": "Db major",
    }

    return relatives.get(key1) == key2


def check_parallel_keys(key1: str, key2: str) -> bool:
    """Check if two keys are parallel (same tonic, different mode)."""
    if " " not in key1 or " " not in key2:
        return False

    tonic1, mode1 = key1.split(" ", 1)
    tonic2, mode2 = key2.split(" ", 1)

    return tonic1 == tonic2 and mode1 != mode2


def analyze_key_detection_results(df: pd.DataFrame) -> Dict:
    """Analyze key detection results."""
    summary = {}

    for detector in df["detector"].unique():
        detector_df = df[df["detector"] == detector]
        valid_df = detector_df[detector_df["error"].isna()]

        summary[detector] = {
            "success_rate": len(valid_df) / len(detector_df) * 100,
            "avg_processing_time": valid_df["processing_time"].mean(),
            "avg_confidence": valid_df["confidence"].mean() if "confidence" in valid_df else None,
        }

        # If ground truth is available
        if "actual_key" in df.columns:
            valid_with_truth = valid_df[valid_df["actual_key"].notna()]

            if len(valid_with_truth) > 0:
                summary[detector].update(
                    {
                        "exact_match_rate": valid_with_truth["exact_match"].mean() * 100,
                        "tonic_match_rate": valid_with_truth["tonic_match"].mean() * 100,
                        "relative_key_rate": valid_with_truth["is_relative"].mean() * 100,
                        "parallel_key_rate": valid_with_truth["is_parallel"].mean() * 100,
                    }
                )

    return summary


def main():
    """Main function for testing key detection."""
    sample_dir = Path(__file__).parent.parent / "sample_data"

    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        print("Please add sample audio files to the sample_data directory.")
        return

    # Get all audio files
    audio_files = []
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
        if "actual_key" in gt_df.columns:
            ground_truth = dict(zip(gt_df["filename"], gt_df["actual_key"]))
            print(f"Loaded ground truth for {len(ground_truth)} files")

    # Run comparison
    print("\nComparing key detection methods...")
    results_df = compare_key_detection([str(f) for f in audio_files], ground_truth)

    # Save results
    output_dir = Path(__file__).parent
    results_df.to_csv(output_dir / "key_comparison_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'key_comparison_results.csv'}")

    # Analyze and print summary
    summary = analyze_key_detection_results(results_df)

    print("\n" + "=" * 60)
    print("KEY DETECTION COMPARISON SUMMARY")
    print("=" * 60)

    for detector, stats in summary.items():
        print(f"\n{detector}:")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Avg Processing Time: {stats['avg_processing_time']:.3f}s")

        if stats["avg_confidence"] is not None:
            print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")

        if "exact_match_rate" in stats:
            print(f"  Exact Match Rate: {stats['exact_match_rate']:.1f}%")
            print(f"  Tonic Match Rate: {stats['tonic_match_rate']:.1f}%")
            print(f"  Relative Key Rate: {stats['relative_key_rate']:.1f}%")
            print(f"  Parallel Key Rate: {stats['parallel_key_rate']:.1f}%")


if __name__ == "__main__":
    main()
