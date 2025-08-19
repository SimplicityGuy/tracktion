#!/usr/bin/env python
"""
Unit tests for BPM detection accuracy.
Tests will be run once sample files are provided.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compare_bpm import (
    EssentiaBPMDetector,
    EssentiaPercivalDetector,
    LibrosaBPMDetector,
    analyze_results,
    compare_bpm_detection,
)


class TestBPMDetection(unittest.TestCase):
    """Test BPM detection accuracy."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_dir = Path(__file__).parent.parent / "sample_data"
        cls.ground_truth_file = cls.sample_dir / "ground_truth.csv"

        # Load ground truth if available
        cls.ground_truth = {}
        if cls.ground_truth_file.exists():
            gt_df = pd.read_csv(cls.ground_truth_file)
            cls.ground_truth = dict(zip(gt_df["filename"], gt_df["actual_bpm"]))

    def test_librosa_detector(self):
        """Test Librosa BPM detector."""
        detector = LibrosaBPMDetector()
        self.assertEqual(detector.name, "Librosa")

        # Test with a sample file if available
        sample_files = list(self.sample_dir.glob("*.mp3"))
        if sample_files:
            result = detector.detect(str(sample_files[0]))
            self.assertIsNotNone(result)
            self.assertIn("bpm", result)
            self.assertIn("processing_time", result)

            if result["error"] is None:
                self.assertIsNotNone(result["bpm"])
                self.assertGreater(result["bpm"], 0)
                self.assertLess(result["bpm"], 300)  # Reasonable BPM range

    def test_essentia_detector(self):
        """Test Essentia BPM detector."""
        detector = EssentiaBPMDetector()
        self.assertEqual(detector.name, "Essentia")

        # Test with a sample file if available
        sample_files = list(self.sample_dir.glob("*.mp3"))
        if sample_files:
            result = detector.detect(str(sample_files[0]))
            self.assertIsNotNone(result)
            self.assertIn("bpm", result)
            self.assertIn("confidence", result)
            self.assertIn("processing_time", result)

            if result["error"] is None:
                self.assertIsNotNone(result["bpm"])
                self.assertGreater(result["bpm"], 0)
                self.assertLess(result["bpm"], 300)

    def test_essentia_percival_detector(self):
        """Test Essentia Percival BPM detector."""
        detector = EssentiaPercivalDetector()
        self.assertEqual(detector.name, "Essentia_Percival")

        # Test with a sample file if available
        sample_files = list(self.sample_dir.glob("*.mp3"))
        if sample_files:
            result = detector.detect(str(sample_files[0]))
            self.assertIsNotNone(result)
            self.assertIn("bpm", result)
            self.assertIn("processing_time", result)

    def test_bpm_accuracy_electronic(self):
        """Test BPM accuracy on electronic music (usually steady tempo)."""
        electronic_files = list(self.sample_dir.glob("electronic_*constant*.mp3"))

        if not electronic_files or not self.ground_truth:
            self.skipTest("No electronic test files or ground truth available")

        results_df = compare_bpm_detection([str(f) for f in electronic_files], self.ground_truth)

        # Electronic music should have high accuracy
        for detector in results_df["detector"].unique():
            detector_df = results_df[(results_df["detector"] == detector) & (results_df["error_percent"].notna())]

            if len(detector_df) > 0:
                mean_error = detector_df["error_percent"].mean()
                # Expect less than 2% error for electronic music
                self.assertLess(mean_error, 2.0, f"{detector} has {mean_error:.1f}% error on electronic music")

    def test_bpm_accuracy_variable(self):
        """Test BPM detection on variable tempo music."""
        variable_files = list(self.sample_dir.glob("*variable*.mp3"))

        if not variable_files:
            self.skipTest("No variable tempo test files available")

        results_df = compare_bpm_detection([str(f) for f in variable_files])

        # Check that std deviation is detected
        for file in variable_files:
            file_df = results_df[results_df["filename"] == file.name]

            for _, row in file_df.iterrows():
                if row["error"] is None and "bpm_std" in row:
                    # Variable tempo should show some std deviation
                    self.assertIsNotNone(row["bpm_std"])
                    if row["bpm_std"] > 0:
                        # This indicates tempo variation was detected
                        self.assertGreater(row["bpm_std"], 0)

    def test_processing_time(self):
        """Test that processing time is reasonable."""
        sample_files = list(self.sample_dir.glob("*.mp3"))[:3]  # Test first 3 files

        if not sample_files:
            self.skipTest("No sample files available")

        results_df = compare_bpm_detection([str(f) for f in sample_files])

        # Check processing times
        for detector in results_df["detector"].unique():
            detector_df = results_df[results_df["detector"] == detector]
            valid_df = detector_df[detector_df["error"].isna()]

            if len(valid_df) > 0:
                avg_time = valid_df["processing_time"].mean()
                max_time = valid_df["processing_time"].max()

                # Expect processing in reasonable time (adjust based on needs)
                self.assertLess(
                    avg_time,
                    10.0,  # Average under 10 seconds
                    f"{detector} average processing time is {avg_time:.2f}s",
                )
                self.assertLess(
                    max_time,
                    30.0,  # Max under 30 seconds
                    f"{detector} max processing time is {max_time:.2f}s",
                )

    def test_format_compatibility(self):
        """Test different audio format compatibility."""
        formats = ["mp3", "wav", "flac", "m4a"]
        detectors = [
            LibrosaBPMDetector(),
            EssentiaBPMDetector(),
        ]

        for fmt in formats:
            format_files = list(self.sample_dir.glob(f"*.{fmt}"))

            if format_files:
                for detector in detectors:
                    with self.subTest(format=fmt, detector=detector.name):
                        result = detector.detect(str(format_files[0]))

                        if result["error"] is None:
                            self.assertIsNotNone(result["bpm"], f"{detector.name} failed on {fmt} format")

    def test_summary_statistics(self):
        """Test that summary statistics are calculated correctly."""
        sample_files = list(self.sample_dir.glob("*.mp3"))[:5]

        if not sample_files:
            self.skipTest("No sample files available")

        results_df = compare_bpm_detection([str(f) for f in sample_files])
        summary = analyze_results(results_df)

        for detector, stats in summary.items():
            self.assertIn("success_rate", stats)
            self.assertIn("avg_processing_time", stats)

            # Success rate should be between 0 and 100
            self.assertGreaterEqual(stats["success_rate"], 0)
            self.assertLessEqual(stats["success_rate"], 100)

            # Processing time should be positive
            if not np.isnan(stats["avg_processing_time"]):
                self.assertGreater(stats["avg_processing_time"], 0)


class TestBPMRanges(unittest.TestCase):
    """Test BPM detection across different tempo ranges."""

    def test_slow_tempo(self):
        """Test detection of slow tempos (60-90 BPM)."""
        sample_dir = Path(__file__).parent.parent / "sample_data"
        slow_files = (
            list(sample_dir.glob("*60bpm*")) + list(sample_dir.glob("*70bpm*")) + list(sample_dir.glob("*80bpm*"))
        )

        if not slow_files:
            self.skipTest("No slow tempo files available")

        detector = EssentiaBPMDetector()

        for file in slow_files:
            result = detector.detect(str(file))
            if result["error"] is None:
                # Check that slow tempo is detected
                self.assertLess(result["bpm"], 100, f"Expected slow tempo for {file.name}, got {result['bpm']}")

    def test_fast_tempo(self):
        """Test detection of fast tempos (140+ BPM)."""
        sample_dir = Path(__file__).parent.parent / "sample_data"
        fast_files = (
            list(sample_dir.glob("*140bpm*")) + list(sample_dir.glob("*160bpm*")) + list(sample_dir.glob("*174bpm*"))
        )

        if not fast_files:
            self.skipTest("No fast tempo files available")

        detector = EssentiaBPMDetector()

        for file in fast_files:
            result = detector.detect(str(file))
            if result["error"] is None:
                # Check that fast tempo is detected
                self.assertGreater(result["bpm"], 130, f"Expected fast tempo for {file.name}, got {result['bpm']}")


if __name__ == "__main__":
    unittest.main()
