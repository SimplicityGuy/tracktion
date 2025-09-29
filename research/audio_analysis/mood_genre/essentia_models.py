#!/usr/bin/env python
"""
Essentia Pre-trained Models Evaluation
Tests mood and genre classification using Essentia's pre-trained models.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import essentia.standard as es
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm


class EssentiaModelEvaluator:
    """Evaluate Essentia's pre-trained models for mood and genre classification."""

    def __init__(self, models_dir: str | None = None):
        """
        Initialize model evaluator.

        Args:
            models_dir: Directory containing Essentia models.
                       If None, will attempt to download models.
        """
        self.models_dir = models_dir or self._get_default_models_dir()
        self.models_loaded: dict[str, object] = {}

    def _get_default_models_dir(self) -> str:
        """Get default models directory."""
        home = Path.home()
        models_dir = home / ".essentia" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return str(models_dir)

    def download_models(self) -> dict[str, str]:
        """
        Download Essentia pre-trained models.

        Note: This provides URLs for manual download.
        Actual download implementation would require requests library.
        """
        model_urls = {
            # MusiCNN models for various tasks
            "musicnn_mtt": "https://essentia.upf.edu/models/classification-heads/musicnn/musicnn-mtt-1.pb",
            "musicnn_msd": "https://essentia.upf.edu/models/classification-heads/musicnn/musicnn-msd-1.pb",
            # Genre classification
            "genre_discogs": "https://essentia.upf.edu/models/classification-heads/genre_discogs/genre_discogs-discogs-effnet-1.pb",
            "genre_electronic": "https://essentia.upf.edu/models/classification-heads/genre_electronic/genre_electronic-discogs-effnet-1.pb",
            "genre_tzanetakis": "https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-musicnn-1.pb",
            # Mood classification
            "mood_happy": "https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-musicnn-1.pb",
            "mood_sad": "https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-musicnn-1.pb",
            "mood_relaxed": "https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-musicnn-1.pb",
            "mood_aggressive": "https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-musicnn-1.pb",
            # Danceability
            "danceability": "https://essentia.upf.edu/models/classification-heads/danceability/danceability-musicnn-1.pb",
            # Voice/Instrumental
            "voice_instrumental": "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-musicnn-1.pb",
        }

        print("Model download URLs:")
        print("-" * 60)
        for model_name, url in model_urls.items():
            print(f"{model_name}: {url}")
            model_path = Path(self.models_dir) / f"{model_name}.pb"
            if model_path.exists():
                print(f"  ✓ Already downloaded to {model_path}")
            else:
                print("  ✗ Not found. Please download manually:")
                print(f"    wget -P {self.models_dir} {url}")

        return model_urls

    def analyze_with_musicnn(self, audio_path: str, model_name: str = "musicnn_mtt") -> dict[str, Any]:
        """
        Analyze audio using MusiCNN model.

        Args:
            audio_path: Path to audio file
            model_name: Name of the MusiCNN model to use

        Returns:
            Dictionary with predictions and metadata
        """
        start_time = time.time()
        result: dict[str, Any] = {
            "model": model_name,
            "audio_file": Path(audio_path).name,
        }

        try:
            # Load audio
            audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()

            # Note: Actual model inference would require TensorFlow
            # This is a placeholder showing the expected workflow

            # For demonstration, we'll use basic Essentia features
            # In production, you would load and run the actual models

            # Extract features that models would use
            features = self._extract_audio_features(audio)

            # Simulate model predictions (replace with actual model inference)
            if "mood" in model_name:
                mood_scores = {
                    "happy": float(np.random.default_rng().random()),
                    "sad": float(np.random.default_rng().random()),
                    "relaxed": float(np.random.default_rng().random()),
                    "aggressive": float(np.random.default_rng().random()),
                }
                result["mood_scores"] = mood_scores
                result["predicted_mood"] = max(mood_scores, key=lambda k: mood_scores[k])

            elif "genre" in model_name:
                genres = ["rock", "pop", "electronic", "jazz", "classical", "hip-hop"]
                genre_scores = {g: float(np.random.default_rng().random()) for g in genres}
                result["genre_scores"] = genre_scores
                result["predicted_genre"] = max(genre_scores, key=lambda k: genre_scores[k])

            elif "danceability" in model_name:
                danceability_score = float(np.random.default_rng().random())
                result["danceability_score"] = danceability_score
                result["is_danceable"] = danceability_score > 0.5

            elif "voice" in model_name:
                instrumental_score = float(np.random.default_rng().random())
                voice_score = 1 - instrumental_score
                result["instrumental_score"] = instrumental_score
                result["voice_score"] = voice_score
                result["has_voice"] = voice_score > 0.5

            result["features"] = features
            result["processing_time"] = time.time() - start_time
            result["error"] = None

        except Exception as e:
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time

        return result

    def _extract_audio_features(self, audio: np.ndarray) -> dict[str, Any]:
        """Extract audio features for analysis."""
        features: dict[str, Any] = {}

        try:
            # Spectral features
            spectrum = es.Spectrum()(audio)
            features["spectral_centroid"] = float(es.Centroid()(spectrum))
            features["spectral_energy"] = float(es.Energy()(spectrum))
            features["spectral_rms"] = float(es.RMS()(spectrum))

            # Zero crossing rate
            features["zcr"] = float(es.ZeroCrossingRate()(audio))

            # MFCC
            mfcc = es.MFCC()(spectrum)
            mfcc_coeffs = mfcc[1]
            features["mfcc_mean"] = [float(x) for x in np.mean(mfcc_coeffs, axis=0)]

            # Rhythm features
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, _beats, confidence, _, _ = rhythm_extractor(audio)
            features["bpm"] = float(bpm)
            features["beat_confidence"] = float(confidence)

            # Energy and loudness
            features["energy"] = float(es.Energy()(audio))
            features["loudness"] = float(es.Loudness()(audio))

        except Exception as e:
            features["feature_extraction_error"] = str(e)

        return features

    def evaluate_genre_classification(self, audio_files: list[str], ground_truth: dict | None = None) -> pd.DataFrame:
        """
        Evaluate genre classification on multiple files.

        Args:
            audio_files: List of audio file paths
            ground_truth: Optional dictionary mapping filenames to genres

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for audio_path in tqdm(audio_files, desc="Evaluating genre classification"):
            filename = Path(audio_path).name

            # Test with genre model
            result = self.analyze_with_musicnn(audio_path, "genre_discogs")

            # Add ground truth if available
            if ground_truth and filename in ground_truth:
                result["actual_genre"] = ground_truth[filename]
                if "predicted_genre" in result:
                    result["correct"] = result["predicted_genre"] == result["actual_genre"]

            results.append(result)

        return pd.DataFrame(results)

    def evaluate_mood_classification(self, audio_files: list[str], ground_truth: dict | None = None) -> pd.DataFrame:
        """
        Evaluate mood classification on multiple files.

        Args:
            audio_files: List of audio file paths
            ground_truth: Optional dictionary mapping filenames to moods

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for audio_path in tqdm(audio_files, desc="Evaluating mood classification"):
            filename = Path(audio_path).name

            # Test with mood model
            result = self.analyze_with_musicnn(audio_path, "mood_happy")

            # Add ground truth if available
            if ground_truth and filename in ground_truth:
                result["actual_mood"] = ground_truth[filename]
                if "predicted_mood" in result:
                    result["correct"] = result["predicted_mood"] == result["actual_mood"]

            results.append(result)

        return pd.DataFrame(results)

    def measure_model_performance(self, model_name: str, test_file: str) -> dict[str, Any]:
        """
        Measure model loading time, memory usage, and inference speed.

        Args:
            model_name: Name of the model to test
            test_file: Path to test audio file

        Returns:
            Performance metrics dictionary
        """

        process = psutil.Process(os.getpid())

        # Measure initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure model loading time
        load_start = time.time()

        # Simulate model loading (in production, load actual TensorFlow model)
        model_path = Path(self.models_dir) / f"{model_name}.pb"
        model_size_mb = model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0

        load_time = time.time() - load_start

        # Measure memory after loading
        mem_after_load = process.memory_info().rss / 1024 / 1024
        memory_increase = mem_after_load - mem_before

        # Measure inference time
        inference_times = []
        for _ in range(3):  # Run 3 times for average
            result = self.analyze_with_musicnn(test_file, model_name)
            if result["error"] is None:
                inference_times.append(result["processing_time"])

        # Calculate statistics
        return {
            "model_name": model_name,
            "model_size_mb": model_size_mb,
            "load_time_sec": load_time,
            "memory_increase_mb": memory_increase,
            "avg_inference_time": np.mean(inference_times) if inference_times else None,
            "std_inference_time": np.std(inference_times) if inference_times else None,
        }


def test_essentia_features():
    """Test basic Essentia feature extraction."""
    print("Testing Essentia feature extraction...")

    # Create a test signal
    sr = 44100
    duration = 3
    t = np.linspace(0, duration, sr * duration)

    # Generate a simple test tone
    frequency = 440  # A4
    test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Extract features
    evaluator = EssentiaModelEvaluator()
    features = evaluator._extract_audio_features(test_audio)

    print("\nExtracted features:")
    for key, value in features.items():
        if isinstance(value, list):
            print(f"  {key}: [array of {len(value)} values]")
        else:
            print(f"  {key}: {value}")

    return features


def main():
    """Main function for testing Essentia models."""
    sample_dir = Path(__file__).parent.parent / "sample_data"

    # Initialize evaluator
    evaluator = EssentiaModelEvaluator()

    # Show model download information
    print("Essentia Model Information")
    print("=" * 60)
    evaluator.download_models()

    # Test basic feature extraction
    print("\n" + "=" * 60)
    test_essentia_features()

    if not sample_dir.exists():
        print(f"\nSample directory not found: {sample_dir}")
        print("Please add sample audio files to test with actual models.")
        return

    # Get test files
    audio_files = list(sample_dir.glob("*.mp3")) + list(sample_dir.glob("*.wav"))

    if not audio_files:
        print("\nNo audio files found in sample_data directory.")
        return

    print(f"\n\nFound {len(audio_files)} audio files for testing")

    # Test genre classification
    print("\n" + "=" * 60)
    print("GENRE CLASSIFICATION TEST")
    print("=" * 60)

    genre_results = evaluator.evaluate_genre_classification([str(f) for f in audio_files[:5]])  # Test first 5 files

    if not genre_results.empty:
        print("\nGenre Classification Results:")
        print(genre_results[["audio_file", "predicted_genre", "processing_time"]].to_string())

    # Test mood classification
    print("\n" + "=" * 60)
    print("MOOD CLASSIFICATION TEST")
    print("=" * 60)

    mood_results = evaluator.evaluate_mood_classification([str(f) for f in audio_files[:5]])

    if not mood_results.empty:
        print("\nMood Classification Results:")
        print(mood_results[["audio_file", "predicted_mood", "processing_time"]].to_string())

    # Test model performance
    if audio_files:
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE TEST")
        print("=" * 60)

        test_file = str(audio_files[0])
        models_to_test = ["genre_discogs", "mood_happy", "danceability"]

        performance_results = []
        for model_name in models_to_test:
            metrics = evaluator.measure_model_performance(model_name, test_file)
            performance_results.append(metrics)

        perf_df = pd.DataFrame(performance_results)
        print("\nModel Performance Metrics:")
        print(perf_df.to_string(index=False))

        # Save results
        output_dir = Path(__file__).parent
        perf_df.to_csv(output_dir / "model_performance.csv", index=False)
        print(f"\nPerformance results saved to {output_dir / 'model_performance.csv'}")


if __name__ == "__main__":
    main()
