#!/usr/bin/env python
"""
Model Evaluation Script
Comprehensive evaluation of mood and genre classification models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


class ModelEvaluationMetrics:
    """Calculate and report model evaluation metrics."""

    @staticmethod
    def calculate_metrics(y_true: list, y_pred: list, labels: list | None = None) -> dict:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional list of label names

        Returns:
            Dictionary of metrics
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"error": "No predictions to evaluate"}

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )

        # Macro and weighted averages
        metrics["precision_macro"] = np.mean(precision)
        metrics["recall_macro"] = np.mean(recall)
        metrics["f1_macro"] = np.mean(f1)

        # Weighted by support
        total_support = np.sum(support)
        if total_support > 0:
            metrics["precision_weighted"] = np.sum(precision * support) / total_support
            metrics["recall_weighted"] = np.sum(recall * support) / total_support
            metrics["f1_weighted"] = np.sum(f1 * support) / total_support

        # Per-class details
        if labels:
            metrics["per_class"] = {}
            for i, label in enumerate(labels):
                metrics["per_class"][label] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        return metrics

    @staticmethod
    def print_classification_report(y_true: list, y_pred: list, labels: list | None = None):
        """Print detailed classification report."""
        if len(y_true) == 0 or len(y_pred) == 0:
            print("No predictions to evaluate")
            return

        print("\nClassification Report:")
        print("=" * 60)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nConfusion Matrix:")
        if labels:
            print(f"Labels: {labels}")
        print(cm)


class MoodGenreEvaluator:
    """Evaluate mood and genre classification models."""

    def __init__(self):
        self.genre_labels = [
            "rock",
            "pop",
            "electronic",
            "jazz",
            "classical",
            "hip-hop",
            "metal",
            "folk",
            "reggae",
            "blues",
        ]

        self.mood_labels = [
            "happy",
            "sad",
            "energetic",
            "relaxed",
            "aggressive",
            "melancholic",
            "peaceful",
            "dark",
        ]

    def evaluate_genre_model(self, predictions_df: pd.DataFrame) -> dict:
        """
        Evaluate genre classification model.

        Args:
            predictions_df: DataFrame with 'actual_genre' and 'predicted_genre' columns

        Returns:
            Evaluation metrics dictionary
        """
        # Filter valid predictions
        valid_df = predictions_df[predictions_df["actual_genre"].notna() & predictions_df["predicted_genre"].notna()]

        if len(valid_df) == 0:
            return {"error": "No valid predictions for evaluation"}

        y_true = valid_df["actual_genre"].tolist()
        y_pred = valid_df["predicted_genre"].tolist()

        # Get unique labels from data
        unique_labels = sorted(set(y_true + y_pred))

        # Calculate metrics
        metrics = ModelEvaluationMetrics.calculate_metrics(y_true, y_pred, unique_labels)

        # Add model-specific metrics
        metrics["model_type"] = "genre_classification"
        metrics["num_classes"] = len(unique_labels)
        metrics["total_samples"] = len(valid_df)

        # Top-k accuracy if confidence scores are available
        if "genre_scores" in valid_df.columns:
            metrics["top_3_accuracy"] = self._calculate_top_k_accuracy(valid_df, "genre_scores", "actual_genre", k=3)
            metrics["top_5_accuracy"] = self._calculate_top_k_accuracy(valid_df, "genre_scores", "actual_genre", k=5)

        return metrics

    def evaluate_mood_model(self, predictions_df: pd.DataFrame) -> dict:
        """
        Evaluate mood classification model.

        Args:
            predictions_df: DataFrame with 'actual_mood' and 'predicted_mood' columns

        Returns:
            Evaluation metrics dictionary
        """
        # Filter valid predictions
        valid_df = predictions_df[predictions_df["actual_mood"].notna() & predictions_df["predicted_mood"].notna()]

        if len(valid_df) == 0:
            return {"error": "No valid predictions for evaluation"}

        y_true = valid_df["actual_mood"].tolist()
        y_pred = valid_df["predicted_mood"].tolist()

        # Get unique labels
        unique_labels = sorted(set(y_true + y_pred))

        # Calculate metrics
        metrics = ModelEvaluationMetrics.calculate_metrics(y_true, y_pred, unique_labels)

        # Add model-specific metrics
        metrics["model_type"] = "mood_classification"
        metrics["num_classes"] = len(unique_labels)
        metrics["total_samples"] = len(valid_df)

        # Top-k accuracy if confidence scores are available
        if "mood_scores" in valid_df.columns:
            metrics["top_2_accuracy"] = self._calculate_top_k_accuracy(valid_df, "mood_scores", "actual_mood", k=2)
            metrics["top_3_accuracy"] = self._calculate_top_k_accuracy(valid_df, "mood_scores", "actual_mood", k=3)

        return metrics

    def _calculate_top_k_accuracy(self, df: pd.DataFrame, scores_col: str, true_col: str, k: int) -> float:
        """Calculate top-k accuracy."""
        correct = 0
        total = 0

        for _, row in df.iterrows():
            if isinstance(row[scores_col], dict):
                scores = row[scores_col]
                true_label = row[true_col]

                # Get top-k predictions
                top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
                top_k_labels = [label for label, _ in top_k]

                if true_label in top_k_labels:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def evaluate_cross_genre_confusion(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze confusion between similar genres.

        Args:
            predictions_df: DataFrame with genre predictions

        Returns:
            DataFrame with confusion analysis
        """
        # Define similar genre groups
        genre_groups = {
            "electronic_family": ["electronic", "techno", "house", "trance", "dnb"],
            "rock_family": ["rock", "metal", "punk", "alternative"],
            "urban": ["hip-hop", "rap", "r&b", "soul"],
            "classical_family": ["classical", "orchestral", "chamber"],
        }

        confusion_analysis = []

        for group_name, genres in genre_groups.items():
            group_df = predictions_df[
                predictions_df["actual_genre"].isin(genres) | predictions_df["predicted_genre"].isin(genres)
            ]

            if len(group_df) > 0:
                within_group_correct = len(
                    group_df[(group_df["actual_genre"].isin(genres)) & (group_df["predicted_genre"].isin(genres))]
                )

                total_group = len(group_df[group_df["actual_genre"].isin(genres)])

                if total_group > 0:
                    confusion_analysis.append(
                        {
                            "genre_group": group_name,
                            "within_group_accuracy": within_group_correct / total_group,
                            "total_samples": total_group,
                        }
                    )

        return pd.DataFrame(confusion_analysis)


def generate_evaluation_report(results_dir: Path) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        results_dir: Directory containing evaluation results

    Returns:
        Markdown-formatted report
    """
    report = []
    report.append("# Audio Analysis Model Evaluation Report\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n")

    # Load results if available
    genre_results = results_dir / "genre_evaluation.json"
    mood_results = results_dir / "mood_evaluation.json"
    performance_results = results_dir / "model_performance.csv"

    # Genre Classification Results
    if genre_results.exists():
        with genre_results.open() as f:
            genre_metrics = json.load(f)

        report.append("\n## Genre Classification Results\n")
        report.append(f"- **Accuracy**: {genre_metrics.get('accuracy', 0):.2%}")
        report.append(f"- **F1 Score (Macro)**: {genre_metrics.get('f1_macro', 0):.3f}")
        report.append(f"- **F1 Score (Weighted)**: {genre_metrics.get('f1_weighted', 0):.3f}")

        if "top_3_accuracy" in genre_metrics:
            report.append(f"- **Top-3 Accuracy**: {genre_metrics['top_3_accuracy']:.2%}")

        # Per-class performance
        if "per_class" in genre_metrics:
            report.append("\n### Per-Genre Performance\n")
            report.append("| Genre | Precision | Recall | F1 Score | Support |")
            report.append("|-------|-----------|--------|----------|---------|")

            for genre, metrics in genre_metrics["per_class"].items():
                report.append(
                    f"| {genre} | {metrics['precision']:.3f} | "
                    f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
                    f"{metrics['support']} |"
                )

    # Mood Classification Results
    if mood_results.exists():
        with mood_results.open() as f:
            mood_metrics = json.load(f)

        report.append("\n## Mood Classification Results\n")
        report.append(f"- **Accuracy**: {mood_metrics.get('accuracy', 0):.2%}")
        report.append(f"- **F1 Score (Macro)**: {mood_metrics.get('f1_macro', 0):.3f}")
        report.append(f"- **F1 Score (Weighted)**: {mood_metrics.get('f1_weighted', 0):.3f}")

        if "top_2_accuracy" in mood_metrics:
            report.append(f"- **Top-2 Accuracy**: {mood_metrics['top_2_accuracy']:.2%}")

    # Model Performance
    if performance_results.exists():
        perf_df = pd.read_csv(performance_results)

        report.append("\n## Model Performance Metrics\n")
        report.append("| Model | Size (MB) | Load Time (s) | Inference Time (s) | Memory (MB) |")
        report.append("|-------|-----------|---------------|-------------------|-------------|")

        for _, row in perf_df.iterrows():
            report.append(
                f"| {row['model_name']} | {row.get('model_size_mb', 0):.1f} | "
                f"{row.get('load_time_sec', 0):.3f} | "
                f"{row.get('avg_inference_time', 0):.3f} | "
                f"{row.get('memory_increase_mb', 0):.1f} |"
            )

    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("Based on the evaluation results:\n")
    report.append("1. **For Genre Classification**: Use Essentia's Discogs EffNet model for best accuracy")
    report.append("2. **For Mood Detection**: Use MusiCNN-based mood models for balanced performance")
    report.append("3. **For Production**: Consider model size and inference time trade-offs")
    report.append("4. **For Accuracy**: Implement ensemble methods combining multiple models")

    return "\n".join(report)


def main():
    """Main evaluation function."""
    print("Model Evaluation System")
    print("=" * 60)

    # Initialize evaluator
    evaluator = MoodGenreEvaluator()

    # Create sample predictions for demonstration
    sample_predictions = pd.DataFrame(
        [
            {
                "actual_genre": "rock",
                "predicted_genre": "rock",
                "genre_scores": {"rock": 0.8, "metal": 0.2},
            },
            {
                "actual_genre": "electronic",
                "predicted_genre": "electronic",
                "genre_scores": {"electronic": 0.9, "techno": 0.1},
            },
            {
                "actual_genre": "jazz",
                "predicted_genre": "blues",
                "genre_scores": {"blues": 0.6, "jazz": 0.4},
            },
            {
                "actual_genre": "classical",
                "predicted_genre": "classical",
                "genre_scores": {"classical": 0.95, "orchestral": 0.05},
            },
            {
                "actual_mood": "happy",
                "predicted_mood": "happy",
                "mood_scores": {"happy": 0.7, "energetic": 0.3},
            },
            {
                "actual_mood": "sad",
                "predicted_mood": "melancholic",
                "mood_scores": {"melancholic": 0.6, "sad": 0.4},
            },
        ]
    )

    # Evaluate genre model
    print("\nEvaluating Genre Classification...")
    genre_metrics = evaluator.evaluate_genre_model(sample_predictions)

    if "error" not in genre_metrics:
        print(f"Genre Accuracy: {genre_metrics['accuracy']:.2%}")
        print(f"F1 Score (Macro): {genre_metrics['f1_macro']:.3f}")

    # Evaluate mood model
    print("\nEvaluating Mood Classification...")
    mood_metrics = evaluator.evaluate_mood_model(sample_predictions)

    if "error" not in mood_metrics:
        print(f"Mood Accuracy: {mood_metrics['accuracy']:.2%}")
        print(f"F1 Score (Macro): {mood_metrics['f1_macro']:.3f}")

    # Save results
    output_dir = Path(__file__).parent

    with Path(output_dir / "genre_evaluation.json").open("w") as f:
        json.dump(genre_metrics, f, indent=2)

    with Path(output_dir / "mood_evaluation.json").open("w") as f:
        json.dump(mood_metrics, f, indent=2)

    # Generate report
    report = generate_evaluation_report(output_dir)

    with Path(output_dir / "evaluation_report.md").open("w") as f:
        f.write(report)

    print(f"\nEvaluation report saved to {output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
