from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels: list[int], predictions: list[int], num_classes: int) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    false_positive_rate = []
    for index in range(num_classes):
        false_positives = matrix[:, index].sum() - matrix[index, index]
        true_negatives = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        denominator = false_positives + true_negatives
        false_positive_rate.append(float(false_positives / denominator) if denominator else 0.0)
    majority = max(np.bincount(labels, minlength=num_classes)) / len(labels) if labels else 0.0
    return {
        "accuracy": accuracy_score(labels, predictions) if labels else 0.0,
        "macro_f1": float(np.mean(f1)),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "false_positive_rate": false_positive_rate,
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
        "random_baseline": 1 / num_classes,
        "majority_baseline": float(majority),
    }


def aggregate_video_predictions(video_ids: list[str], probabilities: np.ndarray) -> dict[str, int]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for video_id, probability in zip(video_ids, probabilities):
        grouped[video_id].append(probability)
    return {video_id: int(np.mean(values, axis=0).argmax()) for video_id, values in grouped.items()}


def binary_video_metrics(labels: list[int], drowning_behavior_scores: list[float],
                         threshold: float = 0.5) -> dict[str, object]:
    """Evaluate video-level scores; scores are not calibrated risk probabilities."""
    if len(labels) != len(drowning_behavior_scores):
        raise ValueError("labels and drowning_behavior_scores must have equal length")
    metrics = classification_metrics(labels, [int(score >= threshold) for score in drowning_behavior_scores], 2)
    return {"score_name": "drowning_behavior_score", "score_is_calibrated_risk_probability": False,
            "threshold": threshold, **metrics}
