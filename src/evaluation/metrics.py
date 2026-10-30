"""Evaluation metrics for multi-label (CheXpert) and multi-class (Brain Tumor) classification tasks."""

from collections.abc import Sequence
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def compute_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Computes per-label and mean AUROC for multi-label classification.

    Labels and scores must have shape (N, num_labels). Returns dict with
    'mean_auroc' (float) and 'per_label_auroc' (dict mapping label name to
    AUROC float). Skips labels with only one class present in the batch
    (returns NaN for that label).

    Args:
        labels: Ground truth binary indicator matrix of shape (N, num_labels).
        scores: Predicted continuous probability or logit scores of shape (N, num_labels).
        label_names: Optional sequence of human-readable label names.

    Returns:
        Dictionary containing 'mean_auroc' and 'per_label_auroc'.
    """
    num_labels: int = labels.shape[1]
    names: list[str] = (
        list(label_names)
        if label_names is not None
        else [f"label_{i}" for i in range(num_labels)]
    )

    per_label_auroc: dict[str, float] = {}
    valid_scores: list[float] = []

    for i in range(num_labels):
        col_labels: np.ndarray = labels[:, i]
        col_scores: np.ndarray = scores[:, i]
        name: str = names[i]

        if len(np.unique(col_labels)) < 2:
            per_label_auroc[name] = float("nan")
            continue

        try:
            score: float = float(roc_auc_score(col_labels, col_scores))
            per_label_auroc[name] = score
            valid_scores.append(score)
        except ValueError:
            per_label_auroc[name] = float("nan")

    mean_auroc: float = float(np.mean(valid_scores)) if valid_scores else float("nan")

    return {
        "mean_auroc": mean_auroc,
        "per_label_auroc": per_label_auroc,
    }


def compute_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """Computes macro-averaged F1 score.

    For multi-label tasks, preds should be binarized at threshold 0.5 before
    calling. For multi-class, preds are class indices.

    Args:
        labels: Ground truth binary matrix or class index vector.
        preds: Predicted binary matrix or class index vector.

    Returns:
        Macro-averaged F1 score.
    """
    return float(f1_score(labels, preds, average="macro", zero_division=0.0))


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """Computes overall classification accuracy.

    For multi-class tasks. labels and preds are integer class index arrays of shape (N,).

    Args:
        labels: Ground truth integer class indices of shape (N,).
        preds: Predicted integer class indices of shape (N,).

    Returns:
        Accuracy proportion.
    """
    return float(accuracy_score(labels, preds))


def compute_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Returns confusion matrix and per-class metrics (precision, recall, F1).

    Useful for Brain Tumor 3-class analysis. Returns dict with 'matrix'
    (np.ndarray) and 'report' (str from classification_report).

    Args:
        labels: Ground truth integer class indices.
        preds: Predicted integer class indices.
        class_names: Optional sequence of class names.

    Returns:
        Dictionary containing 'matrix' and 'report'.
    """
    cm: np.ndarray = confusion_matrix(labels, preds)
    cnames: list[str] | None = list(class_names) if class_names is not None else None
    report: str = classification_report(labels, preds, target_names=cnames, zero_division=0.0)

    return {
        "matrix": cm,
        "report": report,
    }
