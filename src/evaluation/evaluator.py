"""Evaluation loop for DINOv2 PEFT models.

Runs inference on a DataLoader, computes all relevant metrics, saves results
to JSON, and optionally appends a row to the experiment log.
"""

from collections.abc import Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_accuracy, compute_auroc, compute_confusion_matrix, compute_macro_f1


def _to_json_serializable(val: Any) -> Any:
    """Recursively converts NumPy and scientific data types to native JSON types.

    Args:
        val: Arbitrary data value, container, or NumPy array.

    Returns:
        JSON-compliant data representation.
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.floating, float)):
        return None if math.isnan(float(val)) else float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, dict):
        return {k: _to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_json_serializable(v) for v in val]
    return val


class Evaluator:
    """Evaluation runner computing task-specific metrics and saving JSON summaries."""

    def __init__(
        self,
        task: str,
        label_names: Sequence[str] | None = None,
        output_dir: str | Path = "outputs/eval",
    ) -> None:
        """Sets up evaluator for 'multilabel' (CheXpert) or 'multiclass' (Brain Tumor) task.

        label_names used for AUROC labeling. Results are saved to output_dir as JSON.

        Args:
            task: Identifier of evaluation task ('multilabel' or 'multiclass').
            label_names: Optional sequence of label or class names.
            output_dir: Target directory path for exported evaluation JSON files.
        """
        self.task: str = task
        self.label_names: list[str] | None = list(label_names) if label_names is not None else None
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        split: str = "val",
    ) -> dict[str, Any]:
        """Runs full inference loop on dataloader.

        Collects logits and labels, computes metrics, saves results JSON,
        and returns metrics dict. Model should be the full backbone+head nn.Module.
        Puts model in eval mode and restores train mode after. Uses torch.no_grad().

        Args:
            model: Full PyTorch model module comprising backbone and classification head.
            dataloader: DataLoader delivering batches with 'pixel_values' and targets.
            device: Compute device string ('cuda' or 'cpu').
            split: Dataset split identifier used for naming output results file.

        Returns:
            Dictionary containing computed task evaluation metrics.
        """
        target_device: torch.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        was_training: bool = model.training
        model.eval()
        model.to(target_device)

        logits_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values: torch.Tensor = batch["pixel_values"].to(target_device)

                if self.task == "multilabel":
                    target_tensor: torch.Tensor = batch.get("labels", batch.get("label"))
                else:
                    target_tensor = batch.get("label", batch.get("labels"))

                outputs: Any = model(pixel_values)
                if isinstance(outputs, torch.Tensor):
                    batch_logits: torch.Tensor = outputs
                elif hasattr(outputs, "logits"):
                    batch_logits = outputs.logits
                elif hasattr(outputs, "last_hidden_state"):
                    batch_logits = outputs.last_hidden_state[:, 0, :]
                else:
                    batch_logits = outputs

                logits_list.append(batch_logits.detach().cpu().to(torch.float32).numpy())
                labels_list.append(target_tensor.detach().cpu().numpy())

        if was_training:
            model.train()

        all_logits: np.ndarray = np.concatenate(logits_list, axis=0)
        all_labels: np.ndarray = np.concatenate(labels_list, axis=0)

        metrics: dict[str, Any] = {}

        if self.task == "multilabel":
            probabilities: np.ndarray = 1.0 / (1.0 + np.exp(-all_logits))
            auroc_dict: dict[str, Any] = compute_auroc(
                labels=all_labels,
                scores=probabilities,
                label_names=self.label_names,
            )
            binary_preds: np.ndarray = (probabilities >= 0.5).astype(int)
            macro_f1: float = compute_macro_f1(all_labels, binary_preds)

            metrics = {
                "mean_auroc": auroc_dict["mean_auroc"],
                "per_label_auroc": auroc_dict["per_label_auroc"],
                "val_macro_f1" if split == "val" else "test_macro_f1": macro_f1,
            }
        else:
            class_preds: np.ndarray = np.argmax(all_logits, axis=1)
            accuracy: float = compute_accuracy(all_labels, class_preds)
            macro_f1 = compute_macro_f1(all_labels, class_preds)
            cm_dict: dict[str, Any] = compute_confusion_matrix(
                labels=all_labels,
                preds=class_preds,
                class_names=self.label_names,
            )

            metrics = {
                "val_accuracy" if split == "val" else "test_accuracy": accuracy,
                "val_macro_f1" if split == "val" else "test_macro_f1": macro_f1,
                "confusion_matrix": cm_dict["matrix"],
                "classification_report": cm_dict["report"],
            }

        out_file: Path = self.output_dir / f"{split}_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(metrics), f, indent=2)

        return metrics
