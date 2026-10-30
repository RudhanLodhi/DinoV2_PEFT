"""Visualization utilities for DINOv2 PEFT experiments.

Produces publication-quality plots for ROC curves, training curves,
and DINOv2 self-attention maps.
"""

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn


def plot_roc_curves(
    labels: np.ndarray,
    scores: np.ndarray,
    label_names: Sequence[str],
    title: str = "ROC Curves",
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Plots per-label ROC curves and mean ROC curve with AUC in legend.

    One curve per label plus mean. Saves to save_path if provided.

    Args:
        labels: Ground truth binary matrix of shape (N, num_labels).
        scores: Predicted continuous scores of shape (N, num_labels).
        label_names: Sequence of string names for each target label.
        title: Plot title string.
        save_path: Optional destination file path to save the generated plot.

    Returns:
        Matplotlib Figure object containing the rendered ROC curves.
    """
    num_labels: int = labels.shape[1]
    fig, ax = plt.subplots(figsize=(9, 7))

    valid_aucs: list[float] = []

    for i in range(num_labels):
        col_labels: np.ndarray = labels[:, i]
        col_scores: np.ndarray = scores[:, i]

        if len(np.unique(col_labels)) < 2:
            continue

        fpr, tpr, _ = roc_curve(col_labels, col_scores)
        auc_val: float = float(roc_auc_score(col_labels, col_scores))
        valid_aucs.append(auc_val)
        ax.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {auc_val:.3f})")

    if valid_aucs:
        mean_auc: float = float(np.mean(valid_aucs))
        ax.plot([], [], " ", label=f"Mean AUC = {mean_auc:.3f}")

    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.5, label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    if save_path is not None:
        out_p: Path = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=300)

    return fig


def plot_training_curves(
    log_df: pd.DataFrame,
    metric_col: str = "val_auroc",
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Plots training loss and validation metric vs epoch from a log DataFrame.

    Expects columns: epoch, train_loss, and the metric_col. Returns Figure.

    Args:
        log_df: DataFrame containing per-epoch metric records.
        metric_col: Column name of the validation evaluation metric to plot.
        title: Overall plot title string.
        save_path: Optional file path for exporting the plot.

    Returns:
        Matplotlib Figure object containing loss and metric curves.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs: Any = (
        log_df["epoch"]
        if "epoch" in log_df.columns
        else list(range(1, len(log_df) + 1))
    )

    if "train_loss" in log_df.columns:
        ax1.plot(epochs, log_df["train_loss"], color="tab:blue", marker="o", label="Train Loss")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Training Loss", color="tab:blue", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="tab:blue")

    if metric_col in log_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(epochs, log_df[metric_col], color="tab:red", marker="s", label=metric_col)
        ax2.set_ylabel(metric_col, color="tab:red", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        out_p: Path = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=300)

    return fig


def plot_attention_maps(
    model: nn.Module,
    processor: Any,
    image: Image.Image,
    patch_size: int = 14,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Extracts and visualises DINOv2 self-attention maps from the last transformer block.

    Reshapes attention weights to the spatial grid of patches and overlays them
    on the original image. Useful for qualitative analysis of what the model
    attends to in X-ray and MRI images. Requires model.config.output_attentions
    or patching forward.

    Shows one subplot per attention head plus the mean attention.

    Args:
        model: Pretrained DINOv2 model or DINOv2Classifier wrapper.
        processor: Preprocessing processor instance for the model.
        image: Source PIL image to inspect.
        patch_size: Square patch size in pixels used by the ViT backbone.
        save_path: Optional destination file path to save the visualization.

    Returns:
        Matplotlib Figure object containing multi-head attention map overlays.
    """
    backbone: nn.Module = model.backbone if hasattr(model, "backbone") else model
    device: torch.device = next(backbone.parameters()).device

    inputs: dict[str, Any] = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs: Any = backbone(**inputs, output_attentions=True)

    attentions: tuple[torch.Tensor, ...] = outputs.attentions
    last_block_attn: torch.Tensor = attentions[-1][0]
    cls_attention: torch.Tensor = last_block_attn[:, 0, 1:]

    pixel_values: torch.Tensor = inputs["pixel_values"]
    img_h: int = int(pixel_values.shape[-2])
    img_w: int = int(pixel_values.shape[-1])
    h_patches: int = img_h // patch_size
    w_patches: int = img_w // patch_size

    num_patches: int = int(cls_attention.shape[1])
    if h_patches * w_patches == num_patches:
        maps_np: np.ndarray = (
            cls_attention.view(-1, h_patches, w_patches)
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
    else:
        side: int = int(math.isqrt(num_patches))
        maps_np = (
            cls_attention[:, : side * side]
            .view(-1, side, side)
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

    num_heads: int = maps_np.shape[0]
    mean_map: np.ndarray = np.mean(maps_np, axis=0, keepdims=True)
    all_maps: np.ndarray = np.concatenate([maps_np, mean_map], axis=0)
    total_subplots: int = num_heads + 1

    ncols: int = 4
    nrows: int = math.ceil(total_subplots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    flat_axes: np.ndarray = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for i in range(num_heads):
        flat_axes[i].imshow(image)
        flat_axes[i].imshow(
            all_maps[i],
            cmap="jet",
            alpha=0.5,
            extent=(0, image.width, image.height, 0),
        )
        flat_axes[i].set_title(f"Head {i + 1}", fontsize=10)
        flat_axes[i].axis("off")

    flat_axes[num_heads].imshow(image)
    flat_axes[num_heads].imshow(
        all_maps[-1],
        cmap="jet",
        alpha=0.5,
        extent=(0, image.width, image.height, 0),
    )
    flat_axes[num_heads].set_title("Mean Attention", fontsize=10)
    flat_axes[num_heads].axis("off")

    for j in range(total_subplots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()

    if save_path is not None:
        out_p: Path = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=300)

    return fig
