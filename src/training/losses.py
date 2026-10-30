"""Task-specific loss functions for multi-label (CheXpert) and multi-class (Brain Tumor) classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelBCELoss(nn.Module):
    """Binary cross-entropy with logits loss for multi-label classification."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        """Binary cross-entropy with logits loss for multi-label classification.

        Optionally accepts a pos_weight tensor of shape (num_labels,) to up-weight
        positive classes and address class imbalance. Recommended for CheXpert
        where pathology prevalence is low.

        Args:
            pos_weight: Optional weight tensor for positive samples per label.
        """
        super().__init__()
        self.pos_weight: torch.Tensor | None = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes binary cross-entropy loss from predicted logits and targets.

        Args:
            logits: Predicted raw logits tensor of shape (batch_size, num_labels).
            targets: Binary ground truth targets tensor of shape (batch_size, num_labels).

        Returns:
            Scalar loss tensor.
        """
        pw: torch.Tensor | None = (
            self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets.to(device=logits.device, dtype=logits.dtype), pos_weight=pw
        )


class MultiClassCELoss(nn.Module):
    """Cross-entropy loss for multi-class classification."""

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        """Cross-entropy loss for multi-class classification.

        Optionally weighted for class imbalance. Used for Brain Tumor 3-class task.

        Args:
            weight: Optional class weights tensor of shape (num_classes,).
        """
        super().__init__()
        self.weight: torch.Tensor | None = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes multi-class cross-entropy loss.

        Args:
            logits: Predicted class logits tensor of shape (batch_size, num_classes).
            targets: Ground truth class indices tensor of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        w: torch.Tensor | None = (
            self.weight.to(device=logits.device, dtype=logits.dtype)
            if self.weight is not None
            else None
        )
        return F.cross_entropy(logits, targets.to(device=logits.device, dtype=torch.long), weight=w)
