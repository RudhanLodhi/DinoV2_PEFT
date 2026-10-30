"""Classification head modules for multi-label (CheXpert) and multi-class (Brain Tumor) tasks.

Both heads take the CLS token feature vector from DINOv2 backbone as input.
"""

import torch
import torch.nn as nn


class MultiLabelHead(nn.Module):
    """Linear classification head for multi-label binary classification."""

    def __init__(
        self,
        in_features: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        """Linear classification head for multi-label binary classification.

        Raw logits are returned during training for use with BCEWithLogitsLoss.
        Sigmoid is applied only during inference.

        Args:
            in_features: Dimensionality of the input feature vector.
            num_labels: Total number of binary labels to predict.
            dropout: Dropout probability applied prior to the linear projection.
        """
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.linear: nn.Linear = nn.Linear(in_features, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input features to multi-label logits.

        Args:
            x: Input feature tensor of shape (batch_size, in_features).

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        return self.linear(self.dropout(x))


class MultiClassHead(nn.Module):
    """Linear classification head for multi-class classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        """Linear classification head for multi-class classification.

        Returns logits during training; softmax applied externally during inference.

        Args:
            in_features: Dimensionality of the input feature vector.
            num_classes: Total number of mutually exclusive classes.
            dropout: Dropout probability applied prior to the linear projection.
        """
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.linear: nn.Linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input features to multi-class logits.

        Args:
            x: Input feature tensor of shape (batch_size, in_features).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.linear(self.dropout(x))


class DINOv2Classifier(nn.Module):
    """Full classification model combining a DINOv2 backbone with a classification head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        """Combines a DINOv2 feature extractor with a task-specific classification head.

        Args:
            backbone: Pretrained or PEFT-adapted DINOv2 backbone module.
            head: Classification head module (MultiLabelHead or MultiClassHead).
        """
        super().__init__()
        self.backbone: nn.Module = backbone
        self.head: nn.Module = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extracts CLS token representations and computes prediction logits.

        Args:
            pixel_values: Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            Raw prediction logits tensor.
        """
        outputs = self.backbone(pixel_values)
        cls_token: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)
