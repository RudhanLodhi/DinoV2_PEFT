"""Image transforms for DINOv2 input preprocessing and augmentation.

Pipelines match the reference baselines in notebooks/0_Baseline_*.ipynb and the
protocol of Baharoon et al. (2024, arXiv:2312.02366v4):

  Train: RandomResizedCrop(size, scale=(0.75, 1.0)) + RandomHorizontalFlip(0.5)
  Eval:  Resize(int(size * 1.14)) + CenterCrop(size)
  Both:  ToTensor + ImageNet normalization
"""

from collections.abc import Sequence
from typing import Any

import torchvision.transforms as transforms

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class MRIPercentileClip:
    """Clips MRI intensities to a percentile window and rescales to [0, 255].

    Per Baharoon et al. (2024), MRI slices are clipped to the [5th, 95th]
    percentile range before feature extraction to reduce the influence of
    extreme intensity values. Matches 0_Baseline_BrainTumor.ipynb.
    """

    def __init__(self, low_pct: float = 5.0, high_pct: float = 95.0) -> None:
        """Initialises the percentile clipper.

        Args:
            low_pct: Lower percentile boundary (0-100).
            high_pct: Upper percentile boundary (0-100).
        """
        self.low_pct: float = float(low_pct)
        self.high_pct: float = float(high_pct)

    def __call__(self, img: Any) -> Any:
        """Applies percentile clipping and rescales to uint8 [0, 255].

        Args:
            img: Input PIL image.

        Returns:
            Preprocessed PIL RGB image with clipped intensities.
        """
        import numpy as np
        from PIL import Image

        arr: Any = np.array(img, dtype=np.float32)
        lo: float = float(np.percentile(arr, self.low_pct))
        hi: float = float(np.percentile(arr, self.high_pct))
        if hi > lo:
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


def build_brain_tumor_transforms(
    image_size: int = 224,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
    crop_scale_min: float = 0.75,
    crop_scale_max: float = 1.0,
    resize_size: int = 256,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Builds (train, val) transforms for Brain Tumor MRI following the baseline.

    Args:
        image_size: Target square image dimension.
        mean: Per-channel normalization means (defaults to ImageNet statistics).
        std: Per-channel normalization standard deviations.
        low_pct: Lower MRI clipping percentile.
        high_pct: Upper MRI clipping percentile.
        crop_scale_min: Lower RandomResizedCrop scale bound.
        crop_scale_max: Upper RandomResizedCrop scale bound.
        resize_size: Resize dimension used before CenterCrop for eval.

    Returns:
        Tuple of (train_transforms, val_transforms).
    """
    norm_mean: list[float] = list(mean) if mean is not None else IMAGENET_MEAN
    norm_std: list[float] = list(std) if std is not None else IMAGENET_STD

    train_tf = transforms.Compose(
        [
            MRIPercentileClip(low_pct=low_pct, high_pct=high_pct),
            transforms.RandomResizedCrop(
                image_size,
                scale=(crop_scale_min, crop_scale_max),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    val_tf = transforms.Compose(
        [
            MRIPercentileClip(low_pct=low_pct, high_pct=high_pct),
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    return train_tf, val_tf


def build_transforms(
    image_size: int = 224,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    augment: bool = False,
    crop_scale_min: float = 0.75,
    crop_scale_max: float = 1.0,
    resize_ratio: float = 1.14,
) -> transforms.Compose:
    """Builds standard torchvision transformation pipelines for DINOv2 inputs.

    For training (augment=True), applies random resized crops within
    [crop_scale_min, crop_scale_max] and random horizontal flips before tensor
    conversion and normalization. For validation/testing (augment=False),
    applies an aspect-preserving resize followed by a central crop, matching the
    paper's evaluation protocol.

    Args:
        image_size: Target square image dimension in pixels.
        mean: Per-channel normalization means. Defaults to ImageNet statistics.
        std: Per-channel normalization standard deviations. Defaults to ImageNet statistics.
        augment: Whether to apply data augmentation (training) transformations.
        crop_scale_min: Lower bound of the RandomResizedCrop scale range.
        crop_scale_max: Upper bound of the RandomResizedCrop scale range.
        resize_ratio: Multiplier applied to image_size for the eval Resize step.

    Returns:
        Composed torchvision transform pipeline.
    """
    norm_mean: list[float] = list(mean) if mean is not None else IMAGENET_MEAN
    norm_std: list[float] = list(std) if std is not None else IMAGENET_STD

    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(crop_scale_min, crop_scale_max),
                    ratio=(0.75, 1.333),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

    resize_dim: int = int(image_size * resize_ratio)
    return transforms.Compose(
        [
            transforms.Resize(resize_dim, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
