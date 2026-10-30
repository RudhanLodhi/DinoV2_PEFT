"""Image transforms for DINOv2 input preprocessing and augmentation."""

from typing import Sequence
import torchvision.transforms as transforms

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def build_transforms(
    image_size: int = 224,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    augment: bool = False,
) -> transforms.Compose:
    """Builds standard torchvision transformation pipelines for DINOv2 inputs.

    For training (augment=True), applies random horizontal flips and random resized
    crops within the [0.8, 1.0] scale range before conversion to tensor and normalization.
    For validation and testing (augment=False), applies a standard DINOv2 evaluation
    pipeline with a 1.143x aspect-preserving resize followed by a central crop to
    the target resolution before conversion to tensor and normalization.

    Args:
        image_size: Target square image dimension in pixels.
        mean: Sequence of per-channel normalization means. Defaults to ImageNet statistics.
        std: Sequence of per-channel normalization standard deviations. Defaults to ImageNet statistics.
        augment: Whether to apply data augmentation transformations.

    Returns:
        Composed torchvision transform pipeline.
    """
    norm_mean: list[float] = list(mean) if mean is not None else IMAGENET_MEAN
    norm_std: list[float] = list(std) if std is not None else IMAGENET_STD

    if augment:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

    resize_dim: int = int(image_size * 1.143)
    return transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
