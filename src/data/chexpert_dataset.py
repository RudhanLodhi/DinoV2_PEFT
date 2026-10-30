"""CheXpert 14-label multi-label chest X-ray dataset loader.

Loads from danjacobellis/chexpert on HuggingFace. Supports uncertainty policy
(u_zeros/u_ones), frontal-only filter, and configurable training subset (10k, 20k,
or full). Validation and test splits are always kept at full size.
"""

from collections.abc import Sequence
import math
from typing import Any

from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class CheXpertDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace CheXpert splits."""

    def __init__(
        self,
        hf_dataset: Any,
        transforms: Any | None = None,
        label_names: Sequence[str] | None = None,
        uncertainty_policy: str = "u_zeros",
    ) -> None:
        """Wraps a HuggingFace CheXpert split into a PyTorch Dataset.

        Applies uncertainty remapping to binary labels.

        Args:
            hf_dataset: HuggingFace Dataset split.
            transforms: Optional torchvision or callable transform.
            label_names: Ordered sequence of target pathology label names.
            uncertainty_policy: Policy for remapping uncertain annotations (-1.0).
                Must be 'u_zeros' (maps to 0.0) or 'u_ones' (maps to 1.0).
        """
        self.hf_dataset: Any = hf_dataset
        self.transforms: Any | None = transforms
        self.label_names: list[str] = list(label_names) if label_names is not None else LABEL_NAMES
        self.uncertainty_policy: str = uncertainty_policy

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset split."""
        return len(self.hf_dataset)

    def _remap_label(self, val: float | int | None) -> float:
        """Remaps uncertainty labels according to the configured uncertainty policy.

        Maps -1.0 to 0.0 for u_zeros or -1.0 to 1.0 for u_ones. Passes through
        0.0 and 1.0 unchanged. Treats None or NaN as 0.0.

        Args:
            val: Raw annotation value from dataset record.

        Returns:
            Remapped floating point binary label (0.0 or 1.0).
        """
        if val is None:
            return 0.0
        val_float: float = float(val)
        if math.isnan(val_float):
            return 0.0
        if val_float == -1.0:
            return 0.0 if self.uncertainty_policy == "u_zeros" else 1.0
        return val_float

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves and transforms an image and its 14-label target vector.

        Args:
            idx: Integer index of the sample to retrieve.

        Returns:
            Dictionary containing 'pixel_values' tensor and 'labels' float tensor.
        """
        item: dict[str, Any] = self.hf_dataset[idx]
        image_data: Any = item.get("image", item.get("img"))
        if not isinstance(image_data, Image.Image):
            image_obj: Image.Image = Image.fromarray(image_data)
        else:
            image_obj = image_data
        image_rgb: Image.Image = image_obj.convert("RGB")

        pixel_values: torch.Tensor = (
            self.transforms(image_rgb)
            if self.transforms is not None
            else image_rgb
        )

        labels: torch.Tensor = torch.tensor(
            [self._remap_label(item.get(name, 0.0)) for name in self.label_names],
            dtype=torch.float32,
        )

        return {"pixel_values": pixel_values, "labels": labels}


def build_chexpert_dataloaders(
    config: dict[str, Any],
    train_transforms: Any,
    val_transforms: Any,
) -> tuple[DataLoader, DataLoader]:
    """Loads CheXpert from HuggingFace, applies subset sampling, and builds DataLoaders.

    Args:
        config: Configuration dictionary specifying dataset, subset, and training parameters.
        train_transforms: Transform pipeline for the training split.
        val_transforms: Transform pipeline for the validation split.

    Returns:
        Tuple containing (train_loader, val_loader).
    """
    repo_name: str = config.get("dataset", {}).get("hf_repo", "danjacobellis/chexpert")
    uncertainty_policy: str = config.get("dataset", {}).get("uncertainty_policy", "u_zeros")
    frontal_only: bool = config.get("dataset", {}).get("frontal_only", True)
    train_subset_size: int | None = config.get("subset", {}).get("train_size")
    seed: int = config.get("training", {}).get("seed", config.get("seed", 42))
    batch_size: int = config.get("training", {}).get("batch_size", 32)
    num_workers: int = config.get("training", {}).get("num_workers", 2)
    pin_memory: bool = config.get("training", {}).get("pin_memory", True)

    dataset_dict: Any = load_dataset(repo_name)
    train_split: Any = dataset_dict["train"]
    val_split: Any = dataset_dict["validation"] if "validation" in dataset_dict else dataset_dict["val"]

    if frontal_only:
        if "Frontal/Lateral" in train_split.column_names:
            train_split = train_split.filter(lambda r: r["Frontal/Lateral"] == "Frontal")
            val_split = val_split.filter(lambda r: r["Frontal/Lateral"] == "Frontal")
        elif "view" in train_split.column_names:
            train_split = train_split.filter(lambda r: r["view"] == "Frontal")
            val_split = val_split.filter(lambda r: r["view"] == "Frontal")

    if train_subset_size is not None and train_subset_size < len(train_split):
        train_split = train_split.shuffle(seed=seed).select(range(train_subset_size))

    train_dataset: CheXpertDataset = CheXpertDataset(
        hf_dataset=train_split,
        transforms=train_transforms,
        label_names=LABEL_NAMES,
        uncertainty_policy=uncertainty_policy,
    )
    val_dataset: CheXpertDataset = CheXpertDataset(
        hf_dataset=val_split,
        transforms=val_transforms,
        label_names=LABEL_NAMES,
        uncertainty_policy=uncertainty_policy,
    )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
