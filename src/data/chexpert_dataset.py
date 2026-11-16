"""CheXpert multi-label chest X-ray dataset loader.

Reference: Baharoon et al. (2024, arXiv:2312.02366v4) and
notebooks/0_Baseline_CheXpert.ipynb.

Loads danjacobellis/chexpert from HuggingFace and mirrors the reference baseline
protocol:

  * 5 CheXpert competition labels (Atelectasis, Cardiomegaly, Consolidation,
    Edema, Pleural Effusion).
  * Frontal-view filtering.
  * A 10,000-sample (configurable) training subset drawn with a fixed seed.
  * A deterministic held-out test carve-out from that subset, so AUROC is not
    reported on the tiny (234-sample) official validation split alone.
  * U-Zeros / U-Ones uncertainty remapping.

Label schema note
-----------------
danjacobellis/chexpert stores pathology columns as ClassLabel integers:

    0 = unlabeled, 1 = uncertain, 2 = absent (negative), 3 = present (positive)

Set ``dataset.hf_4class_encoding: true`` in configs/chexpert.yaml to interpret
labels this way. For CSV-style encodings, values are expected as
-1.0 (uncertain), 0.0 (absent), 1.0 (present); set the flag to false.
"""

from collections.abc import Sequence
import math
from typing import Any

from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

# 5 CheXpert competition labels used by the base paper's benchmark.
LABEL_NAMES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# danjacobellis/chexpert 4-class integer encoding.
_HF_UNLABELED: float = 0.0
_HF_UNCERTAIN: float = 1.0
_HF_ABSENT: float = 2.0
_HF_PRESENT: float = 3.0


class CheXpertDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace CheXpert splits."""

    def __init__(
        self,
        hf_dataset: Any,
        transforms: Any | None = None,
        label_names: Sequence[str] | None = None,
        uncertainty_policy: str = "u_zeros",
        hf_4class_encoding: bool = True,
    ) -> None:
        """Wraps a HuggingFace CheXpert split into a PyTorch Dataset.

        Args:
            hf_dataset: HuggingFace Dataset split.
            transforms: Optional torchvision or callable transform pipeline.
            label_names: Ordered sequence of target pathology label names.
            uncertainty_policy: Policy for remapping uncertain annotations.
                Must be 'u_zeros' (maps to 0.0) or 'u_ones' (maps to 1.0).
            hf_4class_encoding: Whether labels use the 4-class integer encoding
                (0=unlabeled, 1=uncertain, 2=absent, 3=present).
        """
        if uncertainty_policy not in ("u_zeros", "u_ones"):
            raise ValueError(
                f"uncertainty_policy must be 'u_zeros' or 'u_ones', got {uncertainty_policy!r}"
            )
        self.hf_dataset: Any = hf_dataset
        self.transforms: Any | None = transforms
        self.label_names: list[str] = list(label_names) if label_names is not None else LABEL_NAMES
        self.uncertainty_policy: str = uncertainty_policy
        self.hf_4class_encoding: bool = hf_4class_encoding

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset split."""
        return len(self.hf_dataset)

    def _remap_label(self, val: float | int | None) -> float:
        """Remaps a raw CheXpert annotation to a binary {0.0, 1.0} label.

        Handles both the HuggingFace 4-class integer encoding
        (0=unlabeled, 1=uncertain, 2=absent, 3=present) and the CSV encoding
        (-1.0=uncertain, 0.0=absent, 1.0=present). Uncertain values are resolved
        according to the configured uncertainty policy.

        Args:
            val: Raw annotation value from the dataset record.

        Returns:
            Remapped binary label as a float (0.0 or 1.0).
        """
        if val is None:
            return 0.0
        val_float: float = float(val)
        if math.isnan(val_float):
            return 0.0

        if self.hf_4class_encoding:
            if val_float == _HF_PRESENT:
                return 1.0
            if val_float == _HF_UNCERTAIN:
                return 0.0 if self.uncertainty_policy == "u_zeros" else 1.0
            # unlabeled (0) and absent (2) both map to negative
            return 0.0

        if val_float == -1.0:
            return 0.0 if self.uncertainty_policy == "u_zeros" else 1.0
        return val_float

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves and transforms an image and its label vector.

        Args:
            idx: Integer index of the sample to retrieve.

        Returns:
            Dictionary with 'pixel_values' tensor and 'labels' float tensor.
        """
        item: dict[str, Any] = self.hf_dataset[idx]
        image_data: Any = item.get("image", item.get("img"))
        if not isinstance(image_data, Image.Image):
            image_obj: Image.Image = Image.fromarray(image_data)
        else:
            image_obj = image_data
        image_rgb: Image.Image = image_obj.convert("RGB")

        pixel_values: torch.Tensor = (
            self.transforms(image_rgb) if self.transforms is not None else image_rgb
        )

        labels: torch.Tensor = torch.tensor(
            [self._remap_label(item.get(name, 0.0)) for name in self.label_names],
            dtype=torch.float32,
        )

        return {"pixel_values": pixel_values, "labels": labels}


def _filter_frontal(split: Any) -> Any:
    """Filters a split to frontal views if a view column is present.

    Args:
        split: HuggingFace Dataset split.

    Returns:
        Filtered split (unchanged if no recognized view column exists).
    """
    if "Frontal/Lateral" in split.column_names:
        # danjacobellis/chexpert: ClassLabel 0 = Frontal, 1 = Lateral.
        return split.filter(lambda r: r["Frontal/Lateral"] in (0, "Frontal", "0Frontal"))
    if "view" in split.column_names:
        return split.filter(lambda r: r["view"] == "Frontal")
    return split


def build_chexpert_dataloaders(
    config: dict[str, Any],
    train_transforms: Any,
    val_transforms: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loads CheXpert and builds train/val/test DataLoaders.

    Protocol:
      * Validation uses the full official validation split (model selection).
      * Test is a deterministic carve-out (``subset.test_fraction``) from the
        shuffled training subset, so a genuine held-out test set exists.

    Args:
        config: Configuration dict specifying dataset/subset/training parameters.
        train_transforms: Transform pipeline for the training split.
        val_transforms: Transform pipeline for validation and test splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset_cfg: dict[str, Any] = config.get("dataset", {})
    subset_cfg: dict[str, Any] = config.get("subset", {})
    training_cfg: dict[str, Any] = config.get("training", {})

    repo_name: str = dataset_cfg.get("hf_repo", "danjacobellis/chexpert")
    uncertainty_policy: str = dataset_cfg.get("uncertainty_policy", "u_zeros")
    hf_4class: bool = bool(dataset_cfg.get("hf_4class_encoding", True))
    label_names: list[str] = list(dataset_cfg.get("label_names", LABEL_NAMES))
    frontal_only: bool = bool(dataset_cfg.get("frontal_only", True))

    train_subset_size: int | None = subset_cfg.get("train_size")
    test_fraction: float = float(subset_cfg.get("test_fraction", 0.0))
    seed: int = int(subset_cfg.get("seed", training_cfg.get("seed", 42)))

    batch_size: int = int(training_cfg.get("batch_size", 32))
    num_workers: int = int(training_cfg.get("num_workers", 2))
    pin_memory: bool = bool(training_cfg.get("pin_memory", True))

    dataset_dict: Any = load_dataset(repo_name)
    train_split: Any = dataset_dict["train"]
    val_split: Any = dataset_dict.get("validation") or dataset_dict.get("val")

    if frontal_only:
        train_split = _filter_frontal(train_split)
        val_split = _filter_frontal(val_split)

    # Deterministic subset selection (shuffle with fixed seed, take first N).
    train_split = train_split.shuffle(seed=seed)
    if train_subset_size is not None and train_subset_size < len(train_split):
        train_split = train_split.select(range(train_subset_size))

    # Deterministic held-out test carve-out from the training subset.
    if test_fraction > 0.0:
        num_test: int = int(round(len(train_split) * test_fraction))
        test_split = train_split.select(range(num_test))
        train_split = train_split.select(range(num_test, len(train_split)))
    else:
        # Fall back to the official validation split as the test set.
        test_split = val_split

    train_dataset: CheXpertDataset = CheXpertDataset(
        hf_dataset=train_split,
        transforms=train_transforms,
        label_names=label_names,
        uncertainty_policy=uncertainty_policy,
        hf_4class_encoding=hf_4class,
    )
    val_dataset: CheXpertDataset = CheXpertDataset(
        hf_dataset=val_split,
        transforms=val_transforms,
        label_names=label_names,
        uncertainty_policy=uncertainty_policy,
        hf_4class_encoding=hf_4class,
    )
    test_dataset: CheXpertDataset = CheXpertDataset(
        hf_dataset=test_split,
        transforms=val_transforms,
        label_names=label_names,
        uncertainty_policy=uncertainty_policy,
        hf_4class_encoding=hf_4class,
    )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
