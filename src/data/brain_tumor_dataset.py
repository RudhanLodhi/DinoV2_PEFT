"""Brain Tumor MRI 3-class classification dataset.

Loads from the Cheng brain tumor NDJSON manifest (Ultralytics format). Images
are 512x512 T1-weighted MRI slices; class labels are derived from YOLO annotation
class indices. Bounding box annotations are retained as metadata for future ROI
experiments but are not used during training.
"""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES: dict[int, str] = {
    0: "meningioma",
    1: "glioma",
    2: "pituitary",
}


class BrainTumorDataset(Dataset):
    """PyTorch Dataset for 3-class Brain Tumor MRI classification."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        image_root: str | Path,
        transforms: Any | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        """Each record is a dict with keys: file, split, label (int), bbox_metadata.

        Args:
            records: Sequence of parsed NDJSON image records.
            image_root: Root directory containing MRI slice image files.
            transforms: Optional torchvision or callable transform pipeline.
            class_names: Mapping from integer class index to pathology string.
        """
        self.records: list[dict[str, Any]] = list(records)
        self.image_root: Path = Path(image_root)
        self.transforms: Any | None = transforms
        self.class_names: dict[int, str] = class_names if class_names is not None else CLASS_NAMES

    def __len__(self) -> int:
        """Returns the total number of records in the dataset split."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Loads and transforms an MRI image and returns label and box metadata.

        Args:
            idx: Integer index of the record.

        Returns:
            Dictionary containing 'pixel_values', 'label', and 'bbox_metadata'.
        """
        record: dict[str, Any] = self.records[idx]
        image_path: Path = self.image_root / record["file"]

        if image_path.is_file():
            image: Image.Image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (512, 512), color=0)

        pixel_values: torch.Tensor = (
            self.transforms(image)
            if self.transforms is not None
            else image
        )
        label: torch.Tensor = torch.tensor(record["label"], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "label": label,
            "bbox_metadata": record.get("bbox_metadata", []),
        }


def brain_tumor_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a list of BrainTumorDataset samples into a batch dictionary.

    Args:
        batch: List of sample dictionaries from BrainTumorDataset.

    Returns:
        Batched dictionary with stacked tensors and a list of box metadata.
    """
    pixel_values: torch.Tensor = torch.stack([item["pixel_values"] for item in batch])
    labels: torch.Tensor = torch.stack([item["label"] for item in batch])
    bbox_metadata: list[Any] = [item.get("bbox_metadata", []) for item in batch]

    return {
        "pixel_values": pixel_values,
        "label": labels,
        "bbox_metadata": bbox_metadata,
    }


def load_ndjson_records(
    ndjson_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parses the Ultralytics NDJSON manifest into split-specific record lists.

    Skips the first line (dataset metadata). Each image record has type='image'.
    Extracts file, split ('train'/'val'/'test'), label (first annotation box class index),
    and full bbox list. Returns (train_records, val_records, test_records).

    Args:
        ndjson_path: Path to the NDJSON manifest file.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str: str = line.strip()
            if not line_str:
                continue
            record: dict[str, Any] = json.loads(line_str)
            if record.get("type") != "image":
                continue

            boxes: list[list[float]] = record.get("annotations", {}).get("boxes", [])
            if not boxes:
                continue

            label_idx: int = int(boxes[0][0])
            split: str = str(record.get("split", "train")).lower()

            item_dict: dict[str, Any] = {
                "file": record.get("file", ""),
                "split": split,
                "label": label_idx,
                "bbox_metadata": boxes,
            }

            if split == "train":
                train_records.append(item_dict)
            elif split in ("val", "validation"):
                val_records.append(item_dict)
            elif split == "test":
                test_records.append(item_dict)

    return train_records, val_records, test_records


def build_brain_tumor_dataloaders(
    config: dict[str, Any],
    ndjson_path: str | Path,
    image_root: str | Path,
    train_transforms: Any,
    val_transforms: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Parses NDJSON, creates datasets for all splits, and returns DataLoaders.

    Args:
        config: Configuration dictionary specifying batch size and workers.
        ndjson_path: Path to Ultralytics NDJSON manifest.
        image_root: Root directory of MRI image slices.
        train_transforms: Transform pipeline for the training split.
        val_transforms: Transform pipeline for validation and test splits.

    Returns:
        Tuple containing (train_loader, val_loader, test_loader).
    """
    batch_size: int = config.get("training", {}).get("batch_size", 32)
    num_workers: int = config.get("training", {}).get("num_workers", 2)
    pin_memory: bool = config.get("training", {}).get("pin_memory", True)

    train_recs, val_recs, test_recs = load_ndjson_records(ndjson_path)

    train_ds: BrainTumorDataset = BrainTumorDataset(
        records=train_recs,
        image_root=image_root,
        transforms=train_transforms,
        class_names=CLASS_NAMES,
    )
    val_ds: BrainTumorDataset = BrainTumorDataset(
        records=val_recs,
        image_root=image_root,
        transforms=val_transforms,
        class_names=CLASS_NAMES,
    )
    test_ds: BrainTumorDataset = BrainTumorDataset(
        records=test_recs,
        image_root=image_root,
        transforms=val_transforms,
        class_names=CLASS_NAMES,
    )

    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=brain_tumor_collate_fn,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=brain_tumor_collate_fn,
    )
    test_loader: DataLoader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=brain_tumor_collate_fn,
    )

    return train_loader, val_loader, test_loader
