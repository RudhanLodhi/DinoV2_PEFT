"""Model components: backbone loading, classification heads, and PEFT wrappers."""

from .backbone import (
    count_trainable_parameters,
    freeze_backbone,
    get_attention_layer_names,
    load_backbone,
    unfreeze_backbone,
)
from .classification_head import DINOv2Classifier, MultiClassHead, MultiLabelHead
from .peft_wrapper import (
    apply_bitfit,
    apply_ia3,
    apply_lora,
    merge_and_save,
    resolve_layers,
)

__all__ = [
    "load_backbone",
    "freeze_backbone",
    "unfreeze_backbone",
    "get_attention_layer_names",
    "count_trainable_parameters",
    "MultiLabelHead",
    "MultiClassHead",
    "DINOv2Classifier",
    "apply_lora",
    "apply_ia3",
    "apply_bitfit",
    "merge_and_save",
    "resolve_layers",
]
