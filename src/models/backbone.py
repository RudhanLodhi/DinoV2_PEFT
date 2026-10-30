"""DINOv2 backbone loading and inspection utilities.

Wraps MHRDYN7/dinov2_vitb14 (ViT-B/14, hidden_size=768) loaded via AutoModel.
Provides freeze/unfreeze helpers and layer-name inspection for LoRA target
configuration.
"""

from typing import Any
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

BACKBONE_MODEL_ID: str = "MHRDYN7/dinov2_vitb14"
HIDDEN_SIZE: int = 768
PROCESSOR_MODEL_ID: str = "facebook/dinov2-base"


def load_backbone(
    model_id: str = BACKBONE_MODEL_ID,
) -> tuple[Any, Any]:
    """Loads DINOv2 ViT-B/14 backbone and its image processor.

    Returns (model, processor). Uses device_map='auto'. The model outputs
    BaseModelOutputWithPooling; use last_hidden_state[:,0,:] (CLS token) as
    feature vector.

    Args:
        model_id: HuggingFace model hub repository identifier.

    Returns:
        Tuple containing the pretrained AutoModel backbone and AutoImageProcessor.
    """
    try:
        model = AutoModel.from_pretrained(model_id, device_map="auto")
    except Exception:
        model = AutoModel.from_pretrained(model_id)

    processor = AutoImageProcessor.from_pretrained(PROCESSOR_MODEL_ID)
    return model, processor


def freeze_backbone(model: nn.Module) -> None:
    """Freezes all backbone parameters.

    Used for linear probing (E0 baseline). Classification head parameters
    must be set to requires_grad=True separately after calling this function.

    Args:
        model: PyTorch model or backbone module to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all backbone parameters.

    Used before applying PEFT wrappers that need to identify trainable parameters.

    Args:
        model: PyTorch model or backbone module to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_attention_layer_names(model: nn.Module) -> list[str]:
    """Inspects the model's named modules and returns a sorted list of module names.

    Corresponds to self-attention components (query, key, value, output projections).
    Used to verify LoRA target_modules before applying PEFT. Filters for names
    containing 'attention' or 'query' or 'value' or 'key'.

    Args:
        model: PyTorch model module to inspect.

    Returns:
        Sorted list of matching module names.
    """
    keywords: tuple[str, ...] = ("attention", "query", "value", "key")
    matched_names: set[str] = {
        name
        for name, _ in model.named_modules()
        if any(kw in name.lower() for kw in keywords)
    }
    return sorted(matched_names)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params).

    Use to log parameter efficiency before and after PEFT.

    Args:
        model: PyTorch model to inspect.

    Returns:
        Tuple of (trainable_parameters_count, total_parameters_count).
    """
    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params: int = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
