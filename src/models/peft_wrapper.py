"""PEFT method wrappers for DINOv2 backbone.

Supports LoRA (fully implemented) and IA3 (scaffolded). Uses the HuggingFace
PEFT library.
"""

from pathlib import Path
from typing import Any

from peft import IA3Config, LoraConfig, TaskType, get_peft_model
import torch.nn as nn


def apply_lora(model: nn.Module, lora_cfg: dict[str, Any]) -> Any:
    """Applies LoRA to the DINOv2 backbone using the provided configuration dict.

    Keys used: r, lora_alpha, lora_dropout, target_modules, bias, modules_to_save.
    Returns the PEFT-wrapped model. Use model.print_trainable_parameters() after
    calling to verify parameter efficiency.

    Args:
        model: Pretrained DINOv2 backbone module.
        lora_cfg: Configuration dictionary specifying LoRA hyperparameters.

    Returns:
        PEFT-wrapped model with trainable low-rank adaptation matrices.
    """
    r: int = lora_cfg.get("r", 48)
    lora_alpha: int = lora_cfg.get("lora_alpha", lora_cfg.get("alpha", 16))
    lora_dropout: float = lora_cfg.get("lora_dropout", 0.1)
    target_modules: Any = lora_cfg.get("target_modules", ["query", "value"])
    bias: str = lora_cfg.get("bias", "none")
    modules_to_save: list[str] | None = lora_cfg.get("modules_to_save", None)
    layers_to_transform: list[int] | None = lora_cfg.get(
        "layers_to_transform", lora_cfg.get("adapted_layers", None)
    )

    config_kwargs: dict[str, Any] = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
        "modules_to_save": modules_to_save,
        "task_type": TaskType.FEATURE_EXTRACTION,
    }

    if layers_to_transform is not None:
        config_kwargs["layers_to_transform"] = layers_to_transform

    peft_config: LoraConfig = LoraConfig(**config_kwargs)
    return get_peft_model(model, peft_config)


def apply_ia3(model: nn.Module, ia3_cfg: dict[str, Any]) -> Any:
    """Scaffolded IA3 wrapper.

    Applies IA3 learned scaling vectors to keys, values, and FFN intermediate
    activations. This method is prepared for Phase III experiments. Returns the
    PEFT-wrapped model.

    Args:
        model: Pretrained DINOv2 backbone module.
        ia3_cfg: Dictionary specifying IA3 hyperparameters.

    Returns:
        PEFT-wrapped model with IA3 scaling vectors.
    """
    target_modules: Any = ia3_cfg.get("target_modules", ["key", "value", "dense"])
    feedforward_modules: Any = ia3_cfg.get("feedforward_modules", ["dense"])
    modules_to_save: list[str] | None = ia3_cfg.get("modules_to_save", None)

    ia3_config: IA3Config = IA3Config(
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
        modules_to_save=modules_to_save,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, ia3_config)


def apply_bitfit(model: nn.Module) -> nn.Module:
    """BitFit baseline: freezes all parameters except bias terms.

    Modifies the model in place. Returns the model for chaining.

    Args:
        model: Pretrained DINOv2 backbone or combined model module.

    Returns:
        The modified model with only bias parameters requiring gradients.
    """
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def merge_and_save(peft_model: Any, output_dir: str | Path) -> None:
    """Merges LoRA adapter weights into the frozen backbone parameters and saves to output_dir.

    The resulting model has no adapter overhead and can be loaded with
    AutoModel.from_pretrained. Call only after training is complete.

    Args:
        peft_model: Fine-tuned PEFT model with trained adapters.
        output_dir: Directory path to save the merged model weights.
    """
    out_path: Path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_path))
