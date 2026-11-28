"""PEFT method wrappers for DINOv2 backbone.

Supports LoRA (fully implemented) and IA3 (scaffolded). Uses the HuggingFace
PEFT library.

Layer-selective LoRA (E3c/E3d/E4) requires restricting which transformer blocks
receive adapters. HuggingFace `LoraConfig` exposes `layers_to_transform`
(a list of integer block indices). This module also accepts a regex
`layers_pattern` (e.g. ``r"layer\\.(9|10|11)\\."``) and converts it to the
corresponding integer indices, so notebooks can express layer selection directly.
"""

import re
from pathlib import Path
from typing import Any

from peft import IA3Config, LoraConfig, TaskType, get_peft_model
import torch.nn as nn


def _infer_num_layers(model: nn.Module) -> int | None:
    """Infers the number of transformer blocks from common attribute names.

    Args:
        model: Backbone module to inspect.

    Returns:
        Number of blocks if discoverable, otherwise None.
    """
    for attr in ("encoder", "model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            block_list = getattr(sub, "layer", None)
            if block_list is not None and hasattr(block_list, "__len__"):
                return len(block_list)
    return None


def resolve_layers(
    model: nn.Module,
    layers_pattern: str | None = None,
    strategy: str | None = None,
    num_layers: int | None = None,
) -> list[int] | None:
    """Resolves a layer-selection specification to a list of block indices.

    Exactly one of ``layers_pattern`` or ``strategy`` is expected. ``strategy``
    is one of 'upper_quarter', 'lower_quarter', or 'distributed'.

    Args:
        model: Backbone module used to infer the total layer count.
        layers_pattern: Optional regex matched against named modules; block
            indices are extracted from ``layer.<n>.`` matches.
        strategy: Optional named selection strategy.
        num_layers: Optional explicit total block count (overrides inference).

    Returns:
        Sorted list of integer block indices, or None when no selection is made.
    """
    total: int | None = num_layers if num_layers is not None else _infer_num_layers(model)

    if layers_pattern:
        # Case 1: pattern already encodes explicit indices, e.g. "layer\.(9|10|11)\."
        group = re.search(r"\(([\d|]+)\)", layers_pattern)
        if group:
            return sorted({int(x) for x in group.group(1).split("|") if x.strip().isdigit()})

        # Case 2: bare "layer.<n>." occurrences in the pattern string.
        matches = re.findall(r"layer\.(\d+)\.", layers_pattern)
        if matches:
            return sorted({int(m) for m in matches})

        # Case 3: treat the pattern as a regex to match against real module names.
        found: set[int] = set()
        named_modules = getattr(model, "named_modules", None)
        if named_modules is not None:
            for name, _ in named_modules():
                if re.search(layers_pattern, name):
                    found.update(int(m) for m in re.findall(r"layer\.(\d+)\.", name))
        return sorted(found) if found else None

    if strategy and total is not None:
        if strategy == "upper_quarter":
            start = max(0, total - max(1, total // 4))
            return list(range(start, total))
        if strategy == "lower_quarter":
            return list(range(0, max(1, total // 4)))
        if strategy == "distributed":
            return list(range(0, total, 2))
        raise ValueError(f"Unknown layer strategy: {strategy!r}")

    return None


_ATTN_NAME_FALLBACKS: dict[str, str] = {
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "output": "o_proj",
    "q_proj": "query",
    "k_proj": "key",
    "v_proj": "value",
    "o_proj": "dense",
}


def _resolve_target_modules(model: nn.Module, target_modules: Any) -> Any:
    """Maps attention target names between transformers 4.x (query/value) and 5.x (q_proj/v_proj)."""
    if not isinstance(target_modules, (list, tuple)):
        return target_modules
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return target_modules
    linear_leaf_names: set[str] = {
        name.rsplit(".", 1)[-1]
        for name, mod in named_modules()
        if name and isinstance(mod, nn.Linear)
    }
    if not linear_leaf_names:
        return target_modules
    resolved: list[str] = []
    for target in target_modules:
        if target in linear_leaf_names:
            resolved.append(target)
        elif _ATTN_NAME_FALLBACKS.get(target) in linear_leaf_names:
            resolved.append(_ATTN_NAME_FALLBACKS[target])
        elif target == "output" and "dense" in linear_leaf_names:
            resolved.append("dense")
        else:
            resolved.append(target)
    return list(dict.fromkeys(resolved))


def apply_lora(model: nn.Module, lora_cfg: dict[str, Any]) -> Any:
    """Applies LoRA to the DINOv2 backbone using the provided configuration dict.

    Keys used: r, lora_alpha, lora_dropout, target_modules, bias,
    modules_to_save, plus optional layer selection via ``layers_to_transform``,
    ``layers_pattern``, or ``layer_selection`` {enabled, strategy, layers}.

    Args:
        model: Pretrained DINOv2 backbone module.
        lora_cfg: Configuration dictionary specifying LoRA hyperparameters.

    Returns:
        PEFT-wrapped model with trainable low-rank adaptation matrices.
    """
    r: int = lora_cfg.get("r", 48)
    lora_alpha: int = lora_cfg.get("lora_alpha", lora_cfg.get("alpha", 16))
    lora_dropout: float = lora_cfg.get("lora_dropout", 0.1)
    target_modules: Any = _resolve_target_modules(
        model, lora_cfg.get("target_modules", ["query", "value"])
    )
    bias: str = lora_cfg.get("bias", "none")
    modules_to_save: list[str] | None = lora_cfg.get("modules_to_save", None)

    layers_to_transform: list[int] | None = lora_cfg.get(
        "layers_to_transform", lora_cfg.get("adapted_layers", None)
    )

    if layers_to_transform is None:
        layers_pattern: str | None = lora_cfg.get("layers_pattern")
        layer_selection: dict[str, Any] = lora_cfg.get("layer_selection", {}) or {}
        if layers_pattern is None and layer_selection.get("enabled"):
            strategy: str | None = layer_selection.get("strategy")
            explicit: list[int] | None = layer_selection.get("layers")
            if explicit:
                layers_to_transform = list(explicit)
            elif strategy:
                layers_to_transform = resolve_layers(
                    model,
                    strategy=strategy,
                    num_layers=lora_cfg.get("num_layers"),
                )
        elif layers_pattern is not None:
            layers_to_transform = resolve_layers(
                model,
                layers_pattern=layers_pattern,
                num_layers=lora_cfg.get("num_layers"),
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
        config_kwargs["layers_to_transform"] = list(layers_to_transform)

    peft_config: LoraConfig = LoraConfig(**config_kwargs)
    return get_peft_model(model, peft_config)


def apply_ia3(model: nn.Module, ia3_cfg: dict[str, Any]) -> Any:
    """Scaffolded IA3 wrapper.

    Applies IA3 learned scaling vectors to keys, values, and FFN intermediate
    activations. Prepared for Phase III experiments.

    Args:
        model: Pretrained DINOv2 backbone module.
        ia3_cfg: Dictionary specifying IA3 hyperparameters.

    Returns:
        PEFT-wrapped model with IA3 scaling vectors.
    """
    target_modules: Any = _resolve_target_modules(
        model, ia3_cfg.get("target_modules", ["key", "value", "dense"])
    )
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
        param.requires_grad = "bias" in name
    return model


def merge_and_save(peft_model: Any, output_dir: str | Path) -> None:
    """Merges adapter weights into the frozen backbone and saves to output_dir.

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
