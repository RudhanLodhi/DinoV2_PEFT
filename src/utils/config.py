"""YAML configuration loader with dot-notation override support.

Loads base YAML files and merges override dicts for per-experiment customization
from notebooks.
"""

from collections.abc import Mapping
import copy
from pathlib import Path
from typing import Any
import yaml


def _deep_merge_two(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep-merges update dictionary into base dictionary.

    Args:
        base: Target dictionary.
        update: Source dictionary containing values to merge.

    Returns:
        New dictionary with merged values.
    """
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, val in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, Mapping):
            merged[key] = _deep_merge_two(merged[key], dict(val))
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merges multiple config dicts left to right.

    Later dicts take precedence for scalar values; nested dicts are recursively
    merged. Useful for combining dataset and PEFT configs in notebooks.

    Args:
        *configs: Variable sequence of configuration dictionaries.

    Returns:
        Merged configuration dictionary.
    """
    result: dict[str, Any] = {}
    for cfg in configs:
        if cfg:
            result = _deep_merge_two(result, cfg)
    return result


def load_config(
    yaml_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Loads a YAML config file and optionally merges a flat overrides dict.

    Override keys use dot notation (e.g., 'lora.r': 16 overrides config['lora']['r']).
    Returns the merged config dict.

    Args:
        yaml_path: File system path to the YAML configuration file.
        overrides: Optional flat dictionary with dot-delimited key paths.

    Returns:
        Loaded and updated configuration dictionary.
    """
    path: Path = Path(yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides is not None:
        for dot_key, value in overrides.items():
            keys: list[str] = dot_key.split(".")
            curr: dict[str, Any] = config
            for sub_key in keys[:-1]:
                if sub_key not in curr or not isinstance(curr[sub_key], dict):
                    curr[sub_key] = {}
                curr = curr[sub_key]
            curr[keys[-1]] = value

    return config
