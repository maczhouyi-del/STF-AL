from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_entries = cfg.get("base")
    if not base_entries:
        return cfg
    if isinstance(base_entries, (str, Path)):
        base_entries = [base_entries]
    merged: Dict[str, Any] = {}
    for base in base_entries:
        base_path = (path.parent / base).resolve()
        merged = deep_merge(merged, load_config(base_path))
    return deep_merge(merged, cfg)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
