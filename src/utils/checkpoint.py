from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: Dict, config: Dict, metadata: Dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu") -> Dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
