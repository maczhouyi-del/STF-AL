from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def impute_missing_values(x: np.ndarray, strategy: str = "interpolate") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isnan(x).any():
        return x
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [time, channels], got {x.shape}")
    filled = x.copy()
    for col in range(filled.shape[1]):
        values = filled[:, col]
        valid = ~np.isnan(values)
        if valid.all():
            continue
        if not valid.any():
            filled[:, col] = 0.0
        elif strategy == "interpolate":
            indices = np.arange(len(values))
            filled[:, col] = np.interp(indices, indices[valid], values[valid]).astype(np.float32)
        elif strategy == "zero":
            filled[~valid, col] = 0.0
        else:
            raise ValueError(f"Unsupported missing value strategy: {strategy}")
    return filled


@dataclass
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, x: np.ndarray) -> "ChannelNormalizer":
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [samples, time, channels], got {x.shape}")
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        std = x.reshape(-1, x.shape[-1]).std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    @classmethod
    def from_dict(cls, data: Dict) -> "ChannelNormalizer":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            eps=float(data.get("eps", 1e-6)),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (self.std + self.eps)).astype(np.float32)

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": float(self.eps),
        }
