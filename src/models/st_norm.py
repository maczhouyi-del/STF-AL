from __future__ import annotations

import torch
import torch.nn as nn


class TemporalNorm(nn.Module):
    """Learnable temporal normalization for [batch, time, channels] tensors."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = var.sqrt()
        return (x - mean) / (std + self.eps) * self.gamma + self.beta


class SpatialNorm(nn.Module):
    """Learnable feature-wise normalization for [batch, time, channels] tensors."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = var.sqrt()
        return (x - mean) / (std + self.eps) * self.gamma + self.beta
