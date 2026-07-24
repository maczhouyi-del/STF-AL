from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=None, dropout: float = 0.5):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        dims = [input_dim] + hidden_dims
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
