from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    """Four-layer domain discriminator used by M1 and later variants.

    The training path returns logits for use with ``BCEWithLogitsLoss``. This
    is numerically equivalent to applying the listed sigmoid output activation
    followed by binary cross-entropy. ``predict_proba`` exposes the explicit
    sigmoid output when domain probabilities are needed.
    """

    DEFAULT_HIDDEN_DIMS = (384, 384, 64)

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("Domain discriminator input_dim must be positive")
        hidden_dims = list(self.DEFAULT_HIDDEN_DIMS if hidden_dims is None else hidden_dims)
        if any(dim < 1 for dim in hidden_dims):
            raise ValueError("Domain discriminator hidden dimensions must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        dims = [self.input_dim, *self.hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.output_activation = nn.Sigmoid()

    @property
    def num_linear_layers(self) -> int:
        return len(self.hidden_dims) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.forward(x))
