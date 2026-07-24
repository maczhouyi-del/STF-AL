from __future__ import annotations

import torch
import torch.nn as nn

from .dilated_attention import LocalDilatedMultiheadAttention


class SlidingWindowDilatedTemporalAttention(nn.Module):
    """Sliding-window dilated temporal attention.

    The module runs local temporal attention across one or more dilation rates,
    then projects the concatenated features back to the model dimension.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dilation: int = 4,
        window_size: int = 16,
        dropout: float = 0.5,
        dilation_rates=None,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim or d_model)
        self.dilation = int(dilation)
        self.window_size = int(window_size)
        self.rates = self._parse_rates(dilation_rates, self.dilation)
        self.attentions = nn.ModuleList(
            [
                LocalDilatedMultiheadAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    dilation=rate,
                    window_size=self.window_size,
                    dropout=dropout,
                    input_dim=self.input_dim,
                )
                for rate in self.rates
            ]
        )
        self.rate_projection = nn.Linear(d_model * len(self.rates), d_model)
        self.window_projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _parse_rates(dilation_rates, dilation: int) -> list[int]:
        raw_rates = dilation_rates if dilation_rates is not None else [1, 2, dilation]
        rates = []
        for rate in raw_rates:
            rate = int(rate)
            if rate < 1:
                raise ValueError("All SWDTA dilation rates must be >= 1")
            if rate not in rates:
                rates.append(rate)
        if not rates:
            raise ValueError("SWDTA requires at least one dilation rate")
        return rates

    def forward(self, x):
        outputs = [attention(x) for attention in self.attentions]
        per_window_features = self.rate_projection(torch.cat(outputs, dim=-1))
        aggregated = self.window_projection(per_window_features)
        return self.dropout(self.norm(aggregated))
