from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDilatedMultiheadAttention(nn.Module):
    """Local dilated multi-head attention for temporal sequences.

    The module accepts tensors shaped as [batch, time, channels]. For every
    time step it gathers a fixed-size local temporal neighborhood with the
    configured dilation rate, then computes attention only inside that
    neighborhood.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dilation: int,
        window_size: int,
        dropout: float,
        input_dim: int | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.d_model = int(d_model)
        self.input_dim = int(input_dim or d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.dilation = int(dilation)
        self.window_size = int(window_size)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.input_dim, self.d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_dropout = nn.Dropout(dropout)

    def _offsets(self, device: torch.device) -> torch.Tensor:
        center = self.window_size // 2
        return torch.arange(self.window_size, device=device) - center

    def _local_windows(self, values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # values: [B, T, H, Dh]
        dilated_offsets = offsets * self.dilation
        pad_left = int((-dilated_offsets.min()).item())
        pad_right = int(dilated_offsets.max().item())
        span = pad_left + pad_right + 1

        padded = F.pad(values, (0, 0, 0, 0, pad_left, pad_right))
        windows = padded.unfold(dimension=1, size=span, step=1)
        select_idx = (dilated_offsets + pad_left).long()
        windows = windows.index_select(dim=-1, index=select_idx)
        return windows.permute(0, 1, 4, 2, 3).contiguous()

    def _valid_mask(self, length: int, offsets: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(length, device=offsets.device).unsqueeze(1)
        neighbor_positions = positions + offsets.unsqueeze(0) * self.dilation
        return (neighbor_positions >= 0) & (neighbor_positions < length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, time, channels], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input channels, got {x.shape[-1]}"
            )
        batch_size, length, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, length, self.num_heads, self.head_dim)
        k = k.view(batch_size, length, self.num_heads, self.head_dim)
        v = v.view(batch_size, length, self.num_heads, self.head_dim)

        offsets = self._offsets(x.device)
        query_windows = self._local_windows(q, offsets)
        key_windows = self._local_windows(k, offsets)
        value_windows = self._local_windows(v, offsets)
        center_index = int((offsets == 0).nonzero(as_tuple=False).item())
        selected_queries = query_windows[:, :, center_index]

        scores = torch.einsum("bthd,btkhd->bthk", selected_queries, key_windows) * self.scale
        valid = self._valid_mask(length, offsets)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(2), -math.inf)

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.einsum("bthk,btkhd->bthd", weights, value_windows)
        out = out.reshape(batch_size, length, self.d_model)
        return self.out_dropout(self.out_proj(out))
