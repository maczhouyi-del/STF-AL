from __future__ import annotations

import math

import torch
import torch.nn as nn

from .st_norm import SpatialNorm, TemporalNorm
from .swdta import SlidingWindowDilatedTemporalAttention


class TemporalFeatureLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_swdta: bool,
        dilation: int,
        window_size: int,
        dilation_rates=None,
    ):
        super().__init__()
        self.temporal_norm = TemporalNorm(d_model)
        if use_swdta:
            self.attention = SlidingWindowDilatedTemporalAttention(
                d_model=d_model,
                num_heads=num_heads,
                dilation=dilation,
                window_size=window_size,
                dropout=dropout,
                dilation_rates=dilation_rates,
                input_dim=d_model * 2,
            )
        else:
            self.expand = nn.Linear(d_model * 2, d_model)
            self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.use_swdta = use_swdta
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        high = self.temporal_norm(h)
        x = torch.cat([h, high], dim=-1)
        if self.use_swdta:
            attn = self.attention(x)
        else:
            x = self.expand(x)
            attn, _ = self.attention(x, x, x, need_weights=False)
        h = self.norm1(h + attn)
        h = self.norm2(h + self.ffn(h))
        return h


class FeatureSpatialAttention(nn.Module):
    """Feature-wise spatial attention over latent sensor channels.

    After the Transformer encoder, spatial dependencies are represented as
    attention among latent feature channels using the temporal axis as context.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        input_dim: int | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.input_dim = int(input_dim or d_model)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.qkv = nn.Conv1d(self.input_dim, d_model * 3, kernel_size=1)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x.transpose(1, 2)).chunk(3, dim=1)
        batch_size, _, length = q.shape
        q = q.view(batch_size, self.num_heads, self.head_dim, length)
        k = k.view(batch_size, self.num_heads, self.head_dim, length)
        v = v.view(batch_size, self.num_heads, self.head_dim, length)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(1, length))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(weights), v)
        out = out.reshape(batch_size, self.d_model, length).transpose(1, 2)
        return self.out(out)


class SpatialFeatureLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.spatial_norm = SpatialNorm(d_model)
        self.attention = FeatureSpatialAttention(
            d_model,
            num_heads,
            dropout,
            input_dim=d_model * 2,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        local = self.spatial_norm(h)
        x = torch.cat([h, local], dim=-1)
        attn = self.attention(x)
        h = self.norm1(h + attn)
        h = self.norm2(h + self.ffn(h))
        return h


class SpatioTemporalFusion(nn.Module):
    """Fuse temporal and spatial sequence features."""

    def __init__(self, temporal_dim: int, spatial_dim: int, fusion_type: str = "concat"):
        super().__init__()
        self.fusion_type = fusion_type
        self.temporal_dim = int(temporal_dim)
        self.spatial_dim = int(spatial_dim)
        if fusion_type == "concat":
            self.output_dim = self.temporal_dim + self.spatial_dim
            self.norm = nn.Identity()
        elif fusion_type == "add":
            if self.temporal_dim != self.spatial_dim:
                raise ValueError("temporal_dim and spatial_dim must match for add fusion")
            self.output_dim = self.temporal_dim
            self.norm = nn.LayerNorm(self.output_dim)
        elif fusion_type == "weighted":
            if self.temporal_dim != self.spatial_dim:
                raise ValueError("temporal_dim and spatial_dim must match for weighted fusion")
            self.output_dim = self.temporal_dim
            self.temporal_weight = nn.Parameter(torch.tensor(0.5))
            self.spatial_weight = nn.Parameter(torch.tensor(0.5))
            self.norm = nn.LayerNorm(self.output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, temporal_features: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused = torch.cat([temporal_features, spatial_features], dim=-1)
        elif self.fusion_type == "add":
            fused = temporal_features + spatial_features
        else:
            weights = torch.softmax(torch.stack([self.temporal_weight, self.spatial_weight]), dim=0)
            fused = weights[0] * temporal_features + weights[1] * spatial_features
        return self.norm(fused)


class STFENet(nn.Module):
    """Spatiotemporal feature extraction network from the STF-AL paper."""

    def __init__(
        self,
        input_channels: int,
        d_model: int,
        num_heads: int,
        transformer_layers: int,
        feature_layers: int,
        ffn_dim: int,
        dropout: float,
        use_swdta: bool,
        dilation: int,
        window_size: int,
        dilation_rates=None,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.temporal_layers = nn.ModuleList(
            [
                TemporalFeatureLayer(
                    d_model,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_swdta,
                    dilation,
                    window_size,
                    dilation_rates=dilation_rates,
                )
                for _ in range(feature_layers)
            ]
        )
        self.spatial_layers = nn.ModuleList(
            [
                SpatialFeatureLayer(d_model, num_heads, ffn_dim, dropout)
                for _ in range(feature_layers)
            ]
        )
        self.fusion = SpatioTemporalFusion(d_model, d_model, fusion_type="concat")

    @property
    def output_dim(self) -> int:
        return self.fusion.output_dim

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        h = self.input_projection(x)
        h = self.transformer(h)
        ht = h
        hs = h
        for layer in self.temporal_layers:
            ht = layer(ht)
        for layer in self.spatial_layers:
            hs = layer(hs)
        fused = self.fusion(ht, hs)
        pooled = fused.mean(dim=1)
        if return_dict:
            return {
                "pooled": pooled,
                "sequence": fused,
                "transformer": h,
                "temporal": ht,
                "spatial": hs,
            }
        return pooled
