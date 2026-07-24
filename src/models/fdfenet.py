from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyMultiHeadAttention(nn.Module):
    """Multi-head self-attention with Q/K/V projected from a fused input."""

    def __init__(self, input_dim: int, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(input_dim, self.d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        output = output.view(batch_size, length, self.d_model)
        return self.out_dropout(self.out_projection(output))


class FrequencyFeatureLayer(nn.Module):
    """One frequency-domain feature extraction layer from FDFENet."""

    VALID_MODES = {"full", "magnitude", "phase", "mlp"}

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        kernel_size: int = 3,
        spectral_mode: str = "full",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.spectral_mode = str(spectral_mode).lower()
        if self.spectral_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown spectral_mode={spectral_mode!r}; expected one of {sorted(self.VALID_MODES)}"
            )
        self.eps = float(eps)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("FDFENet Conv1D kernel_size must be a positive odd integer")

        spectral_dims = {
            "full": d_model * 3,
            "magnitude": d_model,
            "phase": d_model * 2,
            "mlp": d_model * 3,
        }
        spectral_dim = spectral_dims[self.spectral_mode]
        self.spectral_projection = nn.Linear(spectral_dim, d_model)
        self.spectral_norm = nn.LayerNorm(d_model)
        self.fused_norm = nn.LayerNorm(d_model * 2)
        self.attention = FrequencyMultiHeadAttention(d_model * 2, d_model, num_heads, dropout)

        self.time_projection = nn.Linear(d_model, d_model)
        self.frequency_projection = nn.Linear(d_model, d_model)
        self.attention_projection = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.spectral_mlp = None
        if self.spectral_mode == "mlp":
            bottleneck_dim = max(1, d_model // 8)
            self.spectral_mlp = nn.Sequential(
                nn.Linear(d_model, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, d_model * 3),
            )

    def _standardized_log_magnitude(self, spectrum: torch.Tensor) -> torch.Tensor:
        magnitude = torch.log1p(spectrum.abs())
        mean = magnitude.mean(dim=1, keepdim=True)
        std = magnitude.var(dim=1, keepdim=True, unbiased=False).sqrt()
        return (magnitude - mean) / (std + self.eps)

    def _spectral_encoding(self, e: torch.Tensor) -> torch.Tensor:
        if self.spectral_mlp is not None:
            return self.spectral_mlp(e)

        spectrum = torch.fft.rfft(e, dim=1)
        components = []
        if self.spectral_mode in {"full", "magnitude"}:
            components.append(self._standardized_log_magnitude(spectrum))
        if self.spectral_mode in {"full", "phase"}:
            phase = torch.angle(spectrum)
            components.extend([torch.cos(phase), torch.sin(phase)])
        return torch.cat(components, dim=-1)

    @staticmethod
    def _align_frequency_length(encoded: torch.Tensor, output_length: int) -> torch.Tensor:
        if encoded.shape[1] == output_length:
            return encoded
        return F.interpolate(
            encoded.transpose(1, 2),
            size=output_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        encoded = self._spectral_encoding(e)
        encoded = self._align_frequency_length(encoded, e.shape[1])
        y = self.spectral_norm(self.spectral_projection(encoded))

        u = self.fused_norm(torch.cat([e, y], dim=-1))
        c = self.attention(u)
        fused = (
            self.time_projection(e)
            + self.frequency_projection(y)
            + self.attention_projection(c)
        )
        convolved = self.conv(fused.transpose(1, 2)).transpose(1, 2)
        return self.ffn(convolved)


class FDFENet(nn.Module):
    """Frequency-domain feature extraction network."""

    def __init__(
        self,
        input_channels: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        kernel_size: int = 3,
        spectral_mode: str = "full",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("FDFENet requires at least one feature extraction layer")
        self.input_embedding = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.layers = nn.ModuleList(
            [
                FrequencyFeatureLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    spectral_mode=spectral_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model * num_layers, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.spectral_mode = str(spectral_mode).lower()

    @property
    def output_dim(self) -> int:
        return self.output_norm.normalized_shape[0]

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, time, channels], got {tuple(x.shape)}")
        e = self.input_embedding(x.transpose(1, 2)).transpose(1, 2)
        layer_outputs = []
        for layer in self.layers:
            e = layer(e)
            layer_outputs.append(e)
        sequence = self.output_norm(self.output_projection(torch.cat(layer_outputs, dim=-1)))
        pooled = sequence.mean(dim=1)
        if return_dict:
            return {"pooled": pooled, "sequence": sequence, "layers": layer_outputs}
        return pooled
