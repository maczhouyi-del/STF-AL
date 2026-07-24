from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .classifier import Classifier
from .discriminator import DomainDiscriminator
from .fdfenet import FDFENet
from .grad_reverse import grad_reverse
from .stfenet import STFENet


VARIANT_FEATURES = {
    "m0": {"adversarial": False, "swdta": False, "frequency": False, "fdsm": False},
    "m1": {"adversarial": True, "swdta": False, "frequency": False, "fdsm": False},
    "m2": {"adversarial": True, "swdta": True, "frequency": False, "fdsm": False},
    "m3": {
        "adversarial": True,
        "swdta": True,
        "frequency": True,
        "fdsm": False,
        "spectral_mode": "full",
    },
    "m3_mag": {
        "adversarial": True,
        "swdta": True,
        "frequency": True,
        "fdsm": False,
        "spectral_mode": "magnitude",
    },
    "m3_phase": {
        "adversarial": True,
        "swdta": True,
        "frequency": True,
        "fdsm": False,
        "spectral_mode": "phase",
    },
    "m3_mlp": {
        "adversarial": True,
        "swdta": True,
        "frequency": True,
        "fdsm": False,
        "spectral_mode": "mlp",
    },
    "stfal": {
        "adversarial": True,
        "swdta": True,
        "frequency": True,
        "fdsm": True,
        "spectral_mode": "full",
    },
}

MODEL_VARIANTS = ("m0", "m1", "m2", "m3", "stfal")
SPECTRAL_ABLATION_VARIANTS = ("m3_mag", "m3_phase", "m3_mlp")


class STFALModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config["model"]
        self.variant = str(model_cfg.get("variant", "stfal")).lower().replace("-", "_")
        if self.variant not in VARIANT_FEATURES:
            raise ValueError(f"Unknown model variant: {self.variant}")
        flags = VARIANT_FEATURES[self.variant]
        self.use_adversarial = flags["adversarial"]
        self.use_frequency = flags["frequency"]
        self.use_fdsm = flags["fdsm"]

        input_channels = int(model_cfg["input_channels"])
        d_model = int(model_cfg["d_model"])
        num_heads = int(model_cfg["num_heads"])
        transformer_layers = int(model_cfg["transformer_layers"])
        feature_layers = int(model_cfg["feature_layers"])
        frequency_layers = int(model_cfg["frequency_layers"])
        ffn_dim = int(model_cfg["ffn_dim"])
        dropout = float(model_cfg["dropout"])
        swdta_cfg = model_cfg.get("swdta", {})
        dilation_rates = swdta_cfg.get("dilation_rates")

        self.stfenet = STFENet(
            input_channels=input_channels,
            d_model=d_model,
            num_heads=num_heads,
            transformer_layers=transformer_layers,
            feature_layers=feature_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_swdta=flags["swdta"],
            dilation=int(swdta_cfg.get("dilation", 4)),
            window_size=int(swdta_cfg.get("window_size", 16)),
            dilation_rates=dilation_rates,
        )
        feature_dim = self.stfenet.output_dim
        if self.use_frequency:
            frequency_cfg = model_cfg.get("frequency", {})
            self.fdfenet = FDFENet(
                input_channels=input_channels,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=frequency_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                kernel_size=int(frequency_cfg.get("kernel_size", 3)),
                spectral_mode=flags.get(
                    "spectral_mode", frequency_cfg.get("spectral_mode", "full")
                ),
            )
            concatenated_dim = feature_dim + self.fdfenet.output_dim
            feature_dim = int(model_cfg.get("fusion_dim", d_model * 2))
            self.feature_fusion = nn.Linear(concatenated_dim, feature_dim)
        else:
            self.fdfenet = None
            self.feature_fusion = None

        self.classifier = Classifier(
            input_dim=feature_dim,
            num_classes=int(model_cfg["num_classes"]),
            hidden_dims=model_cfg.get("classifier", {}).get("hidden_dims", [d_model]),
            dropout=dropout,
        )
        if self.use_adversarial:
            self.discriminator = DomainDiscriminator(
                input_dim=feature_dim,
                hidden_dims=model_cfg.get("discriminator", {}).get(
                    "hidden_dims", DomainDiscriminator.DEFAULT_HIDDEN_DIMS
                ),
                dropout=dropout,
            )
        else:
            self.discriminator = None
        self.feature_dim = feature_dim
        self.input_channels = input_channels
        self.num_classes = int(model_cfg["num_classes"])
        self.d_model = d_model

    def extract_features(self, x: torch.Tensor):
        st_outputs = self.stfenet(x, return_dict=True)
        st_features = st_outputs["pooled"]
        freq_features = None
        if self.fdfenet is not None:
            freq_features = self.fdfenet(x)
            features = self.feature_fusion(torch.cat([st_features, freq_features], dim=-1))
        else:
            features = st_features
        aux = {
            "spatiotemporal_sequence": st_outputs["sequence"],
            "transformer_sequence": st_outputs["transformer"],
            "temporal_sequence": st_outputs["temporal"],
            "spatial_sequence": st_outputs["spatial"],
            "spatiotemporal_features": st_features,
            "frequency_features": freq_features,
            "final_features": features,
        }
        return features, freq_features, aux

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        features, _, _ = self.extract_features(x)
        return self.classifier(features)

    def get_model_info(self) -> Dict:
        return {
            "variant": self.variant,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "d_model": self.d_model,
            "feature_dim": self.feature_dim,
            "use_adversarial": self.use_adversarial,
            "use_frequency": self.use_frequency,
            "use_fdsm": self.use_fdsm,
            "discriminator_hidden_dims": list(self.discriminator.hidden_dims)
            if self.discriminator is not None
            else None,
            "discriminator_linear_layers": self.discriminator.num_linear_layers
            if self.discriminator is not None
            else 0,
            "swdta_rates": list(self.stfenet.temporal_layers[0].attention.rates)
            if self.variant in {"m2", "m3", "m3_mag", "m3_phase", "m3_mlp", "stfal"}
            else None,
            "frequency_mode": self.fdfenet.spectral_mode if self.fdfenet is not None else None,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def forward(self, source_x: torch.Tensor, target_x: torch.Tensor | None = None, grl_lambda: float = 1.0):
        source_features, source_freq, source_aux = self.extract_features(source_x)
        out = {
            "logits": self.classifier(source_features),
            "features": source_features,
            "domain_logits": None,
            "aux": {
                "variant": self.variant,
                "frequency_features": source_freq,
                "source": source_aux,
            },
            "source_features": source_features,
            "source_frequency_features": source_freq,
        }
        out["source_logits"] = out["logits"]
        if target_x is not None:
            target_features, target_freq, target_aux = self.extract_features(target_x)
            out["target_features"] = target_features
            out["target_frequency_features"] = target_freq
            out["aux"]["target_frequency_features"] = target_freq
            out["aux"]["target"] = target_aux
            if self.discriminator is not None:
                out["source_domain_logits"] = self.discriminator(grad_reverse(source_features, grl_lambda))
                out["target_domain_logits"] = self.discriminator(grad_reverse(target_features, grl_lambda))
                out["domain_logits"] = {
                    "source": out["source_domain_logits"],
                    "target": out["target_domain_logits"],
                }
        return out


def build_model(config: Dict) -> STFALModel:
    return STFALModel(config)


def create_model(variant: str, config: Dict) -> STFALModel:
    model_config = dict(config)
    model_config["model"] = dict(config["model"])
    model_config["model"]["variant"] = variant.lower().replace("-", "_")
    return STFALModel(model_config)
