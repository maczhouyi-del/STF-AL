from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_weights(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    return weights.numel() * weights / (weights.sum() + eps)


def domain_adversarial_loss(source_logits, target_logits, similarity_matrix=None):
    source_labels = torch.ones_like(source_logits)
    target_labels = torch.zeros_like(target_logits)
    if similarity_matrix is None:
        source_loss = F.binary_cross_entropy_with_logits(source_logits, source_labels)
        target_loss = F.binary_cross_entropy_with_logits(target_logits, target_labels)
        return source_loss + target_loss

    if similarity_matrix.shape != (source_logits.shape[0], target_logits.shape[0]):
        raise ValueError(
            "FDSM shape must be [source_batch, target_batch], got "
            f"{tuple(similarity_matrix.shape)}"
        )
    nonnegative_similarity = 0.5 * (similarity_matrix + 1.0)
    source_weights = _normalize_weights(nonnegative_similarity.mean(dim=1))
    target_weights = _normalize_weights(nonnegative_similarity.mean(dim=0))
    source_per_sample = F.binary_cross_entropy_with_logits(
        source_logits, source_labels, reduction="none"
    )
    target_per_sample = F.binary_cross_entropy_with_logits(
        target_logits, target_labels, reduction="none"
    )
    source_loss = (source_weights * source_per_sample).mean()
    target_loss = (target_weights * target_per_sample).mean()
    return source_loss + target_loss
