from __future__ import annotations

import torch


def frequency_domain_similarity_matrix(source_frequency, target_frequency, eps: float = 1e-8):
    if source_frequency is None or target_frequency is None:
        return None
    if source_frequency.ndim != 2 or target_frequency.ndim != 2:
        raise ValueError("FDSM expects source and target frequency features shaped [batch, features]")
    if source_frequency.shape[1] != target_frequency.shape[1]:
        raise ValueError("Source and target frequency feature dimensions must match")
    numerator = torch.matmul(source_frequency, target_frequency.transpose(0, 1))
    source_norm = torch.linalg.vector_norm(source_frequency, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target_frequency, dim=-1).unsqueeze(0)
    return numerator / (source_norm * target_norm + eps)
