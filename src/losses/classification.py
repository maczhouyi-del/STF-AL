from __future__ import annotations

import torch.nn.functional as F


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)
