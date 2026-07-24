from __future__ import annotations

import torch


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_value: float):
        ctx.lambda_value = float(lambda_value)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_value: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, lambda_value)
