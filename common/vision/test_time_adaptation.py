"""BatchNorm test-time adaptation of a detectron2 model's FrozenBatchNorm2d layers, without labels: NORM (Schneider et
al. 2020), each batch normalized by its own statistics mixed with the source ones."""
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from detectron2.layers import FrozenBatchNorm2d  # type: ignore
from torch import nn


class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d normalizing each batch of B images by its running (source) statistics and the batch's own, the batch's
    weighted B / (prior_strength + B)."""

    def __init__(self, num_features: int, prior_strength: float, eps: float):
        super().__init__(num_features, eps=eps)
        self.prior_strength = prior_strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_weight = x.shape[0] / (self.prior_strength + x.shape[0])
        mean = (1 - batch_weight) * self.running_mean + batch_weight * x.mean(dim=(0, 2, 3))
        var = (1 - batch_weight) * self.running_var + batch_weight * x.var(dim=(0, 2, 3))
        return F.batch_norm(x, mean, var, self.weight, self.bias, training=False, eps=self.eps)


def thawed(frozen: FrozenBatchNorm2d, norm: nn.BatchNorm2d) -> nn.BatchNorm2d:
    """norm with frozen's affine and running statistics, on frozen's device, in eval mode and without gradients."""
    with torch.no_grad():
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(norm, name).copy_(getattr(frozen, name))
    return norm.to(frozen.weight.device).requires_grad_(False).eval()


def replace_frozen_batch_norms(model: nn.Module, build: Callable[[FrozenBatchNorm2d], nn.Module], kept_suffixes: Sequence[str]) -> None:
    """Replaces, in place, every FrozenBatchNorm2d of model by build(it), but those whose name ends with one of kept_suffixes."""
    frozen_norms = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, FrozenBatchNorm2d) and not name.endswith(tuple(kept_suffixes))
    ]
    for name, frozen in frozen_norms:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, build(frozen))

