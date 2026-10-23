"""Detic's CustomRCNN for fine-tuning, registered as MODEL.META_ARCHITECTURE "CustomFT_Detic"."""
from typing import Dict, List

import torch
from detectron2.config import configurable  # type: ignore
from detectron2.modeling import META_ARCH_REGISTRY  # type: ignore

import common.vision.detic  # noqa: F401  # puts Detic on sys.path
from detic.modeling.meta_arch.custom_rcnn import CustomRCNN  # type: ignore


@META_ARCH_REGISTRY.register()
class CustomFT_Detic(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        raise NotImplementedError
