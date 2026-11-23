"""Detic's box-supervised GeneralizedRCNN (CustomRCNN's box path) for fine-tuning on the single-box object views
habitat_embodied_sl/build_dataset.py writes, registered as MODEL.META_ARCHITECTURE "CustomFT_Detic" (example config:
config/r50_detic.yaml). Proposals, in training and inference, come from a frozen teacher copy of the initial
backbone and proposal generator, so localization stays the loaded checkpoint's; training returns the student ROI heads'
losses on them, each weighted by CustomFT_Detic.LOSS_WEIGHTS. Each view labels one object of its frame, so the ROI heads (PositiveOnlyDeticCascadeROIHeads)
learn from proposals matched to a ground-truth box only, and unlabeled objects aren't pushed to background. The classifier
is reset to the CLIP text embeddings of the collection config's kept classes, as build_dataset.py's Detector classifies among.

Trained by habitat_embodied_sl/main.py (its ObjectViewsTrainer builds CustomFT_Detic and feeds it an
ObjectViewsDataset variant's balanced_loader).
"""
import copy
from typing import Dict, List

import torch
from detectron2.config import configurable  # type: ignore
from detectron2.modeling import META_ARCH_REGISTRY, ROI_HEADS_REGISTRY, GeneralizedRCNN  # type: ignore
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals  # type: ignore
from detectron2.structures import ImageList, Instances  # type: ignore
from torch import nn
from torch.cuda.amp import autocast

from detic.modeling.roi_heads.detic_roi_heads import DeticCascadeROIHeads  # type: ignore


@ROI_HEADS_REGISTRY.register()
class PositiveOnlyDeticCascadeROIHeads(DeticCascadeROIHeads):
    """DeticCascadeROIHeads keeping, at every cascade stage, only the proposals matched to a ground-truth box."""

    def label_and_sample_proposals(self, proposals, targets):
        return select_foreground_proposals(super().label_and_sample_proposals(proposals, targets), self.num_classes)[0]

    def _match_and_label_boxes(self, proposals, stage, targets):
        return select_foreground_proposals(super()._match_and_label_boxes(proposals, stage, targets), self.num_classes)[0]


@META_ARCH_REGISTRY.register()
class CustomFT_Detic(GeneralizedRCNN):
    LOSS_WEIGHTS = {"loss_cls": 1.0, "loss_box_reg": 0.0}

    @configurable
    def __init__(self, *, fp16: bool, **kwargs):
        super().__init__(**kwargs)
        self.fp16 = fp16
        self.proposal_generator.requires_grad_(False)
        self.teacher = nn.ModuleDict({
            "backbone": copy.deepcopy(self.backbone), "proposal_generator": copy.deepcopy(self.proposal_generator),
        }).requires_grad_(False).eval()

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({"fp16": cfg.FP16})
        return ret

    def train(self, mode: bool = True) -> "CustomFT_Detic":
        super().train(mode)
        self.teacher.eval()
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Loads a checkpoint without a teacher (e.g. Detic's own) with the teacher taking its backbone and proposal
        generator: a FrozenBatchNorm2d with no statistics in the checkpoint would otherwise reset them to identity."""
        student_names = [name for name in state_dict if name.startswith((f"{prefix}backbone.", f"{prefix}proposal_generator."))]
        for name in student_names:
            state_dict.setdefault(f"{prefix}teacher.{name[len(prefix):]}", state_dict[name])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _backbone_features(self, images: ImageList) -> Dict[str, torch.Tensor]:
        """The student backbone's features of images, run in half precision when FP16 as in Detic's CustomRCNN.forward."""
        if not self.fp16:
            return self.backbone(images.tensor)
        with autocast():
            features = self.backbone(images.tensor.half())
        return {name: feature.float() for name, feature in features.items()}

    @torch.no_grad()
    def _teacher_proposals(self, images: ImageList) -> list:
        """The frozen teacher's proposals for images, its backbone's full-precision features fed to its own proposal
        generator, as GeneralizedRCNN.inference does, so they match the loaded checkpoint's own."""
        return self.teacher["proposal_generator"](images, self.teacher["backbone"](images.tensor), None)[0]

    def _loss_weight(self, name: str) -> float:
        return next((weight for prefix, weight in self.LOSS_WEIGHTS.items() if name.startswith(prefix)), 1.0)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Weighted ROI-head losses of batched_inputs' boxes on the teacher's proposals when training, else inference's
        detections."""
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self._backbone_features(images)
        # training's add_ground_truth_to_proposals needs every proposal field on the gt too: drop eval-mode "scores"
        proposals = [
            Instances(proposal.image_size, proposal_boxes=proposal.proposal_boxes, objectness_logits=proposal.objectness_logits)
            for proposal in self._teacher_proposals(images)
        ]
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        return {name: loss * self._loss_weight(name) for name, loss in detector_losses.items()}

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """GeneralizedRCNN.inference's detections, the student's ROI heads run on the teacher's proposals."""
        images = self.preprocess_image(batched_inputs)
        results, _ = self.roi_heads(images, self.backbone(images.tensor), self._teacher_proposals(images), None)
        return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
