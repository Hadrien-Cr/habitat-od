"""Detic's box-supervised GeneralizedRCNN (CustomRCNN's box path) for fine-tuning on the single-box object views
habitat_embodied_sl/main.py writes, registered as MODEL.META_ARCHITECTURE "CustomFT_Detic" (example config:
config/finetune_detic_swinb.yaml). Training returns Detic's losses, the RPN (CenterNet), ROI classification and box
regression ones each weighted by CustomFT_Detic.LOSS_WEIGHTS, plus a
consistency loss between the student's class-agnostic CenterNet heatmaps and those of a frozen teacher copy of the initial
backbone and proposal generator; FREEZE_RPN freezes the proposal generator. Each view labels one object of its frame, so the ROI heads (PositiveOnlyDeticCascadeROIHeads)
learn from proposals matched to a ground-truth box only, and unlabeled objects aren't pushed to background. The classifier
is reset to the CLIP text embeddings of the collection config's kept classes, as main.py's Detector classifies among.

Trained by root/train.py (its ObjectViewsTrainer builds CustomFT_Detic and feeds it an
ObjectViewsDataset variant's balanced_loader).
"""
import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
from detectron2.config import configurable  # type: ignore
from detectron2.modeling import META_ARCH_REGISTRY, ROI_HEADS_REGISTRY, GeneralizedRCNN  # type: ignore
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals  # type: ignore
from detectron2.structures import ImageList  # type: ignore
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
    LOSS_WEIGHTS = {
        "loss_centernet": 0.0, "loss_cls": 1.0, "loss_box_reg": 1.0, "loss_rpn_consistency": 1.0,
    }
    FREEZE_RPN = False

    @configurable
    def __init__(self, *, fp16: bool, **kwargs):
        super().__init__(**kwargs)
        self.fp16 = fp16
        self.proposal_generator.requires_grad_(not self.FREEZE_RPN)

        self.teacher = nn.ModuleDict({
            "backbone": copy.deepcopy(self.backbone), "proposal_generator": copy.deepcopy(self.proposal_generator),
        }).requires_grad_(False).eval()
        # registered after the teacher's deepcopy so that only the student's head keeps its heatmaps
        self.proposal_generator.centernet_head.register_forward_hook(self._keep_student_heatmaps)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({"fp16": cfg.FP16})
        return ret

    def train(self, mode: bool = True) -> "CustomFT_Detic":
        super().train(mode)
        self.teacher.eval()
        return self

    def set_teacher(self) -> None:
        """Copies the student's backbone and proposal generator weights into the teacher."""
        self.teacher["backbone"].load_state_dict(self.backbone.state_dict())
        self.teacher["proposal_generator"].load_state_dict(self.proposal_generator.state_dict())

    def _keep_student_heatmaps(self, head: nn.Module, inputs: tuple, outputs: tuple) -> None:
        self._student_heatmaps = outputs[2]

    def _backbone_features(self, backbone: nn.Module, images: ImageList) -> Dict[str, torch.Tensor]:
        """backbone's features of images, run in half precision when FP16 as in Detic's CustomRCNN.forward."""
        if not self.fp16:
            return backbone(images.tensor)
        with autocast():
            features = backbone(images.tensor.half())
        return {name: feature.float() for name, feature in features.items()}

    def _consistency_loss(self, images: ImageList) -> torch.Tensor:
        """Mean over levels of the BCE of the student's CenterNet heatmap logits (kept from the proposal generator's last
        forward) against the teacher's probabilities."""
        teacher_rpn = self.teacher["proposal_generator"]

        with torch.no_grad():
            features = self._backbone_features(self.teacher["backbone"], images)
            teacher_heatmaps = teacher_rpn.centernet_head([features[name] for name in teacher_rpn.in_features])[2]

        return torch.stack([
            F.binary_cross_entropy_with_logits(student, teacher.sigmoid()) for student, teacher in zip(self._student_heatmaps, teacher_heatmaps)
        ]).mean()

    def _loss_weight(self, name: str) -> float:
        return next((weight for prefix, weight in self.LOSS_WEIGHTS.items() if name.startswith(prefix)), 1.0)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Weighted losses of batched_inputs' boxes when training, else GeneralizedRCNN.inference's detections."""
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self._backbone_features(self.backbone, images)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        losses = {**proposal_losses, **detector_losses, "loss_rpn_consistency": self._consistency_loss(images)}
        return {name: loss * self._loss_weight(name) for name, loss in losses.items()}
