import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.instances import Instances # type: ignore

from habitat_learn_od.utils.detectron_utils import fast_rcnn_inference
from common.env_utils.sense import BBSense


class MinimalPredictorWrapper(nn.Module):
    def __init__(self, prediction_head: FastRCNNOutputLayers):
        super().__init__()
        assert isinstance(
            prediction_head, FastRCNNOutputLayers
        ), "Trying to wrap a ROIHead different from FastRCNNOutputLayer"
        self.box_predictor = prediction_head

    def forward(self, x):
        return self.box_predictor(x)

    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)

    def inference(self, predictions, proposals):
        return self.box_predictor.inference(predictions, proposals)


class BoxPredictorWrapper(MinimalPredictorWrapper):
    def __init__(
        self, prediction_head: FastRCNNOutputLayers, cls_loss=None, *args, **kwargs
    ):
        super().__init__(prediction_head)
        if cls_loss is None:
            cls_loss = lambda x, y, reduction: cross_entropy(  # noqa: E731
                x, y, reduce=False, reduction=reduction
            )
        self.cls_loss = cls_loss

    def forward(self, x):
        return self.box_predictor(x)

    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)

    def inference(self, predictions, proposals) -> tuple[list[Instances], list[torch.Tensor]]:
        logits, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]

        boxes = self.box_predictor.predict_boxes(predictions, proposals)
        logits_per_image = F.softmax(logits, dim=-1).split(num_inst_per_image, dim=0)

        scores = self.box_predictor.predict_probs(predictions, proposals)

        shapes = [x.image_size for x in proposals]

        return fast_rcnn_inference(
            boxes,
            scores,
            shapes,
            self.box_predictor.test_score_thresh,
            self.box_predictor.test_nms_thresh,
            self.box_predictor.test_topk_per_image,
            logits=logits_per_image,
        ) # type: ignore