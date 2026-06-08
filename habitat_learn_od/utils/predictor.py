from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import ImageList, Instances
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from common.env_utils.sense import BBSense
from habitat_learn_od.utils.roi_head_wrappers import MinimalPredictorWrapper
import pytorch_lightning as pl

def get_semantic_map(classes, mask):

    size = mask.shape[1:]
    segm = np.zeros(size)

    for i in range(len(mask)):
        segm[mask[i].bool()] = classes[i]

    segm[mask.sum(0) == 0] = len(BBSense.CLASSES)  # background

    return segm


def get_gt_mask(bbs: Instances, compact_mask):
    size = bbs.image_size
    segm = np.zeros((len(bbs), *size))

    for i in range(len(bbs)):
        segm[i] = bbs.gt_classes[i]

    segm[bbs.gt_masks.sum(0) == 0] = len(BBSense.CLASSES)  # background

    return segm



def setup_cfg(args):
    if isinstance(args, CfgNode):
        cfg = args
    else:
        cfg = get_cfg()

    cfg.MODEL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if hasattr(args, "config_file"):
        cfg.merge_from_file(args.config_file)

    if hasattr(args, "opts"):
        cfg.merge_from_list(args.opts)

    if hasattr(args, "confidence_threshold"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold

    return cfg


class Predictor(pl.LightningModule):
    def __init__(self, cfg=None, input_format=None, load_checkpoint=True, metadata=None, model=None):
        super().__init__()

        if model is None and cfg is not None:
            if isinstance(cfg, CfgNode) is False:
                cfg = setup_cfg(cfg)
            self.cfg = cfg.clone()  # cfg can be modified by model

            model = build_model(self.cfg)
            if load_checkpoint:
                checkpointer = DetectionCheckpointer(model)
                checkpointer.load(cfg.MODEL.WEIGHTS)
        elif model is None and cfg is None:
            model = model_zoo.get(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                trained=load_checkpoint,
                device='cpu',
            )

        assert model is not None, "No model provided"
        self.model = model

        self.test_map_metric = MeanAveragePrecision(class_metrics=True)
        self.model.eval()

    def on_test_epoch_end(self):
        results = self.test_map_metric.compute()
        for k in results.keys():
            if k == "map_per_class":
                for i, v in enumerate(BBSense.CLASSES.values()):
                    self.log(
                        f"test_{k}_{v}",
                        results[k][i],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
            else:
                self.log(
                    f"test_{k}",
                    results[k],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        self.model.eval()

        predictions = self(batch)[0]

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
                "masks": b['instances'].gt_masks.tensor,
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
                'masks': b['instances'].pred_masks,
            }
            for b in predictions
        ]

        self.test_map_metric.update(pred, gt)

    def set_head_wrapper(self, head_class: MinimalPredictorWrapper):
        """We implement custom ROIHead for box-predictor (e.g., heads with different self).
        This function setup the wrapper for the current head
        """
        self.model.roi_heads.box_predictor = head_class(
            self.model.roi_heads.box_predictor
        )

    def head_parameters(self):
        return self.model.roi_heads.box_head.parameters()

    def forward(self, inputs):
        height = inputs[0]['height']
        width = inputs[0]['width']

        images = self.preprocess_image(inputs)

        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in inputs]
        else:
            gt_instances = None

        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, gt_instances)

        instances, _ = self.model.roi_heads(images, features, proposals, gt_instances)
        mask_features = [features[f] for f in self.model.roi_heads.in_features]
        predictions_images = []

        for i in range(len(instances)):
            predictions_images += [i] * len(instances[i])

        predictions_boxes = [instances[i].pred_boxes for i in range(len(instances))]

        pooled_features = self.model.roi_heads.box_pooler(
            mask_features, predictions_boxes
        )
        box_features = self.model.roi_heads.box_head(pooled_features)

        predictions = self.postprocess(height, width, instances)

        return predictions, box_features, predictions_images

    def __call__(self, inputs):
        with torch.no_grad():
            predictions, features, _ = self.forward(inputs)
        return (predictions, features)

    def infer(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
        """

        height, width = original_image[0].shape[:2]

        inputs = [
            {
                "image": torch.as_tensor(im).permute(2, 0, 1),
                "height": height,
                "width": width,
            }
            for im in original_image
        ]
        return self.__call__(inputs)

    def reinit_head(self, classes_indices_keep: list[int]):
        roi_heads = self.model.roi_heads
        predictor = roi_heads.box_predictor

        original_num_classes = predictor.num_classes
        keep = np.array(classes_indices_keep)
        keep_with_bg = np.append(keep, original_num_classes)
        bbox_mask = np.repeat(keep * 4, 4) + np.tile(np.arange(4), len(keep))

        roi_heads.num_classes = predictor.num_classes = len(keep)

        cls_weight = predictor.cls_score.weight[keep_with_bg]
        cls_bias = predictor.cls_score.bias[keep_with_bg]
        in_features = cls_weight.shape[1]
        predictor.cls_score = nn.Linear(in_features, len(keep_with_bg))
        predictor.cls_score.weight = nn.Parameter(cls_weight)
        predictor.cls_score.bias = nn.Parameter(cls_bias)

        bbox_weight = predictor.bbox_pred.weight[bbox_mask]
        bbox_bias = predictor.bbox_pred.bias[bbox_mask]
        predictor.bbox_pred = nn.Linear(in_features, len(keep) * 4)
        predictor.bbox_pred.weight = nn.Parameter(bbox_weight)
        predictor.bbox_pred.bias = nn.Parameter(bbox_bias)

        if hasattr(roi_heads, "mask_head"):
            mask_pred = roi_heads.mask_head.predictor
            mask_pred.weight = nn.Parameter(mask_pred.weight[keep])
            mask_pred.bias = nn.Parameter(mask_pred.bias[keep])
            mask_pred.num_classes = len(keep)

    @torch.no_grad()
    def head_forward(self, images, features, proposals):
        box_features = [features[f] for f in self.model.roi_heads.in_features]

        predictions_boxes = [x.proposal_boxes for x in proposals]

        pooled_features = self.model.roi_heads.box_pooler(
            box_features, predictions_boxes
        )
        box_features = self.model.roi_heads.box_head(pooled_features)

        predictions = self.model.roi_heads.box_predictor(box_features)
        pred_instances, _ = self.model.roi_heads.box_predictor.inference(
            predictions, proposals
        )

        outputs = self.model.roi_heads.forward_with_given_boxes(
            features, pred_instances
        )
        return outputs

    def postprocess(self, height, width, results):
        processed_results = []
        for results_per_image in results:
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.model.device) for x in batched_inputs]
        images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        return images