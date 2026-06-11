import logging
import os
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling import build_model
from detectron2.config import CfgNode, get_cfg

from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

from common.utils import triplet
from common.vision.clip import get_clip_embeddings, save_clip_embeddings, load_clip_embeddings
from common.vision.detic import (
    reset_cls, CustomRCNN, DETIC_ROOT, add_centernet_config, add_detic_config,
)


__ALL__ = ['MultiStageModel']
log = logging.getLogger(__name__)

def setup_cfg(config_file, confidence_threshold, weights) -> CfgNode:

    config_file = str(Path(DETIC_ROOT).resolve().parent / config_file)
    weights = str(Path(DETIC_ROOT).resolve().parent / weights)

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(
        Path(DETIC_ROOT)
        / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )
    cfg.WITH_IMAGE_LABELS = False
    cfg.MODEL.DYNAMIC_CLASSIFIER = False

    cfg.freeze()
    return cfg


def build_detic_model(cfg: CfgNode, vocab: list[str], vocab_name: str) -> CustomRCNN:
    model: CustomRCNN = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    path_clip_embeddings = Path(f"data/metadata/{vocab_name}.npy")

    if not os.path.exists(path_clip_embeddings):
        classifier = get_clip_embeddings(vocab)
        save_clip_embeddings(classifier, path_clip_embeddings)
    else:
        classifier = load_clip_embeddings(path_clip_embeddings)

    reset_cls(model, classifier, len(vocab))

    model.num_classes = len(vocab)
    return model


class MultiStageModel(pl.LightningModule):
    model: CustomRCNN

    def __init__(
        self,
        detic_args=None,
        lr=0.01,
        loss_weights={},
        use_gt_matching=True,
        optimizer="SGD",
        optimizer_params={},
        loss_margin=0.3,
        mask_on=True,
        load_checkpoint=True,
        vocab=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        metadata = MetadataCatalog.get(vocab)
        classes = metadata.thing_classes
        colors = metadata.thing_colors
        cfg = setup_cfg(detic_args.config_file, detic_args.confidence_threshold, detic_args.weights)
        model = build_detic_model(cfg, vocab=classes, vocab_name=vocab)
        assert isinstance(model, CustomRCNN), "Expected model to be an instance of CustomRCNN"
        self.model = model
        model.to(self.device)

        self.lr = lr

        self.loss_weights = loss_weights
        self.use_gt_matching = use_gt_matching
        self.loss_margin = loss_margin
        self.optimizer = optimizer
        self.opt_params = optimizer_params  # Params for optimizer

        self.feature_projector = triplet.tinyprojection_MLP(1024, out_dim=128)
        self.model.roi_heads.mask_on = mask_on  # Only box-prediction component
        
        # Stages losses
        self.test_map_metric = MeanAveragePrecision(class_metrics=True)
        self.model.eval()


    def _compute_contrastive_loss(self, features, y):
        if self.compute_projector_loss:
            y_mask = y != -1
            y = y[y_mask]

            if len(y) > 1:
                features = self.feature_projector(features[y_mask])
                return triplet.online_mine_hard(
                    y.to(self.device), features, self.loss_margin, device=self.device
                )[0]
            else:
                return features.sum() * 0.0  # connect the gradient
        else:
            return None


    def model_forward(self, inputs: list[dict]) -> tuple[list[dict], dict, torch.Tensor, torch.Tensor]:
        assert self.model.training
        gt_instances = []

        for x in inputs:
            x = x["instances"]

            if hasattr(x, "infos") and len(x.infos) > 0:
                x.gt_ids = torch.tensor(
                    [i['object_id'] for i in x.infos], dtype=torch.int16, device=self.device
                )
            else:
                x.gt_ids = torch.ones(len(x), device=self.device) * -1
            gt_instances.append(x) 
        
        images = self.model.preprocess_image(inputs)
        
        # Backbone forward
        features = self.model.backbone(images.tensor)
        
        # RPN forward
        proposals, proposal_losses = self.model.proposal_generator(images, features, gt_instances=gt_instances)
        
        # Sample and label proposals
        labeled_proposals = self.model.roi_heads.label_and_sample_proposals(proposals, targets=gt_instances)
        labeled_proposal_gt_classes = torch.cat([p.gt_classes for p in labeled_proposals])
        labeled_proposals_ids = torch.cat(
            [
                p.gt_ids if hasattr(p, "gt_ids") else torch.ones(len(p), device=self.device) * -1
                for p in labeled_proposals
            ]
        )
        roi_head_losses = self.model.roi_heads._forward_box(features, labeled_proposals, targets=gt_instances,ann_type="box")
        losses = {**proposal_losses, **roi_head_losses}

        # Get box features and predictions
        feat_list = [features[f] for f in self.model.roi_heads.box_in_features]
        labeled_proposal_boxes = [p.proposal_boxes for p in labeled_proposals]
        pooled_box_features = self.model.roi_heads.box_pooler(feat_list, labeled_proposal_boxes)
        box_features = self.model.roi_heads.box_head[0](pooled_box_features)
        box_cls_predictions, box_reg_predictions = self.model.roi_heads.box_predictor[0](box_features)
        
        # also do inference for evaluation
        pred_instances, _ = self.model.roi_heads.box_predictor[0].inference(
            (box_cls_predictions, box_reg_predictions), labeled_proposals
        )    
        
        # Get foreground mask
        foreground_mask = labeled_proposal_gt_classes < self.model.roi_heads.num_classes
        box_features = box_features[foreground_mask]
        labeled_proposals_ids = labeled_proposals_ids[foreground_mask]
        box_cls_predictions = box_cls_predictions[foreground_mask]
    
        results = []
        for input, pred in zip(inputs, pred_instances):
            height, width = input["image"].shape[1:]
            r = detector_postprocess(pred, height, width).detach()
            results.append({"instances": r})

        return results, losses, box_features, labeled_proposals_ids
    
    
    def inference(self, inputs: list[dict]) -> list[dict]:
        assert not self.model.training
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, gt_instances=None)
        pred_instances, _ = self.model.roi_heads(images, features, proposals, targets=None)

        results = []
        for input, pred in zip(inputs, pred_instances):
            height, width = input["image"].shape[1:]
            r = detector_postprocess(pred, height, width).detach()
            results.append({"instances": r})
        return results


    def configure_optimizers(self, *args, **kwargs) -> torch.optim.Optimizer:
        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )
        return optimizer