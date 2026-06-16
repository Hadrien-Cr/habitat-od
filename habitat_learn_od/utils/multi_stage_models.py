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
from detectron2.utils.events import get_event_storage

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference

from common.utils import triplet
from common.vision.clip import get_clip_embeddings, save_clip_embeddings, load_clip_embeddings
from common.vision.detic import (
    reset_cls, CustomRCNN, DETIC_ROOT, add_centernet_config, add_detic_config,
)

def detach_instances(instances):
    detached = Instances(instances.image_size)

    for k, v in instances.get_fields().items():
        if isinstance(v, torch.Tensor):
            detached.set(k, v.detach())

        elif hasattr(v, "tensor"):  # Boxes, RotatedBoxes, etc.
            new_v = type(v)(v.tensor.detach())
            detached.set(k, new_v)

        else:
            detached.set(k, v)

    return detached

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


def build_detic_model(cfg: CfgNode, classes: list[str], vocab_name: str) -> CustomRCNN:
    model: CustomRCNN = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    path_clip_embeddings = Path(f"data/metadata/{vocab_name}.npy")

    if not os.path.exists(path_clip_embeddings):
        classifier = get_clip_embeddings(["a " + c.replace("_", " ") for c in classes])
        save_clip_embeddings(classifier, path_clip_embeddings)
    else:
        classifier = load_clip_embeddings(path_clip_embeddings)

    reset_cls(model, classifier, len(classes))

    model.num_classes = len(classes)
    return model


class MultiStageModel(pl.LightningModule):
    model: CustomRCNN

    def __init__(
        self,
        detic_args: CfgNode,
        lr=0.01,
        loss_weights={},
        use_gt_matching=True,
        optimizer="SGD",
        optimizer_params={},
        loss_margin=0.3,
        mask_on=True,
        load_checkpoint=True,
        vocab_name: str = "HSSD40",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        metadata = MetadataCatalog.get(vocab_name)
        classes = metadata.thing_classes
        colors = metadata.thing_colors
        cfg = setup_cfg(detic_args.config_file, detic_args.confidence_threshold, detic_args.weights)
        model = build_detic_model(cfg, classes=classes, vocab_name=vocab_name)
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
        roi_head_losses = self.roi_head_forward(features, labeled_proposals, targets=gt_instances)
        losses = {**proposal_losses, **roi_head_losses}        

        # Get box features and predictions
        feat_list = [features[f] for f in self.model.roi_heads.box_in_features]
        labeled_proposal_boxes = [p.proposal_boxes for p in labeled_proposals]
        pooled_box_features = self.model.roi_heads.box_pooler(feat_list, labeled_proposal_boxes)
        box_features = self.model.roi_heads.box_head[0](pooled_box_features)
        box_cls_predictions, box_reg_predictions = self.model.roi_heads.box_predictor[0](box_features)
        
        # Get foreground mask
        foreground_mask = labeled_proposal_gt_classes < self.model.roi_heads.num_classes
        box_features = box_features[foreground_mask]
        labeled_proposals_ids = labeled_proposals_ids[foreground_mask]
        box_cls_predictions = box_cls_predictions[foreground_mask]
    
        # also do inference for evaluation
        proposals = self.centernet_inference(images, features)        
        results = self.roi_head_inference(inputs, features, proposals)
        return results, losses, box_features, labeled_proposals_ids

    def model_inference(self, inputs: list[dict]) -> list[dict]:
        assert not self.model.training
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        proposals = self.centernet_inference(images, features)      
        results = self.roi_head_inference(inputs, features, proposals)

        return results

    def centernet_inference(self, images, features) -> list[dict]:
        feat_list = [features[f] for f in self.model.roi_heads.box_in_features]

        clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = \
            self.model.proposal_generator.centernet_head(feat_list)
        grids = self.model.proposal_generator.compute_grids(feat_list)
        
        proposals, _ = self.model.proposal_generator.inference(
            images, clss_per_level, reg_pred_per_level, 
            agn_hm_pred_per_level, grids)
        
        return proposals
    
    def roi_head_inference(self, inputs, features, proposals) -> list[Instances]:
        if self.model.roi_heads.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]
        
        feat_list = [features[f] for f in self.model.roi_heads.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.model.roi_heads.num_cascade_stages):
            if k > 0:
                proposals = self.model.roi_heads._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
            predictions = self.model.roi_heads._run_stage(feat_list, proposals, k)
            prev_pred_boxes = self.model.roi_heads.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((self.model.roi_heads.box_predictor[k], predictions, proposals))
        
        # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
        scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
        scores = [
            sum(list(scores_per_image)) * (1.0 / self.model.roi_heads.num_cascade_stages)
            for scores_per_image in zip(*scores_per_stage)
        ]
        if self.model.roi_heads.mult_proposal_score:
            scores = [(s * ps[:, None]) ** 0.5 \
                for s, ps in zip(scores, proposal_scores)]
        if self.model.roi_heads.one_class_per_proposal:
            scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
        
        predictor, predictions, proposals = head_outputs[-1]
        boxes = predictor.predict_boxes(
            (predictions[0], predictions[1]), proposals)
        
        pred_instances, _ = fast_rcnn_inference(
            boxes,
            scores,
            image_sizes,
            predictor.test_score_thresh,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )

        boxes = [x.pred_boxes for x in pred_instances]
        box_features = self.model.roi_heads.mask_pooler(feat_list, boxes)
        box_features = self.model.roi_heads.mask_head.layers(box_features)
        mask_rcnn_inference(box_features, pred_instances)

        results = []
        for input, pred in zip(inputs, pred_instances):
            height, width = input["image"].shape[1:]
            r = detector_postprocess(pred, height, width)
            results.append({"instances": detach_instances(r)})

        return results

    def roi_head_forward(self, features, proposals, targets=None) -> dict:
        assert self.model.training
        
        feat_list = [features[f] for f in self.model.roi_heads.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.model.roi_heads.num_cascade_stages):
            if k > 0:
                proposals = self.model.roi_heads._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                proposals = self.model.roi_heads._match_and_label_boxes(
                    proposals, k, targets)
                
            predictions = self.model.roi_heads._run_stage(feat_list, proposals, k)
            prev_pred_boxes = self.model.roi_heads.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((self.model.roi_heads.box_predictor[k], predictions, proposals))
        
        losses = {}
        storage = get_event_storage()
        for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
            with storage.name_scope("stage{}".format(stage)):
                stage_losses = predictor.losses((predictions[0], predictions[1]), proposals,)
            losses.update({k + "_stage{}".format(stage): v  for k, v in stage_losses.items()})
        return losses

    def configure_optimizers(self, *args, **kwargs) -> torch.optim.Optimizer:
        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )
        return optimizer