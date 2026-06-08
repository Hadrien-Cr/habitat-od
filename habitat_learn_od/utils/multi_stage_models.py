import logging
from copy import copy, deepcopy
from typing import List

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from common.utils import triplet
from habitat_learn_od.utils.predictor import Predictor, setup_cfg
from habitat_learn_od.utils.roi_head_wrappers import BoxPredictorWrapper

__ALL__ = ['MultiStageModel']
log = logging.getLogger(__name__)


class MultiStageModel(Predictor):
    def __init__(
        self,
        cfg=None,
        lr=0.01,
        loss_weights={},
        use_gt_matching=True,
        optimizer="SGD",
        optimizer_params={},
        compute_loss=True,
        loss_margin=0.3,
        head_cls=BoxPredictorWrapper,
        mask_on=True,
        load_checkpoint=True,
        vocab=None,
        *args,
        **kwargs,
    ):

        cfg = setup_cfg(cfg).clone()
        super().__init__(setup_cfg(cfg), load_checkpoint=load_checkpoint)

        metadata = MetadataCatalog.get(vocab)
        classes = metadata.thing_classes
        colors = metadata.thing_colors

        self.set_head_wrapper(head_cls)
        self.lr = lr

        self.loss_weights = loss_weights
        self.use_gt_matching = use_gt_matching
        self.loss_margin = loss_margin
        self.optimizer = optimizer
        self.opt_params = optimizer_params  # Params for optimizer

        self.feature_projector = triplet.tinyprojection_MLP(1024, out_dim=128)

        # Stages losses
        self.compute_head_loss = True
        self.compute_projector_loss = True
        self.compute_proposal_loss = True

        self.model.roi_heads.mask_on = mask_on  # Only box-prediction component
        
    def featureprojector_training_mode(self):
        self.compute_head_loss = False
        self.compute_proposal_loss = False
        self.compute_projector_loss = True

    def proposal_training_mode(self):
        self.compute_proposal_loss = True
        self.compute_head_loss = False
        self.compute_projector_loss = False

    def classifier_finetune_mode(self):
        self.compute_proposal_loss = False
        self.compute_head_loss = True
        self.compute_projector_loss = False
        self.compute_projector_loss = True

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

        for param in self.feature_projector.parameters():
            param.requires_grad = True

    def configure_optimizers(self, *args, **kwargs):
        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )
        return optimizer

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

    def common_step(self, batch):
        (
            predictions,
            pred_loss,
            box_features,
            y_matching,
        ) = self._compute(batch)

        contrastive_loss = None
        if self.loss_weights['contrastive_loss'] > 0:
            contrastive_loss = self._compute_contrastive_loss(box_features, y_matching)

        result = {}
        if contrastive_loss is not None:
            contrastive_loss = contrastive_loss * self.loss_weights['contrastive_loss']
            result['loss_contrastive'] = contrastive_loss

        if pred_loss is not None:
            for key, _ in pred_loss.items():
                pred_loss[key] *= self.loss_weights[key]
            result = {**result, **pred_loss}

        return result, predictions

    @torch.no_grad()
    def __call__(self, inputs):
        self.eval()
        height = inputs[0]['height']
        width = inputs[0]['width']

        images = self.preprocess_image(inputs)

        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to(self.model.device) for x in inputs]
        else:
            gt_instances = None

        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, gt_instances)

        instances, _ = self.model.roi_heads(images, features, proposals, gt_instances)
        mask_features = [features[f] for f in self.model.roi_heads.in_features]
        predictions_images = []

        for i in range(len(instances)):
            predictions_images += [i] * len(instances[i])

        if gt_instances is not None:
            boxes = [gt_instances[i].gt_boxes for i in range(len(gt_instances))]
        else:
            boxes = [instances[i].pred_boxes for i in range(len(instances))]

        pooled_features = self.model.roi_heads.box_pooler(mask_features, boxes)
        box_features = self.feature_projector(
            self.model.roi_heads.box_head(pooled_features)
        )

        predictions = self.postprocess(height, width, instances)

        return predictions, box_features  # , predictions_images

    def _compute(self, batched_inputs):
        inputs = []

        for i in batched_inputs:
            if isinstance(i, List):
                inputs += i
            else:
                inputs.append(i)

        gt_instances = []

        for x in inputs:
            x = x["instances"].to(self.model.device)
            if hasattr(x, "infos"):
                x.gt_ids = torch.tensor(
                    [i['object_id'] for i in x.infos], dtype=torch.int16, device=self.model.device
                )
            else:
                x.gt_ids = torch.ones(len(x), dtype=torch.int16, device=self.model.device) * -1

            gt_instances.append(x)

        images = self.preprocess_image(inputs)

        # Forward pass
        features = self.model.backbone(images.tensor)
        proposals, prop_loss = self.model.proposal_generator(
            images, features, gt_instances
        )
        mask_features = [features[f] for f in self.model.roi_heads.in_features]

        # Proposal matching
        labeled_props = self.model.roi_heads.label_and_sample_proposals(
            proposals, gt_instances
        )
        prop_boxes = [p.proposal_boxes for p in labeled_props]
        prop_ids = torch.cat(
            [
                p.gt_ids if hasattr(p, "gt_ids") else torch.ones(len(p), dtype=torch.int16, device=self.model.device) * -1
                for p in labeled_props
            ]
        )
        prop_classes = torch.cat([p.gt_classes for p in labeled_props])
        box_features = self.model.roi_heads.box_pooler(mask_features, prop_boxes)
        box_features = self.model.roi_heads.box_head(box_features)
        y_mask = prop_classes != self.model.roi_heads.num_classes  # Background
        box_features = box_features[y_mask]
        prop_ids = prop_ids[y_mask]
        height = inputs[0]['height']
        width = inputs[0]['width']

        prediction_loss = {**prop_loss}
        
        if self.compute_head_loss:
            self.model.roi_heads.train()
            _, head_loss = self.model.roi_heads(
                images, features, proposals, gt_instances
            )
            prediction_loss = {**prediction_loss, **head_loss}
        
        self.model.roi_heads.eval()
        instances = self.head_forward(images, features, proposals)
        self.model.roi_heads.train()
        outputs = self.postprocess(height, width, instances)

        return (
            outputs,
            prediction_loss,
            box_features,
            prop_ids,
        )