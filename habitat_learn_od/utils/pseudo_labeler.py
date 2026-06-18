import gc
import itertools as it
import logging
import time
from copy import deepcopy
from multiprocessing.pool import Pool
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from detectron2.structures import Boxes, Instances # type: ignore
from omegaconf import DictConfig
from torch.nn import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

from common.env_utils.sense import BBSense
from common.utils import projection_utils as pu
from common.utils.matching import get_objects_ids

from habitat_learn_od.utils.two_stage_models import TwoStageModel

log = logging.getLogger(__name__)


class PseudoLabeler(pl.LightningModule):
    model: TwoStageModel
    
    def __init__(self, model: TwoStageModel, thr=0.7, overlap_thr=0.5, *args, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.thr = thr
        self.reinit(self.model)
        self.overlap_thr = overlap_thr
        
    def reinit(self, model) -> None:
        self.update_model(model)
        self.test_map_metric = MAP(class_metrics=True)

    def update_model(self, model) -> None:
        self.model = model
        self.model.model.roi_heads.box_predictor.test_score_thresh = self.thr
        self.model.eval()

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> tuple[List[Instances], List[dict]]:
        self.model.eval()
        instances = self(batch)
        return instances

    def forward(self, batch) -> tuple[list[Instances], List[dict]]:
        self.model.eval()
        preds = [x['instances'].to('cpu') for x in self.model(batch)[0]]
        ids = get_objects_ids(batch, preds)
        return preds, ids

    def get_pseudo_labels(self, *args, **kwargs) -> List[Instances]:
        raise NotImplementedError


class VanillaPseudoLabeler(PseudoLabeler):
    def __init__(self, model: TwoStageModel, temperature=1, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs) -> List[Instances]:
        result = []

        for out in model_outs:
            for pred, infos in zip(out[0], out[1]):
                mask = pred.scores > 0.0001
                pred = pred[mask]
                target = Instances(len(pred)) # type: ignore
                target.gt_classes = pred.pred_classes
                target.gt_boxes = pred.pred_boxes

                if len(pred) > 0:
                    target.gt_logits = pred.gt_logits
                else:
                    target.gt_logits = torch.Tensor()

                target.scores = pred.scores
                target.gt_masks = pred.pred_masks

                target.infos = [info for idx, info in enumerate(infos) if mask[idx]]
                result.append(target)
                
        return result


class SemanticMapPseudoLabeler(PseudoLabeler):
    def __init__(
        self, model: TwoStageModel, thr=0.7, temperature=1.0, solution="ours", *args, **kwargs
    ):
        super().__init__(model=model, thr=thr)
        self.temperature = temperature
        self.solution = solution

    def reinit(self, model) -> None:
        super().reinit(model)
        self.global_pcds = {}

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> tuple[List[Instances], List[dict]]:
        self.model.eval()
        instances, infos = self(batch)

        for b, prediction, info in zip(batch, instances, infos):
            episode = b['episode']

            if episode in self.global_pcds:
                episode_pcd = self.global_pcds[episode]
            else:
                episode_pcd = pu.SemanticPointCloud(
                    episode=episode, solution=self.solution
                )
                self.global_pcds[episode] = episode_pcd

            _pcd = pu.project_semantic_masks_to_3d(
                b['depth'].squeeze(0),
                b['location'],
                prediction.to(b['depth'].device),
                info,
                update_logits=False # do not update logits locally, only per episode
            )
            _pcd._episode = episode

            episode_pcd += _pcd

            if len(_pcd):
                episode_pcd.update_logits(prediction, info)
             #     # episode_pcd.preprocess()
        return instances, infos

    def get_pseudo_labels(self, model_outs, dataloader) -> List[Instances]:
        results = []

        for k in self.global_pcds.keys():
            self.global_pcds[k].preprocess()

        for batch in tqdm.tqdm(dataloader):
            for data in batch:
                pcd = self.global_pcds[data['episode']]

                # Compute the ray intersections.
                _time = time.time()
                
                (
                    semantic_masks,
                    object_ids,
                    classes,
                    r_logits,
                    _
                ) = pcd._depth_raytracing(data['depth'].squeeze(), data['location'])

                log.info(f"It took {time.time() - _time} for raytracing")
                t = Instances(image_size=data['depth'].squeeze().shape)

                # Get bbs from semantic
                resolved_masks = []
                bounding_boxes = []
                logits = []
                resolved_classes = []
                ids = []

                for mask, object_id, cls, l in zip(
                    semantic_masks, object_ids, classes, r_logits
                ):
                    bb = cv2.boundingRect(mask.numpy().astype('uint8'))

                    x, y, w, h = bb
                    if w == 0 or h == 0:
                        continue
                    if cls >= len(BBSense.CLASSES):
                        continue  # Background or overflowd class
                    logits.append(l)

                    resolved_class = cls

                    ids.append(object_id)
                    bounding_boxes.append(
                        torch.tensor([x, y, x + w, y + h]).unsqueeze(0)
                    )
                    resolved_classes.append(resolved_class)

                    resolved_masks.append(mask)

                t.gt_classes = (
                    torch.tensor(resolved_classes)
                    if len(resolved_classes) > 0
                    else torch.Tensor()
                )

                t.gt_masks = (
                    torch.stack(resolved_masks)
                    if len(resolved_masks)
                    else torch.Tensor()
                )
                # TODO uncertainty information per bbox

                t.gt_logits = torch.stack(logits) if len(logits) > 0 else torch.Tensor()

                t.infos = [{'id_object': id.item()} for id in ids]

                t.gt_boxes = (
                    Boxes(torch.cat(bounding_boxes))
                    if len(bounding_boxes)
                    else Boxes(torch.Tensor())
                )
                results.append(t)
        gc.collect()
        return results


class SoftPseudoLabeler(PseudoLabeler):
    def __init__(self, model: TwoStageModel, temperature=1) -> None:
        super().__init__(model)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs) -> List[Instances]:
        results = []

        predictions = list(it.chain(*[m[0] for m in model_outs]))
        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))
        max_id = max([m['id_object'] for m in y_ids])
        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }

        pred_counting = 0

        for idx in range(len(predictions)):
            preds_per_image = predictions[idx]

            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []
                results.append(target)
                continue

            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)

            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()

                logits = logits_per_instance[y]
                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)
                resolved_class = torch.argmax(soft_softmax[:-1])
                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            results.append(target)

        return results


class MajorityPseudoLabeler(PseudoLabeler):
    def __init__(self, model: TwoStageModel, temperature=1) -> None:
        super().__init__(model)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs) -> List[Instances]:
        results = []
        
        predictions = list(it.chain(*[m[0] for m in model_outs]))

        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))

        max_id = max([m['id_object'] for m in y_ids])

        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }
        pred_counting = 0

        for idx in range(len(predictions)):
            preds_per_image = predictions[idx]
            
            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []

                results.append(target)
                continue

            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)
            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()

                logits = logits_per_instance[y]

                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)
                classes, counts = logits.argmax(1).unique(return_counts=True)
                resolved_class = classes[counts.argmax()]
                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            results.append(target)

        return results


class MaxScorePseudoLabeler(PseudoLabeler):
    def __init__(self, model: TwoStageModel, temperature=1) -> None:
        super().__init__(model)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs) -> List[Instances]:
        results = []

        predictions = list(it.chain(*[m[0] for m in model_outs]))
        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))
        max_id = max([m['id_object'] for m in y_ids])

        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }

        pred_counting = 0

        for idx in range(len(predictions)):
            preds_per_image = predictions[idx]

            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []

                results.append(target)
                continue

            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)
            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()

                logits = logits_per_instance[y]

                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)

                values, indexes = F.softmax(logits).max(1)
                resolved_class = indexes[values.argmax()]

                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            results.append(target)

        return results
