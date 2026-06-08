
import logging
from copy import copy, deepcopy
import albumentations as A
import habitat # type: ignore
import numpy as np
import pytorch_lightning as pl
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog 
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

import wandb

from common.utils.plot_utils import plot_segmentation_gt, plot_segmentation_pred, plot_segmentation_gt_and_pred
from common.utils.dataset_utils import _transform_batch_with_logits

from habitat_learn_od.utils.train_helpers import mixup_batch
from habitat_learn_od.utils.multi_stage_models import MultiStageModel
from habitat_learn_od.utils import multi_stage_models

from habitat_learn_od.utils.pseudo_labeler import (
    PseudoLabeler,
    SemanticMapPseudoLabeler,
    SoftPseudoLabeler,
    VanillaPseudoLabeler,
)

log = logging.getLogger(__name__)

class TeacherStudent(pl.LightningModule):
    pseudo_labeler: PseudoLabeler
    online_network: MultiStageModel

    def __init__(
        self,
        detectron_args,
        pseudo_labeler_method="vanilla",
        temperature=1,
        student_model=None,
        thr=0.7,
        freeze_teacher=True,
        use_teacher=False,
        batch_size=1,
        mixup=False,
        solution="ours",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        switch = {
            "logits": SoftPseudoLabeler,
            "vanilla": VanillaPseudoLabeler,
            "semantic_map": SemanticMapPseudoLabeler,
        }
        # Initialize student and teacher / pseudo-labeler

        if student_model is not None:
            self.student_model_class = getattr(multi_stage_models, student_model)
        else:
            self.student_model_class = multi_stage_models.FocalMultiStageModel # type: ignore

        self.kwargs = kwargs
        self.batch_size = batch_size
        self.mixup = mixup
        self.use_teacher = use_teacher
        self.detectron_args = detectron_args

        self.pseudo_labeler: PseudoLabeler = switch[pseudo_labeler_method](
            model=multi_stage_models.MultiStageModel(detectron_args, **kwargs),
            temperature=temperature,
            thr=thr,
            solution=solution,
            **kwargs
        )
        self.max_steps = None
        if freeze_teacher:
            self.pseudo_labeler.freeze()
        else:
            self.pseudo_labeler.train()

        self.freeze_teacher = freeze_teacher

        self.online_val_map_metric = MAP(class_metrics=True)
        self.test_map_metric = MAP(class_metrics=True)
        self.reinit_online()


    def reinit_online(self) -> None:
        self.online_network = self.student_model_class(self.detectron_args, **self.kwargs)
        self.online_network.model.roi_heads.box_predictor.box_predictor.test_score_thresh = 0.5


    def training_step(self, batched_inputs, batch_idx) -> Tensor:
        batch = []
        for i in batched_inputs:
            if isinstance(i, list):
                batch += i
            else:
                batch.append(i)
        
        if self.mixup:
            mixup_batch(batch)

        losses, predictions = self.online_network.common_step(batch)
        loss: Tensor = sum(losses.values()) # type: ignore

        for k in losses.keys():
            self.log(
                f"train_{k}",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        if self.use_teacher:
            self.target_validation_step(batch, batch_idx)
        self.online_validation_step(batch, batch_idx)

    def target_validation_step(self, batch, batch_idx) -> None:
        self.pseudo_labeler.eval()
        losses, predictions = self.pseudo_labeler.model.common_step(batch)
        if losses is None:
            return None
        
        loss = sum(losses.values())

        for k in losses.keys():
            self.log(
                f"val_{k}_target",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self.log(
            'val_loss_target',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def _val_map(self, batch, predictions) -> None:
        device = self.device
        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor.to(device),
                'labels': b['instances'].gt_classes.int().to(device),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor.to(device),
                'labels': b['instances'].pred_classes.to(device),
                'scores': b['instances'].scores.to(device),
            }
            for b in predictions
        ]
        self.online_val_map_metric.update(pred, gt)

    def online_validation_step(self, batch, batch_idx) -> None:
        self.online_network.eval()
        losses, predictions = self.online_network.common_step(batch)

        if losses is None:
            return None
        
        loss = sum(losses.values())

        for k in losses.keys():
            self.log(
                f"val_{k}_online",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self.log(
            'val_loss_online',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        if (batch_idx%10 == 0):
            self.log_batch(batch, batch_idx, predictions, prefix="val_online")

        self._val_map(batch, predictions)
        # self._val_map_segm(batch, predictions)

    def validation_epoch_end(self, outputs) -> None:
        results = self.online_val_map_metric.compute()
        for k in results.keys():
            if "per_class" in k:
                continue
            self.log(
                f"val_{k}_online",
                results[k].item(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        self.online_val_map_metric = MAP(class_metrics=True)
        self.online_val_map_metric.to(self.device)


    def test_step(self, batch, batch_idx) -> None:
        self.online_network.eval()
        _, predictions = self.online_network.common_step(batch)

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
            }
            for b in predictions
        ]

        self.test_map_metric.update(pred, gt)

    def on_test_epoch_end(self) -> None:
        results = self.test_map_metric.compute()
        for k in results.keys():
            self.log(
                f"test_{k}",
                results[k],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.online_network.configure_optimizers(max_steps=self.max_steps)
        return optimizer

    def reset_metric(self) -> None:
        self.online_val_map_metric = MAP(class_metrics=True)
        self.online_val_map_metric.to(self.device)

    def log_batch(self, batch, batch_idx, predictions = None, prefix = None) -> None:
        for idx in range(len(batch)):
            x = batch[idx]
            if not len(x['instances']) > 0:
                continue

            vocab = x['instances'].infos[0]["env_name"].replace("HSSD-HAB/", "")
            metadata = MetadataCatalog.get(vocab)

            if predictions is not None:
                pred_instances = deepcopy(predictions[idx]['instances'])
            else:
                pred_instances = None
                
            gt_instances = deepcopy(x['instances'])

            rgb = x["rgb_image"]
            img = plot_segmentation_gt_and_pred(
                rgb, 
                gt_instances=gt_instances, 
                pred_instances=pred_instances, 
                classes=metadata.thing_classes, 
                colors=metadata.thing_colors,
                scale=1.0
            )
            wandb_img = wandb.Image(np.array(img), caption=f"{prefix} - Batch {batch_idx} - Image {idx}")

            self.trainer.loggers[0].log_metrics(
                {
                    f"{prefix}-batch-{batch_idx}-img-{idx}": wandb_img, # type: ignore
                    "trainer/global_step": self.trainer.global_step,
                }
            )


class OnlineTeacherStudent(TeacherStudent):
    transform = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(use_teacher=True, *args, **kwargs)

    def training_step(self, batched_inputs, batch_idx) -> Tensor:
        batch = []
        if isinstance(batched_inputs[0], list):  # assuming this is not COCO
            pseudo_batch = batched_inputs[0]
        else:
            pseudo_batch = batched_inputs
        outs = [self.pseudo_labeler.forward(pseudo_batch)]
        pseudo_labels = self.pseudo_labeler.get_pseudo_labels(outs)

        # Apply augmentation only for student training_step
        for b, pseudo in zip(pseudo_batch, pseudo_labels):
            device = b['image'].device
            x, y = _transform_batch_with_logits(
                self.transform, b['image'].permute(1, 2, 0).cpu().numpy(), pseudo
            )
            x.to(device)
            b['image'] = x
            b['instances'] = y

        for i in batched_inputs:
            if isinstance(i, list):
                batch += i
            else:
                batch.append(i)

        if not self.freeze_teacher:
            self.pseudo_labeler.train()
        else:
            self.pseudo_labeler.eval()

        return super().training_step(batch, batch_idx)