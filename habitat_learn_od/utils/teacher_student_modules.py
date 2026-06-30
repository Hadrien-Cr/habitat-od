from fnmatch import fnmatch
import logging
from copy import copy, deepcopy
import os
from sys import prefix
import albumentations as A
import habitat # type: ignore
import cv2

import numpy as np
import pytorch_lightning as pl
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog 
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

import wandb

from common.utils.plot_utils import plot_segmentation_gt, plot_segmentation_pred, plot_segmentation, Image

from habitat_learn_od.utils.train_helpers import mixup_batch
from habitat_learn_od.utils.two_stage_models import TwoStageModel
from habitat_learn_od.utils import two_stage_models

from habitat_learn_od.utils.pseudo_labeler import (
    PseudoLabeler,
    SemanticMapPseudoLabeler,
    SoftPseudoLabeler,
    VanillaPseudoLabeler,
)

log = logging.getLogger(__name__)
LOG_FREQ = 32

class TeacherStudent(pl.LightningModule):
    pseudo_labeler: PseudoLabeler
    online_network: TwoStageModel

    def __init__(
        self,
        detic_args,
        pseudo_labeler_method="vanilla",
        temperature=1,
        student_model=None,
        thr=0.7,
        freeze_teacher=True,
        use_teacher=False,
        batch_size=1,
        mixup=False,
        solution="ours",
        object_params=None,
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

        self.kwargs = kwargs
        self.batch_size = batch_size
        self.mixup = mixup
        self.use_teacher = use_teacher
        self.detic_args = detic_args
        self.vocab_name = object_params["env_name"].replace("HSSD-HAB/", "")

        print(f"Using pseudo-labeler method: {pseudo_labeler_method}")

        self.pseudo_labeler: PseudoLabeler = switch[pseudo_labeler_method](
            model=two_stage_models.TwoStageModel(detic_args, vocab_name=self.vocab_name, **kwargs),
            temperature=temperature,
            thr=thr,
            solution=solution,
            **kwargs
        )
        self.max_steps = None

        self.freeze_teacher = freeze_teacher

        self.online_val_map_metric = MAP(class_metrics=True)
        self.test_map_metric = MAP(class_metrics=True)
        self.loss_weights = kwargs["loss_weights"]
        self.reinit_online()


    def reinit_online(self) -> None:
        self.online_network = two_stage_models.TwoStageModel(self.detic_args, vocab_name=self.vocab_name, **self.kwargs)
        self.online_network.model.roi_heads.box_predictor.test_score_thresh = 0.5

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.online_network.configure_optimizers(max_steps=self.max_steps)
        return optimizer
    
    def training_step(self, batch, batch_idx) -> Tensor:
        results, losses, box_features, proposals = self.online_network.model_forward(batch)

        weighted_losses = {}

        for name, value in losses.items():
            weight = None

            if name in self.loss_weights:
                weight = self.loss_weights[name]
            else:
                for pattern, w in self.loss_weights.items():
                    if fnmatch(name, pattern):
                        weight = w
                        break

            if weight is None:
                raise ValueError(f"Loss weight for {name} not found in loss_weights dictionary. {self.loss_weights}")
            
            weighted_losses[name] = value * weight

        loss = sum(weighted_losses.values())

        for k in losses.keys():
            self.log(
                f"train_{k}",
                weighted_losses[k],
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

        if ((batch_idx) % (LOG_FREQ*self.batch_size) == 0):
            self.log_batch(batch, batch_idx, results, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self.online_validation_step(batch, batch_idx)

    def _val_map(self, batch, results) -> None:
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
            for b in results
        ]
        self.online_val_map_metric.update(pred, gt)

    def online_validation_step(self, batch, batch_idx) -> None:
        self.online_network.eval()
        with torch.no_grad():
            results = self.online_network.model_inference(batch)

        if ((batch_idx) % (LOG_FREQ*self.batch_size) == 0):
            self.log_batch(batch, batch_idx, results, prefix="val_online")

        self._val_map(batch, results)

    def on_validation_epoch_end(self,) -> None:
        results = deepcopy(self.online_val_map_metric.compute())
        
        metadata = MetadataCatalog.get(self.vocab_name)
        classes_ids = results["classes"].cpu().numpy().tolist()
        map_per_class = results["map_per_class"].cpu().numpy().tolist()
        
        for k in results.keys():
            if k.startswith("mar"):
                continue

            if results[k].numel() != 1 and k != "map_per_class":
                continue

            if k == "map_per_class":
                for class_id, map_value in zip(classes_ids, map_per_class):
                    if map_value == -1.0:
                        continue
                    class_name = metadata.thing_classes[class_id]
                    self.log(
                        f"CLASSMAP_val_{class_name}",
                        map_value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=self.batch_size,
                    )
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

    def reset_metric(self) -> None:
        self.online_val_map_metric = MAP(class_metrics=True)
        self.online_val_map_metric.to(self.device)

    def log_batch(self, batch, batch_idx, predictions = None, prefix = None) -> None:
        for idx in range(len(batch)):
            x = batch[idx]

            metadata = MetadataCatalog.get(self.vocab_name)

            if predictions is not None:
                pred_instances = deepcopy(predictions[idx]['instances'])
            else:
                pred_instances = None
                
            gt_instances = deepcopy(x['instances'])

            rgb = x['image'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            im = plot_segmentation(
                rgb, 
                gt_instances=gt_instances, 
                pred_instances=pred_instances, 
                classes=metadata.thing_classes, 
                colors=metadata.thing_colors,
                scale=1.0,
                title=f"P {len(pred_instances) if pred_instances is not None else 0}  GT {len(gt_instances) }"
            )
            img_name = x["image_path"].split("/")[-1].split(".")[0] + ".png"
            self.log_image(im, prefix=prefix, img_name=img_name,save_only = True)

            im = Image.fromarray(rgb)
            img_name = x["image_path"].split("/")[-1].split(".")[0] + ".png"
            self.log_image(im, prefix=prefix + "_rgb", img_name=img_name, save_only = True)

            
    def log_image(
        self,
        im: Image.Image,
        prefix: str,
        img_name: str = None,
        save_only: bool = False
    ) -> None:
        """Log an image to wandb."""
        if not os.path.exists("datadump"):
            os.makedirs("datadump"
    )
        os.makedirs(f"datadump/{prefix}", exist_ok=True)
        im.save(f"datadump/{prefix}/{img_name}")

        if not save_only:
            wandb_img = wandb.Image(np.array(im), caption=img_name)
            self.trainer.loggers[0].log_metrics(
                {
                    f"{prefix}-batch-{batch_idx}-img-{image_idx}": wandb_img, # type: ignore
                    "trainer/global_step": self.trainer.global_step,
                }
            )
            
# class OnlineTeacherStudent(TeacherStudent):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(use_teacher=True, *args, **kwargs)

#     def training_step(self, batched_inputs, batch_idx) -> Tensor:
#         batch = []
#         if isinstance(batched_inputs[0], list):  # assuming this is not COCO
#             pseudo_batch = batched_inputs[0]
#         else:
#             pseudo_batch = batched_inputs
#         outs = [self.pseudo_labeler.forward(pseudo_batch)]
#         pseudo_labels = self.pseudo_labeler.get_pseudo_labels(outs)

#         # Apply augmentation only for student training_step
#         for b, pseudo in zip(pseudo_batch, pseudo_labels):
#             device = b['image'].device
#             x, y = _transform_batch_with_logits(
#                 self.transform, b['image'].permute(1, 2, 0).cpu().numpy(), pseudo
#             )
#             x.to(device)
#             b['image'] = x
#             b['instances'] = y

#         for i in batched_inputs:
#             if isinstance(i, list):
#                 batch += i
#             else:
#                 batch.append(i)

#         return super().training_step(batch, batch_idx)