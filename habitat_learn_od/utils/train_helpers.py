import copy
import logging
import multiprocessing
import os
import pickle
from copy import copy, deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import default_collate

logging.basicConfig(level=logging.INFO)


def get_training_params(cfg):
    logger = [
        _get_wandb_logger(
            project_name="look_around_and_learn",
            exp_name=cfg.training.exp_base_name + "/" + cfg.exp_name,
        ),
    ]
    dataset_path = os.getcwd()
    checkpoint_dir = os.path.join(dataset_path, "checkpoints")
    log_profiler = os.path.join(dataset_path, "profile.txt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_map_50_online",
        mode='max',
        save_last=True,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        every_n_epochs=1,
    )

    gpus = cfg["gpus"]

    if "plugins" in cfg:
        plugins = cfg['plugins']
    else:
        plugins = None

    trainer_configuration = {
        "multiple_trainloader_mode": "min_size",
        "default_root_dir": checkpoint_dir,
        "accelerator": "gpu",
        "devices": gpus,
        "max_epochs": cfg["epochs"],
        "callbacks": [ckpt_cb],
        "enable_checkpointing": True,
        "logger": logger,
        "plugins": plugins,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 1,
    }

    if "debug" in cfg and cfg['debug']:
        torch.autograd.set_detect_anomaly(True)
        trainer_configuration["overfit_batches"] = 50
        trainer_configuration["log_gpu_memory"] = True

    if "early_stopping" in cfg and cfg['early_stopping'] > 0:
        early_stop_callback = EarlyStopping(
            monitor="train_loss_cls_epoch",
            min_delta=0.001,
            patience=cfg["early_stopping"],
            verbose=False,
            mode="min",
        )
        trainer_configuration["callbacks"].append(early_stop_callback)

    return trainer_configuration


def _get_wandb_logger(exp_name: str, project_name: str):
    logger = WandbLogger(
        name=exp_name,
        project=project_name,
    )
    return logger


def collate_fn_helper(batch):
    if isinstance(batch, list):
        return list_helper_collate(batch)
    elif isinstance(batch, dict):
        return dict_helper_collate(batch)
    else:
        return default_collate(batch)


def list_helper_collate(batch):
    return list(chain(*[[elem for elem in elems_list] for elems_list in batch]))


def dict_helper_collate(batch):

    elem = batch[0]
    return [{key: d[key] for key in elem} for d in batch]


def mixup_batch(batch):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    device = batch[0]['image'].device
    indexes = np.random.permutation(len(batch))
    alpha = beta = 1.5
    r = np.random.beta(alpha, beta)  # mixup ratio, alpha=beta=32.0
    for b1, idx in zip(batch, indexes):
        b2_image = deepcopy(batch[idx]['image'])
        b2_instances = deepcopy(batch[idx]['instances'])
        b1['image'] = (b1['image'] * r + b2_image * (1 - r)).int()
        y = Instances(b1['image'].shape)
        y.gt_classes = torch.cat([b1['instances'].gt_classes, b2_instances.gt_classes])

        y.gt_logits = torch.cat([b1['instances'].gt_logits, b2_instances.gt_logits])

        y.infos = b1['instances'].infos + b2_instances.infos

        if hasattr(b1['instances'], "gt_masks"):
            y.gt_masks = BitMasks(
                torch.cat(
                    [b1['instances'].gt_masks.tensor, b2_instances.gt_masks.tensor]
                )
            )

        y.gt_boxes = Boxes(
            torch.cat([b1['instances'].gt_boxes.tensor, b2_instances.gt_boxes.tensor])
        )
        b1['instances'] = y
        del b2_image
        del b2_instances

