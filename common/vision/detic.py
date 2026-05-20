import os
from omegaconf import OmegaConf, DictConfig

import sys
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import torch
from torch.nn import functional as F


from detectron2.data import transforms as T
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor

sys.path.insert(0, "third_party/Detic/third_party/CenterNet2")
from centernet.config import add_centernet_config # type: ignore

sys.path.insert(0, "third_party/Detic/third_party/Deformable-DETR")
from third_party.Detic.detic.config import add_detic_config

from common.vision.clip import get_clip_embeddings, save_clip_embeddings, load_clip_embeddings
from common.utils.plot_utils import make_mosaic

DETIC_ROOT = "third_party/Detic"

def setup_cfg(detic_config: DictConfig) -> CfgNode:
    config_file = str(Path(DETIC_ROOT).resolve().parent / detic_config.config_file)
    weights = str(Path(DETIC_ROOT).resolve().parent / detic_config.weights)

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = detic_config.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detic_config.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        detic_config.confidence_threshold
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


def build_detic_model(detic_config: CfgNode, vocab: list[str], vocab_name: str) -> torch.nn.Module:
    cfg = setup_cfg(detic_config) # type: ignore
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    path_clip_embeddings = Path(f"datasets/metadata/{vocab_name}.npy")

    if not os.path.exists(path_clip_embeddings):
        classifier = get_clip_embeddings(vocab)
        save_clip_embeddings(classifier, path_clip_embeddings)
    else:
        classifier = load_clip_embeddings(path_clip_embeddings)

    reset_cls(model, classifier, len(vocab))
    model.num_classes = len(vocab)
    return model


def build_detic_predictor(detic_config: CfgNode, vocab: list[str], vocab_name: str) -> DefaultPredictor:
    cfg = setup_cfg(detic_config) # type: ignore
    predictor = DefaultPredictor(cfg)

    path_clip_embeddings = Path(f"datasets/metadata/{vocab_name}.npy")

    if not os.path.exists(path_clip_embeddings):
        classifier = get_clip_embeddings(vocab)
        save_clip_embeddings(classifier, path_clip_embeddings)
    else:
        classifier = load_clip_embeddings(path_clip_embeddings)

    reset_cls(predictor.model, classifier, len(vocab))
    predictor.model.num_classes = len(vocab)
    return predictor


def reset_cls(model, cls_path, num_classes, frozen=True):
    model.roi_heads.num_classes = num_classes
    
    if type(cls_path) == str:
        zs_weight = torch.tensor(
            np.load(cls_path), 
            dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path

    assert zs_weight.shape[1] == num_classes, f"Expected {num_classes} classes, got {zs_weight.shape[1]}"

    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], 
        dim=1) # D x (C + 1)
    if model.roi_heads.box_predictor[0].cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)

    for k in range(len(model.roi_heads.box_predictor)):
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
        model.roi_heads.box_predictor[k].num_classes = num_classes

    if frozen:
        for param in model.roi_heads.box_predictor[0].cls_score.parameters():
            param.requires_grad = False
    else:
        for param in model.roi_heads.box_predictor[0].cls_score.parameters():
            param.requires_grad = True