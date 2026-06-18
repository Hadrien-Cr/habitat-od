import sys
import numpy as np
import torch
from pathlib import Path
import torch
from torch.nn import functional as F

DETIC_ROOT = "third_party/Detic"
sys.path.insert(0, "third_party/Detic/third_party/CenterNet2")
sys.path.insert(0, "third_party/Detic/third_party/Deformable-DETR")
from centernet.config import add_centernet_config
from third_party.Detic.detic.config import add_detic_config
from third_party.Detic.detic.modeling.meta_arch.custom_rcnn import CustomRCNN


def reset_cls(model, cls_path, num_classes, frozen=True):
    model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        zs_weight = torch.tensor(
            np.load(cls_path), 
            dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path

    assert zs_weight.shape[1] == num_classes, f"Expected {num_classes} classes, got {zs_weight.shape[1]}"
    if model.roi_heads.box_predictor[0].cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)

    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], 
        dim=1) # D x (C + 1)
    cls_bias = torch.zeros((num_classes + 1), dtype=torch.float32)
    cls_bias[-1] = -1e4
    zs_weight = zs_weight.to(model.device)

    for k in range(len(model.roi_heads.box_predictor)):
        model.roi_heads.box_predictor[k].num_classes = num_classes
        model.roi_heads.box_predictor[k].use_sigmoid_ce = False
        model.roi_heads.box_predictor[k].cls_score.norm_temperature = 20
        model.roi_heads.box_predictor[k].cls_score.use_bias = True
        model.roi_heads.box_predictor[k].cls_score.register_buffer('cls_bias', cls_bias)
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
    if frozen:
        for param in model.roi_heads.box_predictor[0].cls_score.parameters():
            param.requires_grad = False
    else:
        for param in model.roi_heads.box_predictor[0].cls_score.parameters():
            param.requires_grad = True