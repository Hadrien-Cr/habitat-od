import os
import sys
import numpy as np
import torch
from pathlib import Path
import torch
from torch.nn import functional as F

BASE_DIR = os.environ["BASE_DIR"]
DETIC_ROOT = os.path.join(BASE_DIR, "third_party/Detic")
sys.path.insert(0, DETIC_ROOT)
sys.path.insert(0, os.path.join(DETIC_ROOT, "third_party/CenterNet2"))
sys.path.insert(0, os.path.join(DETIC_ROOT, "third_party/Deformable-DETR"))

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.meta_arch.custom_rcnn import CustomRCNN
from detic.modeling.text.text_encoder import (  # noqa:E402
    build_text_encoder,
)

def reset_cls(model, class_embeddings, num_classes):
    model.roi_heads.num_classes = num_classes

    zs_weight = torch.cat(
        [class_embeddings, class_embeddings.new_zeros((class_embeddings.shape[0], 1))], 
        dim=1
    ).to(model.device) # D x (C + 1)
    
    cls_bias = torch.zeros((num_classes + 1), dtype=torch.float32, device=model.device)
    cls_bias[-1] = -1e6

    if model.roi_heads.box_predictor[0].cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)

    for k in range(len(model.roi_heads.box_predictor)):
        # model.roi_heads.box_predictor[k].num_classes = num_classes
        # model.roi_heads.box_predictor[k].use_sigmoid_ce = True
        # model.roi_heads.box_predictor[k].cls_score.norm_temperature = 50
        # model.roi_heads.box_predictor[k].cls_score.use_bias = True
        # model.roi_heads.box_predictor[k].cls_score.register_buffer('cls_bias', cls_bias)
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
        model.roi_heads.box_predictor[k].num_classes = num_classes

    for param in model.roi_heads.box_predictor[0].cls_score.parameters():
        param.requires_grad = False