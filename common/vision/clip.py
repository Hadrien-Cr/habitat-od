import sys
import torch
from torch.nn import functional as F
from pathlib import Path

DETIC_ROOT = str(Path(__file__).parent / "../../third_party/Detic/")

sys.path.insert(
    0, DETIC_ROOT
)
sys.path.insert(
    0, str(Path(DETIC_ROOT) / "third_party/CenterNet2/")
)

from third_party.Detic.detic.modeling.text.text_encoder import (  # noqa:E402
    build_text_encoder,
)

def get_clip_embeddings(vocabulary: list[str], prompt: str = "a ") -> torch.Tensor:
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


def reset_cls_test(model, cls_path, num_classes):
    model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        print('Resetting zs_weight', cls_path)
        zs_weight = torch.tensor(
            np.load(cls_path), 
            dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], 
        dim=1) # D x (C + 1)
    if model.roi_heads.box_predictor[0].cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)
    for k in range(len(model.roi_heads.box_predictor)):
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight