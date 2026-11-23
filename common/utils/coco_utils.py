"""COCO-format helpers shared by the dataset writers and the scene map evaluation: RLE-encoded masks and json files."""
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from pycocotools import mask as mask_util


def encoded_masks(masks) -> List[dict]:
    """RLE of each of the (N, H, W) masks, a tensor or an array."""
    return mask_util.encode(np.asfortranarray(np.asarray(masks).transpose(1, 2, 0).astype(np.uint8)))


def json_rle(rle: dict) -> dict:
    return dict(rle, counts=rle["counts"].decode("ascii"))


def rle_from_json(rle: dict) -> dict:
    """json_rle's inverse."""
    return dict(rle, counts=rle["counts"].encode("ascii"))


def decoded_mask(rle: dict) -> np.ndarray:
    return mask_util.decode(rle).astype(bool)


def stacked_masks(rles: List[dict], image_size: tuple) -> torch.Tensor:
    """(N, H, W) boolean masks of the RLEs, (0, H, W) for none."""
    if not rles:
        return torch.zeros((0, *image_size), dtype=torch.bool)
    return torch.from_numpy(decoded_mask(rles)).permute(2, 0, 1)


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
