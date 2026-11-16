"""COCO-format helpers shared by the dataset writers and the scene map evaluation: RLE-encoded masks and json files."""
import json
from pathlib import Path
from typing import List

import numpy as np
from pycocotools import mask as mask_util


def encoded_masks(masks) -> List[dict]:
    """RLE of each of the (N, H, W) masks, a tensor or an array."""
    return mask_util.encode(np.asfortranarray(np.asarray(masks).transpose(1, 2, 0).astype(np.uint8)))


def json_rle(rle: dict) -> dict:
    return dict(rle, counts=rle["counts"].decode("ascii"))


def decoded_mask(rle: dict) -> np.ndarray:
    return mask_util.decode(rle).astype(bool)


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
