"""Retrieval mosaic of a resolved map: for its largest objects, every remembered frame showing them."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.structures import Boxes, Instances
from PIL import Image

from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import make_mosaic, plot_segmentation, plot_segmentation_pred
from common.utils.projection_utils import ResolvedObjects


_ROW_FRAMES = 5


@dataclass(frozen=True)
class SegmentationStyle:
    classes: list
    colors: list


def instances_from_masks(masks: torch.Tensor, classes: torch.Tensor, image_size) -> Instances:
    boxes = []
    for mask in masks.numpy().astype("uint8"):
        x, y, w, h = cv2.boundingRect(mask)
        boxes.append([x, y, x + w, y + h])
    instances = Instances(image_size)
    instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
    instances.gt_classes = classes.long()
    instances.gt_masks = masks.bool()
    return instances


def segmentation_tile(rgb: np.ndarray, instances: Instances, style: SegmentationStyle, label: str) -> Tuple[str, np.ndarray]:
    """rgb with instances' predicted, or else ground-truth, segmentation drawn on it."""
    if instances.has("pred_boxes"):
        return label, np.array(plot_segmentation_pred(rgb.copy(), instances, style.classes, style.colors))
    return label, np.array(plot_segmentation(rgb=rgb.copy(), gt_instances=instances, pred_instances=None, classes=style.classes, colors=style.colors))


def retrieval_mosaic(fused: ResolvedObjects, memory: ImageMemory, raytraced: dict, style: SegmentationStyle, num_objects: int) -> Optional[Image.Image]:
    """One mosaic row per largest resolved object, of every frame ImageMemory.frames_showing retrieves it in from raytraced (ImageMemory.raytrace_frames)."""
    object_ids, sizes = np.unique(fused.resolved_infos[:, 1], return_counts=True)
    rows = [_retrieval_row(memory, raytraced, style, int(object_id)) for object_id in object_ids[np.argsort(-sizes)[:num_objects]]]
    tiles = [tile for row in rows for tile in row]
    return make_mosaic(tiles, row_sizes=[len(row) for row in rows]) if tiles else None


def _retrieval_row(memory: ImageMemory, raytraced: dict, style: SegmentationStyle, object_id: int) -> List[Tuple[str, np.ndarray]]:
    """The object's 10 largest raytraced views, largest first."""
    frame_ids = sorted(memory.frames_showing(object_id, raytraced), key=lambda frame_id: -_mask_area(raytraced[frame_id], object_id))
    return [_retrieval_tile(memory.get_frame(frame_id), frame_id, object_id, raytraced[frame_id], style) for frame_id in frame_ids[:_ROW_FRAMES]]


def _mask_area(frame_raytraced: tuple, object_id: int) -> int:
    masks, ids, _ = frame_raytraced
    return int(masks[ids == object_id].sum())


def _retrieval_tile(frame: Frame, frame_id: int, object_id: int, frame_raytraced: tuple, style: SegmentationStyle) -> Tuple[str, np.ndarray]:
    """frame with the box and resolved class of the object's raytraced mask, the mask itself left undrawn."""
    masks, ids, classes = frame_raytraced
    instances = instances_from_masks(masks[ids == object_id], classes[ids == object_id], frame.depth.shape)
    instances.remove("gt_masks")
    return segmentation_tile(frame.rgb, instances, style, f"object{object_id} frame{frame_id:03d}")
