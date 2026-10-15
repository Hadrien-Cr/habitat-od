"""Retrieval mosaic of a resolved map: for its largest objects, every remembered frame showing them."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.structures import Boxes, Instances
from PIL import Image

from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import make_mosaic, plot_segmentation, plot_segmentation_pred
from common.utils.projection_utils import ResolvedObjects


ROW_FRAMES = 5


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


def retrieval_mosaic(memory: ImageMemory, raytraced: dict, style: SegmentationStyle, object_ids: List[int], num_frames: int) -> Optional[Image.Image]:
    """One mosaic row per object of object_ids, of its num_frames largest views ImageMemory.frames_showing retrieves from raytraced (ImageMemory.raytrace_frames)."""
    return views_mosaic(memory, raytraced, style, {object_id: _largest_views(memory, raytraced, object_id, num_frames) for object_id in object_ids})


def objects_by_size(fused: ResolvedObjects) -> List[int]:
    """Ids of every resolved object, the one with the most points first."""
    object_ids, sizes = np.unique(fused.resolved_infos[:, 1], return_counts=True)
    return [int(object_id) for object_id in object_ids[np.argsort(-sizes)]]


def largest_objects(fused: ResolvedObjects, num_objects: int) -> List[int]:
    """Ids of the num_objects resolved objects with the most points, largest first."""
    return objects_by_size(fused)[:num_objects]


def views_mosaic(memory: ImageMemory, raytraced: dict, style: SegmentationStyle, object_frames: Dict[int, List[int]]) -> Optional[Image.Image]:
    """One mosaic row per object of object_frames, of its frames in order."""
    rows = [
        [_retrieval_tile(memory.get_frame(frame_id), frame_id, object_id, raytraced[frame_id], style) for frame_id in frame_ids]
        for object_id, frame_ids in object_frames.items()
    ]
    tiles = [tile for row in rows for tile in row]
    return make_mosaic(tiles, row_sizes=[len(row) for row in rows]) if tiles else None


def _largest_views(memory: ImageMemory, raytraced: dict, object_id: int, num_frames: int) -> List[int]:
    """Ids of the object's num_frames largest raytraced views, largest first."""
    return sorted(memory.frames_showing(object_id, raytraced), key=lambda frame_id: -_mask_area(raytraced[frame_id], object_id))[:num_frames]


def _mask_area(frame_raytraced: tuple, object_id: int) -> int:
    masks, ids, _ = frame_raytraced
    return int(masks[ids == object_id].sum())


def _retrieval_tile(frame: Frame, frame_id: int, object_id: int, frame_raytraced: tuple, style: SegmentationStyle) -> Tuple[str, np.ndarray]:
    """frame with the box and resolved class of the object's raytraced mask, the mask itself left undrawn."""
    masks, ids, classes = frame_raytraced
    instances = instances_from_masks(masks[ids == object_id], classes[ids == object_id], frame.depth.shape)
    instances.remove("gt_masks")
    return segmentation_tile(frame.rgb, instances, style, f"object{object_id} frame{frame_id:03d}")
