"""Retrieval of a resolved map: for each object, k diverse, non-redundant remembered views of it, greedily covering its
resolved points from as many yaws as possible, optionally favouring well-visible views."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.structures import Boxes, Instances
from PIL import Image

from common.env_utils.visibility_utils import mesh_visibility_fraction
from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import make_mosaic, plot_segmentation, plot_segmentation_pred
from common.utils.projection_utils import DepthCamera, ResolvedObjects, yaw_bins



@dataclass(frozen=True)
class ViewSelection:
    """Greedy coverage of an object's (resolved point, yaw bin) elements: a view gains its visibility ** alpha times the
    sum over its elements of (1 - gamma) * gamma ** (times already covered by the chosen views)."""

    k: int = 10
    gamma: float = 0.5
    alpha: float = 1.0
    num_yaw_bins: int = 8


@dataclass(frozen=True)
class _RaytracedMap:
    fused: ResolvedObjects
    memory: ImageMemory
    raytraced: dict
    camera: DepthCamera


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


def retrieve_views(fused: ResolvedObjects, memory: ImageMemory, raytraced: dict, camera: DepthCamera, selection: ViewSelection) -> Dict[int, List[int]]:
    """{object id: ids of the frames selection picks among those raytraced (ImageMemory.raytrace_frames) to show it}
    of every resolved object, largest first."""
    raytraced_map = _RaytracedMap(fused, memory, raytraced, camera)
    return {object_id: _select_views(raytraced_map, object_id, selection) for object_id in objects_by_size(fused)}


def greedy_coverage(elements: List[np.ndarray], weights: np.ndarray, selection: ViewSelection) -> List[int]:
    """Indices of up to selection.k views, each in turn the one whose weight times its elements'
    (1 - gamma) * gamma ** (times already covered) is largest, until no view gains anything."""
    counts = np.zeros(max((int(view.max()) + 1 for view in elements if len(view)), default=0))
    chosen = []
    for _ in range(min(selection.k, len(elements))):
        gains = np.array([weight * ((1 - selection.gamma) * selection.gamma ** counts[view]).sum() for weight, view in zip(weights, elements)])
        gains[chosen] = -np.inf
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        counts[elements[best]] += 1
    return chosen


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
        [_object_frame_tile(memory, raytraced, style, object_id, frame_id) for frame_id in frame_ids]
        for object_id, frame_ids in object_frames.items()
    ]
    tiles = [tile for row in rows for tile in row]
    return make_mosaic(tiles, row_sizes=[len(row) for row in rows]) if tiles else None


def first_views_mosaic(memory: ImageMemory, raytraced: dict, style: SegmentationStyle, object_frames: Dict[int, List[int]]) -> Optional[Image.Image]:
    """A single mosaic row of each object's first frame of object_frames, views_mosaic's first column."""
    tiles = [
        _object_frame_tile(memory, raytraced, style, object_id, frame_ids[0])
        for object_id, frame_ids in object_frames.items() if frame_ids
    ]
    return make_mosaic(tiles, N_cols=len(tiles)) if tiles else None


def _object_frame_tile(memory: ImageMemory, raytraced: dict, style: SegmentationStyle, object_id: int, frame_id: int) -> Tuple[str, np.ndarray]:
    return _retrieval_tile(memory.get_frame(frame_id), frame_id, object_id, raytraced[frame_id], style)


def _select_views(raytraced_map: _RaytracedMap, object_id: int, selection: ViewSelection) -> List[int]:
    frame_ids = raytraced_map.memory.frames_showing(object_id, raytraced_map.raytraced)
    visibility = np.array([_visibility(raytraced_map, frame_id, object_id) for frame_id in frame_ids])
    elements = [_covered_elements(raytraced_map, frame_id, object_id, selection.num_yaw_bins) for frame_id in frame_ids]
    return [frame_ids[index] for index in greedy_coverage(elements, visibility ** selection.alpha, selection)]


def _object_mask(raytraced_map: _RaytracedMap, frame_id: int, object_id: int) -> torch.Tensor:
    masks, ids, _ = raytraced_map.raytraced[frame_id]
    return masks[ids == object_id][0]


def _visibility(raytraced_map: _RaytracedMap, frame_id: int, object_id: int) -> float:
    """mesh_visibility_fraction of the object's raytraced mask, capped at 1 as its box-over-face ratio can exceed it."""
    dimensions, depth = raytraced_map.fused.object_dimensions(object_id), np.asarray(raytraced_map.memory.get_frame(frame_id).depth)
    return min(mesh_visibility_fraction(dimensions, _object_mask(raytraced_map, frame_id, object_id).numpy(), depth, raytraced_map.camera.hfov), 1.0)


def _covered_elements(raytraced_map: _RaytracedMap, frame_id: int, object_id: int, num_yaw_bins: int) -> np.ndarray:
    """Element ids, point index * num_yaw_bins + yaw bin, of the object's resolved points its raytraced mask hits in the frame."""
    frame, fused = raytraced_map.memory.get_frame(frame_id), raytraced_map.fused
    indices = fused.raytraced_point_indices(frame.depth, frame.pose, _object_mask(raytraced_map, frame_id, object_id), raytraced_map.camera, object_id)
    return indices * num_yaw_bins + yaw_bins(fused.resolved_points[indices], frame.pose[:3, 3], num_yaw_bins)


def _retrieval_tile(frame: Frame, frame_id: int, object_id: int, frame_raytraced: tuple, style: SegmentationStyle) -> Tuple[str, np.ndarray]:
    """frame with the box and resolved class of the object's raytraced mask, the mask itself left undrawn."""
    masks, ids, classes = frame_raytraced
    instances = instances_from_masks(masks[ids == object_id], classes[ids == object_id], frame.depth.shape)
    instances.remove("gt_masks")
    return segmentation_tile(frame.rgb, instances, style, f"object{object_id} frame{frame_id:03d}")
