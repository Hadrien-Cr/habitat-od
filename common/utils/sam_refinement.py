"""Multi-view SAM refinement of a resolved map: each object is segmented by SAM in a few diverse views of it (retrieval.py's
retrieve_views), then its points carved, kept only where the views seeing them mostly agree they lie on its SAM mask."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from common.utils.image_memory import Frame, ImageMemory
from common.utils.projection_utils import DepthCamera, ResolvedObjects, world_to_pixels
from common.utils.retrieval import ViewSelection, retrieve_views
from common.utils.sam import SamMasker, mask_box

PointPrompt = Tuple[Tuple[int, int], int]


@dataclass(frozen=True)
class RefinementConfig:
    """A point votes in a view where it lies within depth_tolerance of the frame's depth (in iff on the SAM mask) or in front
    of it (out, as free space); it is kept with at least min_inside_fraction of its votes in, or without any vote."""

    selection: ViewSelection = field(default_factory=lambda: ViewSelection(k=5, gamma=0.25, alpha=2.0))
    min_visibility_fraction: float = 0.2
    depth_tolerance: float = 0.1
    min_inside_fraction: float = 0.5


def refined(resolved: ResolvedObjects, memory: ImageMemory, masker: SamMasker, camera: DepthCamera, config: RefinementConfig) -> ResolvedObjects:
    """resolved carved by the SAM masks of each object in its retrieve_views views."""
    raytraced = memory.raytrace_frames(resolved, camera, config.min_visibility_fraction)
    object_frames = retrieve_views(resolved, memory, raytraced, camera, config.selection)
    inside_votes, votes = np.zeros(len(resolved.resolved_points)), np.zeros(len(resolved.resolved_points))
    for frame_id, object_ids in _objects_by_frame(object_frames).items():
        frame = memory.get_frame(frame_id)
        for object_id, sam_mask in zip(object_ids, _sam_masks(masker, frame.rgb, raytraced[frame_id], object_ids)):
            indices = np.flatnonzero(resolved.resolved_infos[:, 1] == object_id)
            inside, cast = _view_votes(resolved.resolved_points[indices], frame, sam_mask, camera, config.depth_tolerance)
            inside_votes[indices] += inside
            votes[indices] += cast
    return resolved.subset(inside_votes >= config.min_inside_fraction * votes)


def _objects_by_frame(object_frames: Dict[int, List[int]]) -> Dict[int, List[int]]:
    frame_objects = defaultdict(list)
    for object_id, frame_ids in object_frames.items():
        for frame_id in frame_ids:
            frame_objects[frame_id].append(object_id)
    return frame_objects


def _sam_masks(masker: SamMasker, rgb: np.ndarray, frame_raytraced: tuple, object_ids: List[int]) -> np.ndarray:
    """SAM mask of each object, prompted with its raytraced mask's box, an in point inside that mask and an out point
    inside every other raytraced object's mask within the box."""
    masks, ids, _ = frame_raytraced
    interiors = {int(object_id): _interior_point(mask.numpy()) for mask, object_id in zip(masks, ids)}
    boxes = [mask_box(masks[ids == object_id][0].numpy()) for object_id in object_ids]
    coords, labels = _padded([_point_prompts(object_id, box, interiors) for object_id, box in zip(object_ids, boxes)])
    return masker.prompted_masks(rgb, torch.tensor(boxes), coords, labels).numpy()


def _interior_point(mask: np.ndarray) -> Tuple[int, int]:
    """(column, row) of the mask pixel farthest from its border."""
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    row, column = np.unravel_index(distances.argmax(), distances.shape)
    return int(column), int(row)


def _point_prompts(object_id: int, box: List[int], interiors: Dict[int, Tuple[int, int]]) -> List[PointPrompt]:
    x0, y0, x1, y1 = box
    others = [point for other_id, point in interiors.items() if other_id != object_id and x0 <= point[0] <= x1 and y0 <= point[1] <= y1]
    return [(interiors[object_id], 1)] + [(point, 0) for point in others]


def _padded(prompts: List[List[PointPrompt]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, P, 2) coordinates and (N, P) labels of the prompts, padded to the longest with SAM's ignored -1 label."""
    size = max(len(prompt) for prompt in prompts)
    coords, labels = torch.zeros((len(prompts), size, 2)), torch.full((len(prompts), size), -1)
    for index, prompt in enumerate(prompts):
        coords[index, :len(prompt)] = torch.tensor([point for point, _ in prompt], dtype=torch.float32)
        labels[index, :len(prompt)] = torch.tensor([label for _, label in prompt])
    return coords, labels


def _view_votes(points: np.ndarray, frame: Frame, sam_mask: np.ndarray, camera: DepthCamera, depth_tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """(in votes, votes) of points in the view: those on its depth surface vote in iff on sam_mask, those in front of it
    vote out, occluded ones and those outside the image don't vote."""
    depth = np.asarray(frame.depth).squeeze()
    pixels, point_depths = world_to_pixels(points, frame.pose, depth.shape, camera.hfov)
    in_image = (point_depths > 0) & np.all((pixels >= 0) & (pixels < [depth.shape[1], depth.shape[0]]), axis=1)
    columns, rows = np.clip(pixels[:, 0], 0, depth.shape[1] - 1), np.clip(pixels[:, 1], 0, depth.shape[0] - 1)
    depth_gap = depth[rows, columns] - point_depths
    on_surface = in_image & (np.abs(depth_gap) <= depth_tolerance)
    return on_surface & sam_mask[rows, columns], on_surface | (in_image & (depth_gap > depth_tolerance))
