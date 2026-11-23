"""Frames remembered by id (Dict[int, Frame]) and the resolved objects raytraced in them."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from common.utils.projection_utils import DepthCamera, ResolvedObjects, nearest_resized


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray


def raytrace_frames(resolved_objects: ResolvedObjects, frames: Dict[int, Frame], camera: DepthCamera, min_mesh_visibility_fraction: float) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """{frame id: (masks, ids, classes)} of the resolved objects raytraced in each frame at least min_mesh_visibility_fraction visible."""
    return {
        frame_id: raytraced[:3] for frame_id, raytraced in raytrace_frame_points(resolved_objects, frames, camera, min_mesh_visibility_fraction).items()
    }


def raytrace_frame_points(
    resolved_objects: ResolvedObjects, frames: Dict[int, Frame], camera: DepthCamera, min_mesh_visibility_fraction: float,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, np.ndarray]]]:
    """{frame id: (masks, ids, classes, {id: hit point indices})} of raytrace_frames' objects (ResolvedObjects.visible_object_points)."""
    return {
        frame_id: resolved_objects.visible_object_points(
            frame.depth, frame.pose, camera.hfov, camera.min_depth, camera.max_depth, min_mesh_visibility_fraction=min_mesh_visibility_fraction,
        )
        for frame_id, frame in tqdm(frames.items(), desc="raytracing", ncols=80)
    }


def object_views(resolved_objects: ResolvedObjects, frames: Dict[int, Frame], camera: DepthCamera, min_mesh_visibility_fraction: float) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """{object id: {frame id: (rgb, raytraced mask resized to rgb)}} of every frame showing each resolved object at least
    min_mesh_visibility_fraction visible."""
    views = defaultdict(dict)
    for frame_id, (masks, ids, _) in raytrace_frames(resolved_objects, frames, camera, min_mesh_visibility_fraction).items():
        rgb = frames[frame_id].rgb
        for mask, object_id in zip(masks, ids.tolist()):
            views[object_id][frame_id] = (rgb, nearest_resized(mask, rgb.shape[:2]).numpy())
    return dict(views)


def frames_showing(raytraced: dict, object_id: int) -> List[int]:
    """Ids of the frames raytraced (raytrace_frames) to show the resolved object."""
    return [frame_id for frame_id, (_, ids, _) in raytraced.items() if (ids == object_id).any()]
