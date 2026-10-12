from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from common.utils.projection_utils import DepthCamera, ResolvedObjects


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray


class ImageMemory:
    frames: Dict[int, Frame]

    def __init__(self):
        self.frames = {}

    def add_frame(self, frame: Frame) -> int:
        frame_id = len(self.frames)
        self.frames[frame_id] = frame
        return frame_id

    def __len__(self):
        return len(self.frames)

    def get_frame(self, frame_id: int) -> Frame:
        if frame_id not in self.frames:
            raise ValueError(f"Frame with id {frame_id} not found in memory.")
        return self.frames[frame_id]

    def raytrace_frames(self, resolved_objects: ResolvedObjects, camera: DepthCamera, min_visibility_fraction: float) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """{frame id: (masks, ids, classes)} of the resolved objects raytraced in each frame at least min_visibility_fraction visible."""
        return {
            frame_id: resolved_objects.visible_objects(
                frame.depth, frame.pose, camera.hfov, camera.min_depth, camera.max_depth, min_visibility_fraction=min_visibility_fraction,
            )
            for frame_id, frame in self.frames.items()
        }

    def object_views(self, resolved_objects: ResolvedObjects, camera: DepthCamera, min_visibility_fraction: float) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """{object id: {frame id: (rgb, raytraced mask)}} of every frame showing each resolved object at least min_visibility_fraction visible."""
        views = defaultdict(dict)
        for frame_id, (masks, ids, _) in self.raytrace_frames(resolved_objects, camera, min_visibility_fraction).items():
            for mask, object_id in zip(masks, ids.tolist()):
                views[object_id][frame_id] = (self.frames[frame_id].rgb, mask.numpy())
        return dict(views)

    def frames_showing(self, object_id: int, raytraced: dict) -> List[int]:
        """Ids of the frames raytraced (ImageMemory.raytrace_frames) to show the resolved object."""
        return [frame_id for frame_id in self.frames if (raytraced[frame_id][1] == object_id).any()]
