from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.structures.instances import Instances

from common.utils.projection_utils import DepthCamera, ResolvedObjects


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    instances: Instances


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
            frame_id: resolved_objects.depth_raytracing(
                frame.depth, frame.pose, camera.hfov, camera.min_depth, camera.max_depth, min_visibility_fraction=min_visibility_fraction,
            )[:3]
            for frame_id, frame in self.frames.items()
        }

    def frames_showing(self, resolved_objects: ResolvedObjects, object_id: int, raytraced: dict) -> List[Tuple[int, Optional[int]]]:
        """(frame id, index of the instance detecting it or None) of every frame where the resolved object is detected or raytraced."""
        detected = resolved_objects.source_instances(object_id)
        return [
            (frame_id, detected.get(frame_id)) for frame_id in self.frames
            if frame_id in detected or (raytraced[frame_id][1] == object_id).any()
        ]
