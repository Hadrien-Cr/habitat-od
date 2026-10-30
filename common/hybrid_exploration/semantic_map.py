"""2D top-down goal-confidence tracking for single-target object search, layered on top of whichever
EstimatedMap (common/hybrid_exploration/exploration_map.py) the owning agent already builds from depth.
Modeled on home-robot's Categorical2DSemanticMapState, narrowed from a per-category channel stack to the
single running "is the target here" score this pipeline needs, since the target class changes every episode
and Detector.predict_target already scores a single caller-chosen embedding per frame, resetting Detic's own
classifier to it."""
from dataclasses import dataclass
from typing import Optional

import cv2
import habitat_sim
import numpy as np
import torch

from common.hybrid_exploration.exploration_map import EstimatedMap
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, nearest_resized, object_points


@dataclass(frozen=True)
class GoalScoreConfig:
    lift_image_size: int = 512
    score_decay: float = 0.9
    goal_radius_px: int = 3


def _masked_points(
    depth: np.ndarray, sensor_state: habitat_sim.SixDOFPose, mask: torch.Tensor, camera: DepthCamera, lift_image_size: int,
) -> np.ndarray:
    """World points of mask's in-range depth pixels, depth and mask resized to lift_image_size together (the
    resolution lifting runs at elsewhere, habitat_embodied_sl/main.py::observe_frame)."""
    lift_shape = (lift_image_size, lift_image_size)
    resized_depth = nearest_resized(np.asarray(depth).squeeze(), lift_shape).numpy()
    resized_mask = nearest_resized(mask[None, None].float(), lift_shape)[0, 0].bool()
    return object_points(resized_depth, sensor_pose_matrix(sensor_state), resized_mask, camera)


class GoalScoreMap:
    """A per-cell running max of the target's detection score over exploration_map's grid, and the world
    position of the best-scoring cell seen so far (the goal, once found_goal latches)."""

    def __init__(self, config: GoalScoreConfig, exploration_map: EstimatedMap):
        self.config = config
        self.exploration = exploration_map
        self.goal_score = np.zeros_like(exploration_map.obstacles, dtype=np.float32)
        self.best_goal_world: Optional[np.ndarray] = None
        self.best_goal_score = 0.0

    def decay(self) -> None:
        self.goal_score *= self.config.score_decay

    def goal_mask(self, goal_score_thresh: float) -> np.ndarray:
        """uint8 goal_score.shape mask: a small fixed disk around best_goal_world once a detection has ever
        cleared goal_score_thresh, so the navigation target stays stable instead of flickering with every
        step's decayed/re-detected goal_score (the target object doesn't move once actually found)."""
        mask = np.zeros_like(self.goal_score, dtype=np.uint8)
        if self.best_goal_world is None or self.best_goal_score < goal_score_thresh:
            return mask
        row, col = self.exploration.to_pixel(self.best_goal_world)
        cv2.circle(mask, (int(col), int(row)), self.config.goal_radius_px, 1, -1)
        return mask

    def update(self, depth: np.ndarray, sensor_state: habitat_sim.SixDOFPose, mask: torch.Tensor, score: float, camera: DepthCamera) -> None:
        """Back-projects mask's in-range depth pixels to world points, scatters score onto their top-down cells
        (kept as each cell's running max, since decay() already ages stale high scores down), and updates the
        best-scoring cell/position seen so far."""
        if score <= 0.0:
            return
        points = _masked_points(depth, sensor_state, mask, camera, self.config.lift_image_size)
        if len(points) == 0:
            return
        pixels = self._mark(points, score)
        if pixels is not None and score > self.best_goal_score:
            self.best_goal_score = score
            self.best_goal_world = points[pixels].mean(axis=0)

    def _mark(self, points: np.ndarray, score: float):
        """Scatters points onto goal_score's top-down cells, kept as each cell's running max, returning the
        in-bounds point mask, or None without any."""
        grid = self.goal_score
        pixels = self.exploration.to_pixels(points)
        in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < grid.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < grid.shape[1])
        if not np.any(in_bounds):
            return None
        rows, cols = pixels[in_bounds, 0], pixels[in_bounds, 1]
        grid[rows, cols] = np.maximum(grid[rows, cols], score)
        return in_bounds
