"""Top-down map of the agent's own depth-observed world: a fixed grid centered on the episode start, built
incrementally from depth (no ground-truth navmesh). Cells start free (unknown is traversable until a depth
point marks it an obstacle, matching facebookresearch/home-robot's DiscretePlanner convention) and
accumulate forever; `explored` tracks what's actually been observed, for frontier detection and for bounding
the window FMMPlanner plans in (planning_window) -- it never gates which cells within it are traversable. Also carries goal_score, a per-cell running max of a single-target
search's own detection score (SearchAgent's "is the target here" belief, modeled on home-robot's
Categorical2DSemanticMapState, narrowed to the one caller-chosen target class Detector.predict_target scores
per frame)."""
from dataclasses import dataclass
from typing import Tuple

import cv2
import habitat_sim
import numpy as np
import torch

from common.hybrid_exploration.frontiers import detect_frontier_waypoints, reveal_fog_of_war, without_small_frontiers
from common.utils.concept_graph import ConceptGraphConfig
from common.utils.projection_utils import (
    DepthCamera, downsample_points, foreground_points, main_cluster_mask, nearest_resized, object_points, project_batch_masks,
)
from common.utils.pose_utils import sensor_pose_matrix


@dataclass(frozen=True)
class MapConfig:
    fov_deg: float
    resolution_m: float = 0.1
    map_size_m: float = 48.0
    visibility_dist_m: float = 3.0
    area_thresh_m: float = 3.0
    min_frontier_size_m: float = 0.5  # shorter frontier fragments are dropped (without_small_frontiers)
    missing_floor_opening_px: int = 5  # depth-image pixels: no-hit regions thinner than this (e.g. 1-px cracks along a
    # wall/floor mesh seam, which line up into a wall-long top-down cluster) are dropped before floor projection
    obstacle_height_m: float = 1.4  # above the agent's own 0.88 m height (common/config/hssd-hab/default.yaml) so a
    # solid rising surface (e.g. a staircase's treads/soffit) stays marked past the agent's head height, short of
    # a typical door lintel (~2 m) wrongly becoming an obstacle too
    max_climb_m: float = 0.2  # matches habitat-sim's NavMeshSettings.agent_max_climb: lower steps/sills are traversable
    min_frontier_dist_m: float = 0.5  # frontier cells this close to the agent are excluded -- reveal_fog_of_war's
    # FOV cone has its apex at the agent's own position, which (before it has moved anywhere) is itself explored
    # but surrounded by unexplored space on most sides, so it would otherwise qualify as a "frontier" and make
    # DiscretePlanner's closest-goal collapse to the agent's current position (distance 0, no real target)
    lift_image_size: int = 512  # goal-mask depth/mask resize before mark_goal's own back-projection
    score_decay: float = 0.9  # mark_goal's running per-cell score, aged by this factor each step within in_view
    planning_margin_m: float = 2.0  # planning_window's growth past explored/goal cells, for paths skirting just outside
    # them -- FMM's cost grows with the window's area, the whole map's being mostly unknown space far from any goal
    # concept-graphs' own per-detection denoising (ConceptGraphConfig), reused by mark_goal to drop a mask's
    # floor/background bleed-through before it reaches goal_score: voxel-downsample, then main_cluster_mask's
    # largest DBSCAN cluster (detection_dbscan_eps/min_points) plus any smaller one attached within dbscan_eps,
    # up to detection_max_attached_fraction of it. Defaults mirror ConceptGraphConfig's own, so the two stay in sync.
    downsample_voxel_size: float = ConceptGraphConfig.downsample_voxel_size
    detection_dbscan_eps: float = ConceptGraphConfig.detection_dbscan_eps
    detection_dbscan_min_points: int = ConceptGraphConfig.detection_dbscan_min_points
    dbscan_eps: float = ConceptGraphConfig.dbscan_eps
    detection_max_attached_fraction: float = ConceptGraphConfig.detection_max_attached_fraction

# white unexplored, light gray explored, dark gray obstacles (habitat_od_search/evaluator.py's old
# _obstacle_map scheme) -- BGR, matching common/utils/plot_utils.py's own convention
_BELIEF_IMAGE_UNEXPLORED = (255, 255, 255)
_BELIEF_IMAGE_EXPLORED = (225, 225, 225)
_BELIEF_IMAGE_OBSTACLE = (90, 90, 90)


def _denoised_points(points: np.ndarray, camera_center: np.ndarray, floor_height: float, config: MapConfig) -> np.ndarray:
    """points without the floor or the wall behind them, then concept-graphs' own per-detection denoising
    (main_cluster_mask on the voxel-downsampled rest) -- the same cleanup test_denoise.py exercises, dropping a
    porous mask's floor/background bleed-through before it ever reaches goal_score."""
    foreground = foreground_points(points, floor_height, camera_center)
    if len(foreground) == 0:
        return foreground
    downsampled = downsample_points(foreground, config.downsample_voxel_size)
    kept = main_cluster_mask(
        downsampled, config.detection_dbscan_eps, config.detection_dbscan_min_points, config.dbscan_eps, config.detection_max_attached_fraction,
    )
    return downsampled[kept]


def _goal_mask_points(
    depth: np.ndarray, sensor_state: habitat_sim.SixDOFPose, mask: torch.Tensor, camera: DepthCamera, floor_height: float, config: MapConfig,
) -> np.ndarray:
    """World points of mask's in-range depth pixels, depth and mask resized to lift_image_size together (the
    resolution lifting runs at elsewhere, habitat_embodied_sl/collect.py::observe_frame), denoised."""
    lift_shape = (config.lift_image_size, config.lift_image_size)
    resized_depth = nearest_resized(np.asarray(depth).squeeze(), lift_shape).numpy()
    resized_mask = nearest_resized(mask[None, None].float(), lift_shape)[0, 0].bool()
    pose = sensor_pose_matrix(sensor_state)
    points = object_points(resized_depth, pose, resized_mask, camera)
    return _denoised_points(points, pose[:3, 3], floor_height, config)


class EstimatedMap:
    def __init__(self, config: MapConfig, position: np.ndarray):
        self.config = config
        self.meters_per_pixel = config.resolution_m
        self.map_pixels_size = self.pixels(config.map_size_m)
        self.origin_world = position - np.array([config.map_size_m / 2, 0.0, config.map_size_m / 2])
        self.obstacles = np.zeros((self.map_pixels_size, self.map_pixels_size), dtype=np.uint8)
        self.explored = np.zeros((self.map_pixels_size, self.map_pixels_size), dtype=np.uint8)
        self.in_view = np.zeros((self.map_pixels_size, self.map_pixels_size), dtype=np.uint8)
        self.frontier_pixels = np.empty((0, 2))
        self._frontier_obstacles = self.obstacles.copy()
        self.goal_score = np.zeros((self.map_pixels_size, self.map_pixels_size), dtype=np.float32)

    def pixels(self, meters: float) -> int:
        return int(meters / self.meters_per_pixel)

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        """(N, 2) (row, col) of (N, 3) world positions."""
        offsets = (positions[:, [2, 0]] - self.origin_world[[2, 0]]) / self.meters_per_pixel
        return offsets.astype(int)

    def to_pixel(self, position: np.ndarray) -> np.ndarray:
        return self.to_pixels(np.asarray(position)[None])[0]

    def to_world_xz(self, pixels: np.ndarray) -> np.ndarray:
        """World (x, z) of (row, col) pixels' own cell centers."""
        offsets = (pixels.astype(float) + 0.5) * self.meters_per_pixel
        return offsets[:, ::-1] + self.origin_world[[0, 2]]

    def to_world(self, pixels: np.ndarray, height: float) -> np.ndarray:
        """World points of (row, col) pixels, at height."""
        xz = self.to_world_xz(pixels)
        return np.stack([xz[:, 0], np.full(len(xz), height), xz[:, 1]], axis=1)

    def free(self) -> np.ndarray:
        return 1 - self.obstacles

    def frontiers(self, height: float) -> np.ndarray:
        return self.to_world(self.frontier_pixels, height) if len(self.frontier_pixels) else np.empty((0, 3))

    def reveal(self, position: np.ndarray, heading: float) -> None:
        """Sets in_view to the field of view at (position, heading), not occluded by known obstacles, adds it
        to explored, and recomputes frontiers if explored or obstacles changed since the last recomputation --
        a frontier placed against a sparsely-observed wall can otherwise go stale and keep targeting a cell
        later depth observations resolve into solid obstacle, without explored itself ever growing again."""
        self.in_view = reveal_fog_of_war(
            self.free(), np.zeros_like(self.explored), self.to_pixel(position), heading, self.config.fov_deg, self.pixels(self.config.visibility_dist_m),
        )
        previous_explored = self.explored
        self.explored = previous_explored | self.in_view
        
        if not np.array_equal(previous_explored, self.explored) or not np.array_equal(self._frontier_obstacles, self.obstacles):
            frontier_pixels = detect_frontier_waypoints(
                self.free(), self.explored, self.pixels(self.config.area_thresh_m ** 2), self.obstacles,
                self.to_pixel(position), self.pixels(self.config.min_frontier_dist_m),
            )
            self.frontier_pixels = without_small_frontiers(frontier_pixels, self.obstacles.shape, self.pixels(self.config.min_frontier_size_m))
            self._frontier_obstacles = self.obstacles.copy()

    def observe_obstacles(self, depth: np.ndarray, sensor_state, camera: DepthCamera, floor_height: float) -> None:
        """Marks obstacles at every in-range depth point above floor_height + max_climb_m (lower ones -- floor,
        door sills, small bumps -- the agent steps over) and at most floor_height + obstacle_height_m (points
        above that -- ceiling, door lintels, tall shelving -- don't block the agent either), plus missing_floor_points."""
        pose = sensor_pose_matrix(sensor_state)
        points = object_points(depth, pose, np.ones(depth.shape[-2:], dtype=bool), camera)
        heights = points[:, 1] - floor_height
        self._mark(self.obstacles, points[(heights > self.config.max_climb_m) & (heights <= self.config.obstacle_height_m)], fill=1)
        self._mark(self.obstacles, self.missing_floor_points(depth, pose, camera, floor_height), fill=1)

    def missing_floor_points(self, depth: np.ndarray, pose: np.ndarray, camera: DepthCamera, floor_height: float) -> np.ndarray:
        """Floor-plane crossings, within visibility_dist_m, of the downward rays whose depth hit no geometry (clipped to
        min_depth) -- e.g. the void past a terrace edge, which would otherwise stay free and keep yielding frontiers.
        The no-hit mask is first morphologically opened by missing_floor_opening_px: a real void is a solid image
        region, while mesh holes and seam cracks are isolated pixels or 1-px lines."""
        no_hit = (np.asarray(depth).squeeze() <= camera.min_depth).astype(np.uint8)
        kernel = np.ones((self.config.missing_floor_opening_px, self.config.missing_floor_opening_px), np.uint8)
        missing = torch.as_tensor(cv2.morphologyEx(no_hit, cv2.MORPH_OPEN, kernel).astype(bool))
        unit_depth_points = project_batch_masks(torch.ones(missing.shape), pose, missing, hfov=camera.hfov, min_depth=0.0, max_depth=2.0).numpy()
        camera_center = pose[:3, 3]
        rays = unit_depth_points - camera_center
        rays = rays[rays[:, 1] < 0]
        crossings = camera_center + rays * ((floor_height - camera_center[1]) / rays[:, 1])[:, None]
        return crossings[np.linalg.norm(crossings[:, [0, 2]] - camera_center[[0, 2]], axis=1) <= self.config.visibility_dist_m]

    def _mark(self, grid: np.ndarray, points: np.ndarray, fill: int) -> None:
        if len(points) == 0:
            return
        pixels = self.to_pixels(points)
        in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < grid.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < grid.shape[1])
        grid[pixels[in_bounds, 0], pixels[in_bounds, 1]] = fill

    def decay_goal_score(self) -> None:
        """Ages down goal_score only within the agent's current field of view -- cells outside it keep their
        last score, since no new (dis)confirming observation of them has happened this step."""
        self.goal_score[self.in_view > 0] *= self.config.score_decay

    def goal_mask(self, goal_score_thresh: float) -> np.ndarray:
        """Every cell tied at goal_score's current peak, once that peak clears goal_score_thresh -- not a single
        frozen point, so DiscretePlanner can pick whichever of them is actually reachable."""
        best_score = float(self.goal_score.max())
        if best_score < goal_score_thresh:
            return np.zeros_like(self.goal_score, dtype=np.uint8)
        return (self.goal_score == best_score).astype(np.uint8)

    def believed_goal_positions(self, goal_score_thresh: float) -> np.ndarray:
        """World (N, 2) (x, z) of every cell goal_mask() currently selects -- the believed goal's horizontal
        footprint, for evaluator.py's TP/FP classification."""
        return self.to_world_xz(np.argwhere(self.goal_mask(goal_score_thresh)))

    def mark_goal(self, depth: np.ndarray, sensor_state: habitat_sim.SixDOFPose, mask: torch.Tensor, score: float, camera: DepthCamera, floor_height: float) -> None:
        """Back-projects mask's in-range depth pixels to world points, denoised, and scatters score onto their
        top-down cells, kept as each cell's running max (decay_goal_score() already ages stale high scores down)."""
        if score <= 0.0:
            return
        points = _goal_mask_points(depth, sensor_state, mask, camera, floor_height, self.config)
        if len(points) == 0:
            return
        self._mark_goal(points, score)

    def _mark_goal(self, points: np.ndarray, score: float) -> None:
        """Scatters points onto goal_score's top-down cells, kept as each cell's running max."""
        grid = self.goal_score
        pixels = self.to_pixels(points)
        in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < grid.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < grid.shape[1])
        rows, cols = pixels[in_bounds, 0], pixels[in_bounds, 1]
        grid[rows, cols] = np.maximum(grid[rows, cols], score)

    def coverage(self) -> float:
        return float((self.explored > 0).sum() / self.explored.size)

    def belief_image(self) -> np.ndarray:
        """BGR top-down view of this map alone: white unexplored, light gray explored, dark gray obstacles --
        no trajectory/goal marker, those are the agent's to draw on top."""
        image = np.full(self.obstacles.shape + (3,), _BELIEF_IMAGE_UNEXPLORED, dtype=np.uint8)
        image[self.explored > 0] = _BELIEF_IMAGE_EXPLORED
        image[self.obstacles > 0] = _BELIEF_IMAGE_OBSTACLE
        return image

    def is_free(self, pixel: np.ndarray) -> bool:
        row, col = int(pixel[0]), int(pixel[1])
        return 0 <= row < self.obstacles.shape[0] and 0 <= col < self.obstacles.shape[1] and not self.obstacles[row, col]

    def is_explored(self, pixel: np.ndarray) -> bool:
        row, col = int(pixel[0]), int(pixel[1])
        return 0 <= row < self.explored.shape[0] and 0 <= col < self.explored.shape[1] and bool(self.explored[row, col])

    def planning_window(self, goal_map: np.ndarray, position: np.ndarray) -> Tuple[slice, slice]:
        """(row, col) slices of the bounding box of the explored cells, goal_map's and position's, grown by
        planning_margin_m and clipped to the map."""
        cells = self.explored | goal_map
        row, col = self.to_pixel(position)
        cells[row, col] = 1
        x, y, width, height = cv2.boundingRect(cells)
        margin = self.pixels(self.config.planning_margin_m)
        rows, cols = self.obstacles.shape
        return slice(max(y - margin, 0), min(y + height + margin, rows)), slice(max(x - margin, 0), min(x + width + margin, cols))

    def frontier_map(self) -> np.ndarray:
        """(M, M) uint8 dense mask of frontier_pixels, for DiscretePlanner.plan()'s frontier_map/goal_map arg."""
        grid = np.zeros_like(self.obstacles)
        if len(self.frontier_pixels):
            pixels = np.round(self.frontier_pixels).astype(int)
            in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < grid.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < grid.shape[1])
            grid[pixels[in_bounds, 0], pixels[in_bounds, 1]] = 1
        return grid
