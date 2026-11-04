"""Top-down map of the agent's own depth-observed world: a fixed grid centered on the episode start, built
incrementally from depth (no ground-truth navmesh). Cells start free (unknown is traversable until a depth
point marks it an obstacle, matching facebookresearch/home-robot's DiscretePlanner convention) and
accumulate forever; `explored` tracks what's actually been observed, for frontier detection only -- it never
gates what FMMPlanner can plan through."""
from dataclasses import dataclass

import numpy as np
import torch

from common.hybrid_exploration.frontiers import detect_frontier_waypoints, reveal_fog_of_war, without_small_frontiers
from common.utils.projection_utils import DepthCamera, object_points, project_batch_masks
from common.utils.pose_utils import sensor_pose_matrix


@dataclass(frozen=True)
class MapConfig:
    fov_deg: float
    resolution_m: float = 0.1
    map_size_m: float = 48.0
    visibility_dist_m: float = 3.0
    area_thresh_m: float = 3.0
    min_frontier_size_m: float = 0.5  # shorter frontier fragments are dropped (without_small_frontiers)
    obstacle_height_m: float = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
    max_climb_m: float = 0.2  # matches habitat-sim's NavMeshSettings.agent_max_climb: lower steps/sills are traversable
    min_frontier_dist_m: float = 0.5  # frontier cells this close to the agent are excluded -- reveal_fog_of_war's
    # FOV cone has its apex at the agent's own position, which (before it has moved anywhere) is itself explored
    # but surrounded by unexplored space on most sides, so it would otherwise qualify as a "frontier" and make
    # DiscretePlanner's closest-goal collapse to the agent's current position (distance 0, no real target)

# white unexplored, light gray explored, dark gray obstacles (habitat_od_search/evaluator.py's old
# _obstacle_map scheme) -- BGR, matching common/utils/plot_utils.py's own convention
_BELIEF_IMAGE_UNEXPLORED = (255, 255, 255)
_BELIEF_IMAGE_EXPLORED = (225, 225, 225)
_BELIEF_IMAGE_OBSTACLE = (90, 90, 90)

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

    def pixels(self, meters: float) -> int:
        return int(meters / self.meters_per_pixel)

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        """(N, 2) (row, col) of (N, 3) world positions."""
        offsets = (positions[:, [2, 0]] - self.origin_world[[2, 0]]) / self.meters_per_pixel
        return offsets.astype(int)

    def to_pixel(self, position: np.ndarray) -> np.ndarray:
        return self.to_pixels(np.asarray(position)[None])[0]

    def to_world(self, pixels: np.ndarray, height: float) -> np.ndarray:
        """World points of (row, col) pixels, at height."""
        offsets = (pixels.astype(float) + 0.5) * self.meters_per_pixel
        xz = offsets[:, ::-1] + self.origin_world[[0, 2]]
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
        min_depth) -- e.g. the void past a terrace edge, which would otherwise stay free and keep yielding frontiers."""
        missing = torch.as_tensor(np.asarray(depth).squeeze() <= camera.min_depth)
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

    def frontier_map(self) -> np.ndarray:
        """(M, M) uint8 dense mask of frontier_pixels, for DiscretePlanner.plan()'s frontier_map/goal_map arg."""
        grid = np.zeros_like(self.obstacles)
        if len(self.frontier_pixels):
            pixels = np.round(self.frontier_pixels).astype(int)
            in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < grid.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < grid.shape[1])
            grid[pixels[in_bounds, 0], pixels[in_bounds, 1]] = 1
        return grid
