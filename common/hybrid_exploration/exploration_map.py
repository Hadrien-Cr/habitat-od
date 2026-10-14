"""Top-down map of the agent's navmesh island: what has been revealed so far, its frontiers, and world <-> pixel conversion."""
from dataclasses import dataclass

import habitat_sim
import magnum as mn
import numpy as np
from habitat.utils.visualizations import maps  # type: ignore

from common.planning.frontiers import detect_frontier_waypoints, reveal_fog_of_war


@dataclass(frozen=True)
class MapConfig:
    fov_deg: float
    resolution: int = 256
    visibility_dist_m: float = 10.0
    area_thresh_m: float = 3.0


class ExplorationMap:
    def __init__(self, pathfinder: habitat_sim.PathFinder, config: MapConfig, position: np.ndarray):
        self.pathfinder = pathfinder
        self.config = config
        self.meters_per_pixel = maps.calculate_meters_per_pixel(config.resolution, pathfinder=pathfinder)
        islands = pathfinder.get_topdown_island_view(self.meters_per_pixel, position[1])
        self.navigable = np.ascontiguousarray(islands == pathfinder.get_island(mn.Vector3(*position)), dtype=np.uint8)
        self.revealed = np.zeros_like(self.navigable)
        self.in_view = np.zeros_like(self.navigable)
        self.frontier_pixels = np.empty((0, 2))

    def pixels(self, meters: float) -> int:
        return int(meters / self.meters_per_pixel)

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        """(N, 2) (row, col) of (N, 3) world positions."""
        lower, upper = (np.array(bound)[[2, 0]] for bound in self.pathfinder.get_bounds())
        cell_size = np.abs(upper - lower) / np.array(self.navigable.shape)
        return ((positions[:, [2, 0]] - lower) / cell_size).astype(int)

    def to_pixel(self, position: np.ndarray) -> np.ndarray:
        return self.to_pixels(np.asarray(position)[None])[0]

    def to_world(self, pixels: np.ndarray, height: float) -> np.ndarray:
        """Navmesh-snapped world points of (row, col) pixels."""
        zs, xs = maps.from_grid(pixels[:, 0], pixels[:, 1], self.navigable.shape, pathfinder=self.pathfinder)
        return np.array([self.pathfinder.snap_point(mn.Vector3(x, height, z)) for x, z in zip(xs, zs)])

    def contains(self, pixel: np.ndarray) -> bool:
        return bool(0 <= pixel[0] < self.navigable.shape[0] and 0 <= pixel[1] < self.navigable.shape[1])

    def is_revealed_navigable(self, pixel: np.ndarray) -> bool:
        return self.contains(pixel) and bool(self.navigable[tuple(pixel)] and self.revealed[tuple(pixel)])

    def frontiers(self, height: float) -> np.ndarray:
        return self.to_world(self.frontier_pixels, height) if len(self.frontier_pixels) else np.empty((0, 3))

    def reveal(self, position: np.ndarray, heading: float) -> None:
        """Sets in_view to the field of view at (position, heading) and adds it to the revealed area, recomputing frontiers if it grew."""
        self.in_view = reveal_fog_of_war(
            self.navigable, np.zeros_like(self.navigable), self.to_pixel(position), heading, self.config.fov_deg, self.pixels(self.config.visibility_dist_m),
        )
        previous = self.revealed
        self.revealed = previous | self.in_view
        if not np.array_equal(previous, self.revealed):
            self.frontier_pixels = detect_frontier_waypoints(self.navigable, self.revealed, self.pixels(self.config.area_thresh_m ** 2))

    def coverage(self) -> float:
        navigable = self.navigable > 0
        return float(((self.revealed > 0) & navigable).sum() / navigable.sum())
