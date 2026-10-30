"""Ground-truth navmesh topdown view, revealed-area tracking, object footprints and the current step's
actual (occluded) field of view -- for the comparison panel logged alongside an agent's own depth-built
EstimatedMap belief. Never consulted by any agent; built from a raw habitat_sim.PathFinder (and optionally
ExplorationEnv.gt_map_tracker()), so it works with or without a full ExplorationEnv/Hydra config, the same
relationship common/env_utils/env_base.py::gt_navigable_mask has to ExplorationEnv.get_gt_topdown_map."""
from typing import List, Tuple

import cv2
import habitat_sim
import numpy as np
import trimesh
from habitat.utils.visualizations import maps  # type: ignore

from common.env_utils.env_base import gt_navigable_mask
from common.hybrid_exploration.frontiers import reveal_fog_of_war
from common.utils.plot_utils import plot_exploration_map
from common.utils.projection_utils import mesh_footprint_on_grid

_FOV_CONE_COLOR = (0, 255, 255)  # BGR -- yellow
_FOOTPRINT_COLOR = (255, 120, 0)  # BGR -- orange


class GroundTruthMapTracker:
    def __init__(self, pathfinder: habitat_sim.PathFinder, position: np.ndarray, fov_deg: float, visibility_dist_m: float, resolution: int = 512):
        self.pathfinder = pathfinder
        self.meters_per_pixel = maps.calculate_meters_per_pixel(resolution, pathfinder=pathfinder)
        self.navigable = gt_navigable_mask(pathfinder, position, resolution)
        self.revealed = np.zeros_like(self.navigable)
        self.in_view = np.zeros_like(self.navigable)
        self.fov_deg = fov_deg
        self.visibility_px = int(visibility_dist_m / self.meters_per_pixel)
        self.footprints = np.zeros_like(self.navigable, dtype=bool)

    def set_object_footprints(self, meshes: List[Tuple[int, int, trimesh.Trimesh]]) -> None:
        """Marks footprints to the union of meshes' (object_id, class_id, world mesh) top-down silhouettes."""
        footprint = np.zeros_like(self.navigable, dtype=bool)
        for _, _, mesh in meshes:
            footprint |= mesh_footprint_on_grid(mesh, self.navigable.shape, self.to_pixels)
        self.footprints = footprint

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        lower, upper = (np.array(bound)[[2, 0]] for bound in self.pathfinder.get_bounds())
        cell_size = np.abs(upper - lower) / np.array(self.navigable.shape)
        return ((positions[:, [2, 0]] - lower) / cell_size).astype(int)

    def to_pixel(self, position: np.ndarray) -> np.ndarray:
        return self.to_pixels(np.asarray(position)[None])[0]

    def reveal(self, position: np.ndarray, heading: float) -> None:
        self.in_view = reveal_fog_of_war(self.navigable, np.zeros_like(self.revealed), self.to_pixel(position), heading, self.fov_deg, self.visibility_px)
        self.revealed = self.revealed | self.in_view

    def _draw_fov_cone(self, image: np.ndarray) -> None:
        """Outlines in_view in yellow -- the actual occluded visible area reveal_fog_of_war computed this step
        (walls block line-of-sight), not the raw unoccluded FOV wedge."""
        contours, _ = cv2.findContours(self.in_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, _FOV_CONE_COLOR, 2)

    def image(self, trajectory_px: list) -> np.ndarray:
        """BGR ground-truth map: navmesh + revealed area + trajectory (plot_exploration_map), with object
        footprints tinted orange and the current step's actual visible area outlined in yellow on top."""
        image = plot_exploration_map(self.navigable, self.revealed, trajectory_px)
        image[self.footprints] = _FOOTPRINT_COLOR
        self._draw_fov_cone(image)
        return image
