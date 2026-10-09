"""Frontier-based exploration, ported from naokiyokoyama/frontier_exploration (MIT, commit 8523aa8):
base_explorer.py::BaseExplorer. The field of view is revealed on the top-down map of the agent's own
navmesh island (the reference uses the whole navmesh slab at the agent's height), the
frontier midpoint with the lowest point-turn completion time along its navmesh shortest path is
picked, and the agent point-turns towards that path's next point with vanilla
move_forward/turn_left/turn_right actions, from its continuous pose. Spins in place until a first
frontier shows up, and stops once none is left."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import habitat_sim
import magnum as mn
import numpy as np
from habitat.tasks.nav.nav import TopDownMap  # type: ignore
from habitat.utils.visualizations import maps  # type: ignore

from common.agents.abstract_agent import Agent
from common.planning.frontier_paths import (
    PointTurnSpeeds, best_first_search, completion_time_lower_bounds, find_path, heading_error, path_time_cost,
)
from common.planning.frontiers import detect_frontier_waypoints, reveal_fog_of_war
from common.utils.plot_utils import plot_exploration_map


@dataclass
class FrontierAgentConfig:
    fov_deg: float
    turn_angle_deg: float
    map_resolution: int = 256
    visibility_dist_m: float = 4.5
    area_thresh_m: float = 3.0
    speeds: PointTurnSpeeds = field(default_factory=lambda: PointTurnSpeeds(0.25, np.deg2rad(10.0)))


class FrontierAgent(Agent):
    def __init__(self, pathfinder: habitat_sim.PathFinder, config: FrontierAgentConfig, rng: np.random.Generator):
        self.pathfinder = pathfinder
        self.config = config
        self.rng = rng
        self.top_down_map: Optional[np.ndarray] = None
        self.fog_of_war_mask: Optional[np.ndarray] = None
        self.frontier_waypoints = np.empty((0, 2))
        self.trajectory: list = []
        self.closest_frontier_waypoint: Optional[np.ndarray] = None
        self._found_frontier = False
        self._spin_action: Optional[str] = None

    def act(self, agent_state: habitat_sim.AgentState) -> str:
        position = np.array(agent_state.position)
        self.trajectory.append(position)
        if self.top_down_map is None:
            self.top_down_map = self._own_island_map(position)
            self.fog_of_war_mask = np.zeros_like(self.top_down_map)
        heading = float(TopDownMap.get_polar_angle(agent_state))
        self._update_frontiers(position, heading)
        self.closest_frontier_waypoint = self._closest_frontier(position, heading)
        if self.closest_frontier_waypoint is None:
            return "stop" if self._found_frontier else self._spin()
        self._found_frontier = True
        return self._point_turn_action(position, heading, self.closest_frontier_waypoint)

    def _meters_per_pixel(self) -> float:
        return maps.calculate_meters_per_pixel(self.config.map_resolution, pathfinder=self.pathfinder)

    def _pixels(self, meters: float) -> int:
        return int(meters / self._meters_per_pixel())

    def _own_island_map(self, position: np.ndarray) -> np.ndarray:
        """Navigable mask of the agent's own navmesh island only - get_topdown_view's +-0.5 m slab would also count furniture tops (e.g. a mattress) as floor."""
        islands = self.pathfinder.get_topdown_island_view(self._meters_per_pixel(), position[1])
        return np.ascontiguousarray(islands == self.pathfinder.get_island(mn.Vector3(*position)), dtype=np.uint8)

    def to_pixel(self, position: np.ndarray) -> np.ndarray:
        """(row, col) of a world position on top_down_map."""
        return np.array(maps.to_grid(position[2], position[0], self.top_down_map.shape, pathfinder=self.pathfinder))

    def _to_world(self, pixels: np.ndarray, height: float) -> np.ndarray:
        """Navmesh-snapped world points of (row, col) map pixels."""
        zs, xs = maps.from_grid(pixels[:, 0], pixels[:, 1], self.top_down_map.shape, pathfinder=self.pathfinder)
        return np.array([self.pathfinder.snap_point(mn.Vector3(x, height, z)) for x, z in zip(xs, zs)])

    def _update_frontiers(self, position: np.ndarray, heading: float) -> None:
        previous = self.fog_of_war_mask.copy()
        self.fog_of_war_mask = reveal_fog_of_war(
            self.top_down_map, self.fog_of_war_mask, self.to_pixel(position), heading,
            self.config.fov_deg, self._pixels(self.config.visibility_dist_m),
        )
        if not np.array_equal(previous, self.fog_of_war_mask):
            self.frontier_waypoints = detect_frontier_waypoints(self.top_down_map, self.fog_of_war_mask, self._pixels(self.config.area_thresh_m ** 2))

    def _closest_frontier(self, position: np.ndarray, heading: float) -> Optional[np.ndarray]:
        if len(self.frontier_waypoints) == 0:
            return None
        speeds = self.config.speeds
        candidates = self._to_world(self.frontier_waypoints, position[1])
        heuristics = completion_time_lower_bounds(candidates, position, heading, speeds)
        idx = best_first_search(candidates, heuristics, lambda target: path_time_cost(self.pathfinder, position, heading, target, speeds))
        return None if idx is None else candidates[idx]

    def _point_turn_action(self, position: np.ndarray, heading: float, target: np.ndarray) -> str:
        next_waypoint = np.array(find_path(self.pathfinder, position, target).points[1])
        error = heading_error(position, next_waypoint, heading)
        turn_angle = np.deg2rad(self.config.turn_angle_deg)
        if error > turn_angle:
            return "turn_right"
        if error < -turn_angle:
            return "turn_left"
        return "move_forward"

    def map_image(self) -> np.ndarray:
        """BGR top-down map of the revealed area, the trajectory so far and the frontier currently pursued."""
        frontier_px = None if self.closest_frontier_waypoint is None else self.to_pixel(self.closest_frontier_waypoint)
        trajectory_px = [self.to_pixel(position) for position in self.trajectory]
        return plot_exploration_map(self.top_down_map, self.fog_of_war_mask, trajectory_px, frontier_px)

    def map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB map_image) mosaic tile of the current step."""
        return f"step{len(self.trajectory) - 1:03d}", self.map_image()[:, :, ::-1].copy()

    def _spin(self) -> str:
        if self._spin_action is None:
            self._spin_action = "turn_left" if self.rng.integers(2) else "turn_right"
        return self._spin_action
