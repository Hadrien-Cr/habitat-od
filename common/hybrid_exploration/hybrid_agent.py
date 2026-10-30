from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import habitat_sim
import numpy as np

from common.agents.base_exploration_agent import BaseExplorationAgent
from common.hybrid_exploration.discrete_planner import DiscretePlanner
from common.hybrid_exploration.exploration_map import EstimatedMap, MapConfig
from common.hybrid_exploration.object_viewpoints import Candidate, ObjectBelief, footprint, sector_positions
from common.hybrid_exploration.plotting import draw_goal_map, draw_in_view, draw_objects
from common.utils.projection_utils import DepthCamera

_VISITED_COLOR = (0, 0, 255)  # BGR -- red, matching BaseExplorationAgent's trajectory color
_TRAJECTORY_COLOR = (255, 255, 0)  # BGR -- yellow, the direction arrows between visited points


@dataclass(frozen=True)
class HybridConfig:
    map_config: MapConfig
    turn_angle_deg: float
    planner_config: "DiscretePlannerConfig"
    n_sectors: int = 8
    confidence_thresh: float = 0.5
    standoffs_m: Tuple[float, ...] = (3.0, 2.0, 1.0, 0.5)
    min_standoff_m: float = 0.5
    max_standoff_m: float = 3.0
    standoff_per_size: float = 1.0
    reach_m: float = 0.5
    max_successive_collisions: int = 3


class HybridFrontierAgent(BaseExplorationAgent):
    """Frontier exploration that also inspects confident objects (confidence_thresh and above) from per-object sector
    viewpoints. Every step builds one goal_map over its own depth-built EstimatedMap: every currently unseen viewpoint
    if any object is confident enough to have one, else every current frontier cell -- viewpoints always take priority
    over frontiers, as a hard fork rather than a weighted tradeoff. That goal_map, plus EstimatedMap.obstacles, is
    handed to a shared DiscretePlanner (the same belief-grid-only FMM planner FrontierExplorationAgent/SearchAgent use)
    for a single multi-goal FMM pass, which both picks the geodesically nearest goal cell and gives the next waypoint/
    action -- no ground-truth navmesh pathfinding and no per-candidate cost comparison. Reaching a viewpoint (within
    reach_m) is checked before invoking the planner each step and short-circuits to turning the agent to face the
    object, since DiscretePlanner has no "look at point" concept.
    Uses ObjectBeliefs. see main.py for usage
    """

    map_config: MapConfig
    rng: np.random.Generator
    planner: Optional[DiscretePlanner]
    estimated_map: Optional[EstimatedMap]
    trajectory: List[Tuple[np.ndarray, float]]
    _spin_action: Optional[str]

    camera: DepthCamera

    def __init__(self, config: HybridConfig, camera: DepthCamera, rng: np.random.Generator, dump_location: str, exp_name: str):
        planner = DiscretePlanner(
            turn_angle=config.planner_config.turn_angle_deg,
            collision_threshold=config.planner_config.collision_threshold,
            step_size=config.planner_config.step_size,
            obs_dilation_selem_radius=config.planner_config.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.planner_config.goal_dilation_selem_radius,
            map_size_cm=int(config.map_config.map_size_m * 100),
            map_resolution=int(config.map_config.resolution_m * 100),
            dump_location=dump_location,
            exp_name=exp_name,
        )
        super().__init__(config.map_config, rng, config.turn_angle_deg, planner)
        self.config = config
        self.camera = camera
        self.objects: Dict[int, ObjectBelief] = {}
        self.seen_sectors: Dict[int, np.ndarray] = {}
        self._goal_map: Optional[np.ndarray] = None
        self._had_goal = False
        self._arrived_at: Optional[Tuple] = None
        self._successive_collisions = 0

    def reset(self, position: np.ndarray, floor_height: float) -> None:
        super().reset(position, floor_height)
        self.objects = {}
        self.seen_sectors = {}
        self._goal_map = None
        self._had_goal = False
        self._arrived_at = None
        self._successive_collisions = 0

    def update_objects(self, objects: Dict[int, ObjectBelief]) -> None:
        self.objects = objects

    def seen(self, object_id: int) -> np.ndarray:
        return self.seen_sectors.setdefault(object_id, np.zeros(self.config.n_sectors, dtype=bool))

    def act(self, agent_state: habitat_sim.AgentState, depth: np.ndarray, collided: bool = False) -> str:
        """The next action towards the nearest current goal (every unseen viewpoint, or every frontier if none),
        turning to face an object once within reach of its viewpoint; "stop" once the last max_successive_collisions
        steps all collided."""
        assert self.estimated_map is not None and self.planner is not None, "Call reset() first"

        self._successive_collisions = self._successive_collisions + 1 if collided else 0
        if self._successive_collisions >= self.config.max_successive_collisions:
            return "stop"
        position, heading = self.observe(agent_state, depth, self.camera)
        self._mark_inspected(position, heading)
        spin_action = self.initial_spin()
        if spin_action is not None:
            return spin_action

        viewpoints = self._object_viewpoints(position[1])
        arrived = self._arrived_viewpoint(position, viewpoints)
        if arrived is not None:
            self._arrived_at = arrived.key()
            action = self._turn_towards(position, heading, arrived.look_at)
            return action if action is not None else "turn_left"

        self._goal_map = self._build_goal_map(viewpoints)
        if not self._goal_map.any():
            return "stop" if self._had_goal else self.spin()
        self._had_goal = True
        action, self._closest_goal_map, self._short_term_goal, _ = self.planner.plan(
            obstacle_map=self.estimated_map.obstacles,
            goal_map=self._goal_map,
            frontier_map=self.estimated_map.frontier_map(),
            sensor_pose=self.sensor_pose(position, heading),
            found_goal=False,
            use_dilation_for_stg=True,
        )
        return self.action_name(action)

    def _arrived_viewpoint(self, position: np.ndarray, viewpoints: List[Candidate]) -> Optional[Candidate]:
        """The viewpoint the agent is within reach of, preferring the one it last arrived at for drift tolerance."""
        in_reach = [viewpoint for viewpoint in viewpoints if self._reached(position, viewpoint)]
        if not in_reach:
            return None
        return next((viewpoint for viewpoint in in_reach if viewpoint.key() == self._arrived_at), in_reach[0])

    def _build_goal_map(self, viewpoints: List[Candidate]) -> np.ndarray:
        """uint8 mask over EstimatedMap.obstacles: every viewpoint if any exist, else every frontier cell."""
        assert self.estimated_map is not None, "Call reset() first"

        if not viewpoints:
            return self.estimated_map.frontier_map()
        
        grid = np.zeros_like(self.estimated_map.obstacles)
        pixels = self.estimated_map.to_pixels(np.array([viewpoint.position for viewpoint in viewpoints]))
        in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < grid.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < grid.shape[1])
        grid[pixels[in_bounds, 0], pixels[in_bounds, 1]] = 1

        return grid

    def _mark_inspected(self, position: np.ndarray, heading: float) -> None:
        """Marks seen the sector of every viewpoint the agent stood at, at any past or current pose, facing its object:
        a sector only becomes a candidate once its object is confident/footprint-grown enough, by which point an earlier
        pose may already have covered it."""
        for viewpoint in self._object_viewpoints(position[1]):
            assert viewpoint.sector is not None and viewpoint.object_id is not None and viewpoint.look_at is not None
            if any(
                self._within_reach(past_position, viewpoint) and self._turn_towards(past_position, past_heading, viewpoint.look_at) is None
                for past_position, past_heading in self.trajectory
            ):
                self.seen(viewpoint.object_id)[viewpoint.sector] = True

    def _object_viewpoints(self, height: float) -> List[Candidate]:
        return [
            viewpoint for object_id, belief in self.objects.items() if belief.confidence >= self.config.confidence_thresh
            for viewpoint in self._unseen_viewpoints(object_id, belief, height)
        ]

    def _unseen_viewpoints(self, object_id: int, belief: ObjectBelief, height: float) -> List[Candidate]:
        center, radius = footprint(belief.points)
        volume = float(np.prod(belief.points.max(axis=0) - belief.points.min(axis=0)))
        target_standoff = np.clip(self.config.standoff_per_size * volume ** (1 / 3), self.config.min_standoff_m, self.config.max_standoff_m)
        rings = [sector_positions(center, radius + standoff, self.config.n_sectors, height) for standoff in self.config.standoffs_m]
        seen = self.seen(object_id)
        candidates = []
        for sector in range(self.config.n_sectors):
            if seen[sector]:
                continue
            position = self._best_standoff_position(rings, sector, target_standoff)
            if position is not None:
                candidates.append(Candidate(position, object_id, sector, center))
        return candidates

    def _best_standoff_position(self, rings: List[np.ndarray], sector: int, target_standoff: float) -> Optional[np.ndarray]:
        """This sector's position, among rings' standoffs, explored and free and closest to target_standoff; None if
        none qualify -- a goal_map cell the planner hasn't yet observed as free is otherwise indistinguishable from
        one it has, so an unexplored standoff could be picked as a goal without any evidence it's actually reachable."""
        candidates = []
        for standoff, ring in zip(self.config.standoffs_m, rings):
            pixel = self.estimated_map.to_pixel(ring[sector])
            if self.estimated_map.is_explored(pixel) and self.estimated_map.is_free(pixel):
                candidates.append((abs(standoff - target_standoff), ring[sector]))
        return min(candidates, key=lambda entry: entry[0])[1] if candidates else None

    def _reached(self, position: np.ndarray, viewpoint: Candidate) -> bool:
        """In reach, or the viewpoint the agent last arrived at: it drifts as the object's points grow, which would
        otherwise flip the agent between facing the object and walking back to it."""
        return viewpoint.key() == self._arrived_at or self._within_reach(position, viewpoint)

    def _within_reach(self, position: np.ndarray, viewpoint: Candidate) -> bool:
        return float(np.hypot(*(viewpoint.position - position)[[0, 2]])) <= self.config.reach_m

    def _turn_towards(self, position: np.ndarray, heading: float, point: np.ndarray) -> Optional[str]:
        """The turn bringing point within one turn step of the heading, None once it already is."""
        bearing = np.arctan2(point[2] - position[2], point[0] - position[0])
        error = np.arctan2(np.sin(bearing - (np.pi / 2 - heading)), np.cos(bearing - (np.pi / 2 - heading)))
        if error > self._turn_angle:
            return "turn_right"
        if error < -self._turn_angle:
            return "turn_left"
        return None

    @property
    def _turn_angle(self) -> float:
        return np.deg2rad(self.config.turn_angle_deg)

    def map_image(self) -> np.ndarray:
        """BGR top-down map: the belief map (EstimatedMap.belief_image), current field of view, trajectory, each
        object's points and confidence, and the current goal_map cells."""
        assert self.estimated_map is not None, "Call reset() first"

        positions = np.array([position for position, _ in self.trajectory]).reshape(-1, 3)
        image = self.estimated_map.belief_image()
        trajectory_px = [(int(col), int(row)) for row, col in self.estimated_map.to_pixels(positions)]

        for point in trajectory_px:
            cv2.circle(image, point, 1, _VISITED_COLOR, -1)
        for start, end in zip(trajectory_px, trajectory_px[1:]):
            cv2.arrowedLine(image, start, end, _TRAJECTORY_COLOR, 2, tipLength=0.2)
        if trajectory_px:
            cv2.circle(image, trajectory_px[-1], 5, _VISITED_COLOR, 2)
            
        draw_in_view(image, self.estimated_map.in_view)
        objects = {object_id: (self.estimated_map.to_pixels(belief.points), belief.confidence) for object_id, belief in self.objects.items()}
        draw_objects(image, objects, self.config.confidence_thresh)
        if self._goal_map is not None:
            draw_goal_map(image, self._goal_map)
        return image

    def map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB map_image) mosaic tile of the current step."""
        return f"step{len(self.trajectory) - 1:03d}", self.map_image()[:, :, ::-1].copy()
