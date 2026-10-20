"""Frontier exploration that also inspects confident objects (confidence_thresh and above) from per-object sector
viewpoints, picking whichever candidate (frontier or viewpoint) has the highest gain per point-turn completion time."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap  # type: ignore

from common.agents.abstract_agent import Agent
from common.hybrid_exploration.exploration_map import ExplorationMap, MapConfig
from common.hybrid_exploration.object_viewpoints import Candidate, ObjectBelief, clear_line, footprint, sector_positions
from common.hybrid_exploration.plotting import draw_candidates, draw_in_view, draw_objects, plot_occupancy
from common.planning.frontier_paths import PointTurnSpeeds, facing_path_time_cost, find_path, heading_error, path_time_cost
from common.utils.plot_utils import plot_exploration_map

_NAVMESH_CLEARANCE_M = 0.2
_MIN_TIME_S = 1e-3


class ViewpointMode(str, Enum):
    """EXHAUSTIVE keeps every viewpoint until it is reached; COVERAGE also drops the neighbours within coverage_angle_deg of a reached one."""
    EXHAUSTIVE = "exhaustive"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class HybridConfig:
    map: MapConfig
    turn_angle_deg: float
    speeds: PointTurnSpeeds = field(default_factory=lambda: PointTurnSpeeds(0.25, np.deg2rad(40.0)))
    n_sectors: int = 8
    confidence_thresh: float = 0.8
    viewpoint_gain: float = 1.0
    standoff_m: float = 0.75
    frontier_gain: float = 0.5
    reach_m: float = 0.5
    viewpoint_mode: ViewpointMode = ViewpointMode.EXHAUSTIVE
    coverage_angle_deg: float = 60.0


@dataclass
class Decision:
    candidates: List[Candidate]
    times: np.ndarray
    scores: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        gains = np.array([candidate.gain for candidate in self.candidates])
        self.scores = np.where(np.isfinite(self.times), gains / np.maximum(self.times, _MIN_TIME_S), 0.0)

    @property
    def goal(self) -> Candidate:
        return self.candidates[int(np.argmax(self.scores))]

    def candidate_labels(self) -> List[Tuple[str, str]]:
        """(u*t, t) of every candidate, t being the time to reach it."""
        return [(f"{score * time:.3f}", f"{time:.0f}") for score, time in zip(self.scores, self.times)]

    def label(self) -> str:
        goal = self.goal
        name = "frontier" if goal.object_id is None else f"obj{goal.object_id}/s{goal.sector}"
        return f"{name} u={self.scores.max():.3f}"


class HybridFrontierAgent(Agent):
    def __init__(self, pathfinder: habitat_sim.PathFinder, config: HybridConfig, rng: np.random.Generator):
        self.pathfinder = pathfinder
        self.config = config
        self.rng = rng
        self.map: Optional[ExplorationMap] = None
        self.trajectory: list = []
        self.objects: Dict[int, ObjectBelief] = {}
        self.seen_sectors: Dict[int, np.ndarray] = {}
        self.decision: Optional[Decision] = None
        self._had_goal = False
        self._spin_action: Optional[str] = None
        self._arrived_at: Optional[Tuple] = None

    def update_objects(self, objects: Dict[int, ObjectBelief]) -> None:
        self.objects = objects

    def seen(self, object_id: int) -> np.ndarray:
        return self.seen_sectors.setdefault(object_id, np.zeros(self.config.n_sectors, dtype=bool))

    def act(self, agent_state: habitat_sim.AgentState) -> str:
        position = np.array(agent_state.position)
        heading = float(TopDownMap.get_polar_angle(agent_state))
        self.trajectory.append(position)
        if self.map is None:
            self.map = ExplorationMap(self.pathfinder, self.config.map, position)
        self.map.reveal(position, heading)
        self.decision = self._decide(position, heading)
        if self.decision is None:
            return "stop" if self._had_goal else self._spin()
        self._had_goal = True
        goal = self.decision.goal
        self._arrived_at = goal.key() if goal.look_at is not None and self._reached(position, goal) else None
        return self._action_towards(position, heading, goal)

    def _decide(self, position: np.ndarray, heading: float) -> Optional[Decision]:
        viewpoints = self._object_viewpoints(position[1])
        for viewpoint in viewpoints:
            if self._is_inspecting(position, heading, viewpoint):
                self.seen(viewpoint.object_id)[self._covered_sectors(viewpoint.sector)] = True
        frontiers = [Candidate(frontier, self.config.frontier_gain) for frontier in self.map.frontiers(position[1])]
        candidates = frontiers + [viewpoint for viewpoint in viewpoints if not self.seen(viewpoint.object_id)[viewpoint.sector]]
        times = np.array([self._candidate_time(position, heading, candidate) for candidate in candidates])
        return Decision(candidates, times) if np.isfinite(times).any() else None

    def _covered_sectors(self, sector: int) -> np.ndarray:
        """Sectors made redundant by reaching sector's viewpoint."""
        if self.config.viewpoint_mode == ViewpointMode.EXHAUSTIVE:
            return np.array([sector])
        span = int(self.config.coverage_angle_deg // (360.0 / self.config.n_sectors))
        return np.arange(sector - span, sector + span + 1) % self.config.n_sectors

    def _object_viewpoints(self, height: float) -> List[Candidate]:
        return [
            viewpoint for object_id, belief in self.objects.items() if belief.confidence >= self.config.confidence_thresh
            for viewpoint in self._unseen_viewpoints(object_id, belief, height)
        ]

    def _unseen_viewpoints(self, object_id: int, belief: ObjectBelief, height: float) -> List[Candidate]:
        center, radius = footprint(belief.points)
        positions = sector_positions(center, radius + self.config.standoff_m, self.config.n_sectors, height)
        seen = self.seen(object_id)
        return [
            Candidate(position, self.config.viewpoint_gain, object_id, sector, center)
            for sector, position in enumerate(positions) if not seen[sector] and self._viewable(position, center, radius)
        ]

    def _viewable(self, position: np.ndarray, center: np.ndarray, radius: float) -> bool:
        """Revealed, navigable, and with nothing on the top-down map between it and the object's footprint."""
        pixel = self.map.to_pixel(position)
        if not self.map.is_revealed_navigable(pixel):
            return False
        return clear_line(self.map.navigable, pixel, self.map.to_pixel(center), self.map.pixels(radius + _NAVMESH_CLEARANCE_M))

    def _in_reach(self, position: np.ndarray, target: np.ndarray) -> bool:
        return float(np.hypot(*(target - position)[[0, 2]])) <= self.config.reach_m

    def _reached(self, position: np.ndarray, viewpoint: Candidate) -> bool:
        """In reach, or the viewpoint the agent last arrived at: it drifts as the object's points grow, which would
        otherwise flip the agent between facing the object and walking back to it."""
        return viewpoint.key() == self._arrived_at or self._in_reach(position, viewpoint.position)

    def _is_inspecting(self, position: np.ndarray, heading: float, viewpoint: Candidate) -> bool:
        return self._reached(position, viewpoint) and self._turn_towards(position, heading, viewpoint.look_at) is None

    def _candidate_time(self, position: np.ndarray, heading: float, candidate: Candidate) -> float:
        speeds = self.config.speeds
        if candidate.look_at is None:
            return path_time_cost(self.pathfinder, position, heading, candidate.position, speeds)
        if self._reached(position, candidate):
            return abs(heading_error(position, candidate.look_at, heading)) / speeds.angular_rad_per_s
        return facing_path_time_cost(self.pathfinder, position, heading, candidate.position, candidate.look_at, speeds)

    def _action_towards(self, position: np.ndarray, heading: float, goal: Candidate) -> str:
        """Turns to face the object once at its viewpoint, otherwise point-turns along the path to the goal."""
        if goal.look_at is not None and self._reached(position, goal):
            return self._turn_towards(position, heading, goal.look_at)
        next_waypoint = np.array(find_path(self.pathfinder, position, goal.position).points[1])
        return self._turn_towards(position, heading, next_waypoint) or "move_forward"

    def _turn_towards(self, position: np.ndarray, heading: float, point: np.ndarray) -> Optional[str]:
        """The turn bringing point within one turn step of the heading, None once it already is."""
        error = heading_error(position, point, heading)
        turn_angle = np.deg2rad(self.config.turn_angle_deg)
        if error > turn_angle:
            return "turn_right"
        if error < -turn_angle:
            return "turn_left"
        return None

    def _spin(self) -> str:
        if self._spin_action is None:
            self._spin_action = "turn_left" if self.rng.integers(2) else "turn_right"
        return self._spin_action

    @property
    def top_down_map(self) -> np.ndarray:
        """FrontierAgent's map interface, so habitat_embodied_sl/main.py's top-down maps draw on this agent's map too."""
        return self.map.navigable

    def meters_per_pixel(self) -> float:
        return self.map.meters_per_pixel

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        return self.map.to_pixels(positions)

    def map_image(self) -> np.ndarray:
        """BGR top-down map of the revealed area, current field of view and trajectory, each object's points and confidence,
        and every candidate colored by its score."""
        image = plot_exploration_map(self.map.navigable, self.map.revealed, list(self.map.to_pixels(np.array(self.trajectory))))
        draw_in_view(image, self.map.in_view)
        objects = {object_id: (self.map.to_pixels(belief.points), belief.confidence) for object_id, belief in self.objects.items()}
        draw_objects(image, objects, self.config.confidence_thresh)
        if self.decision is not None:
            candidates = self.decision.candidates
            pixels = self.map.to_pixels(np.array([c.position for c in candidates]))
            draw_candidates(image, pixels, [c.object_id is not None for c in candidates], self.decision.scores, self.decision.candidate_labels())
        return image

    def object_map_image(self) -> np.ndarray:
        """BGR top-down map of the believed object occupancy over the explored area."""
        return plot_occupancy(self.map.revealed, [self.map.to_pixels(belief.points) for belief in self.objects.values()])

    def map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB map_image) mosaic tile of the current step and decision."""
        return self._step_label(), self.map_image()[:, :, ::-1].copy()

    def object_map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB object_map_image) mosaic tile of the current step."""
        return self._step_label(), self.object_map_image()[:, :, ::-1].copy()

    def _step_label(self) -> str:
        return f"step{len(self.trajectory) - 1:03d}" + ("" if self.decision is None else f" {self.decision.label()}")
