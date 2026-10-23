"""Frontier exploration that also inspects confident objects (confidence_thresh and above) from per-object sector
viewpoints, picking whichever candidate (frontier or viewpoint) has the highest gain per point-turn completion time and
keeping it as its goal until a frontier is reached or a viewpoint inspected."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap  # type: ignore

from common.agents.abstract_agent import Agent
from common.hybrid_exploration.exploration_map import ExplorationMap, MapConfig
from common.hybrid_exploration.object_viewpoints import Candidate, ObjectBelief, footprint, sector_positions
from common.hybrid_exploration.plotting import draw_candidates, draw_in_view, draw_objects
from common.planning.frontier_paths import PointTurnSpeeds, facing_path_time_cost, find_path, heading_error, path_time_cost
from common.utils.plot_utils import plot_exploration_map
from common.utils.pose_utils import normalize_radians

_MIN_TIME_S = 1e-3
# a frontier is only revealed by facing it, so side steps are kept for object viewpoints
_STEP_BEARINGS = {"move_forward": 0.0, "move_right": np.pi / 2, "move_left": -np.pi / 2, "move_backward": np.pi}


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
    confidence_thresh: float = 0.5
    viewpoint_gain: float = 1.0
    standoffs_m: Tuple[float, ...] = (3.0, 2.0, 1.0, 0.5)
    min_standoff_m: float = 0.5
    max_standoff_m: float = 3.0
    standoff_per_size: float = 1.0
    frontier_gain: float = 0.5
    reach_m: float = 0.25
    viewpoint_mode: ViewpointMode = ViewpointMode.EXHAUSTIVE
    coverage_angle_deg: float = 60.0
    max_successive_collisions: int = 3


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
        self.trajectory: List[Tuple[np.ndarray, float]] = []
        self.objects: Dict[int, ObjectBelief] = {}
        self.seen_sectors: Dict[int, np.ndarray] = {}
        self.decision: Optional[Decision] = None
        self.goal: Optional[Candidate] = None
        self._had_goal = False
        self._spin_action: Optional[str] = None
        self._arrived_at: Optional[Tuple] = None
        self._step: Optional[Tuple[Tuple, str]] = None
        self._successive_collisions = 0

    def update_objects(self, objects: Dict[int, ObjectBelief]) -> None:
        self.objects = objects

    def seen(self, object_id: int) -> np.ndarray:
        return self.seen_sectors.setdefault(object_id, np.zeros(self.config.n_sectors, dtype=bool))

    def act(self, agent_state: habitat_sim.AgentState, collided: bool = False) -> str:
        """The next action towards the current goal, picking one only when there is none; "stop" once the last
        max_successive_collisions steps all collided."""
        self._successive_collisions = self._successive_collisions + 1 if collided else 0
        if self._successive_collisions >= self.config.max_successive_collisions:
            return "stop"
        position = np.array(agent_state.position)
        heading = float(TopDownMap.get_polar_angle(agent_state))
        self.trajectory.append((position, heading))
        if self.map is None:
            self.map = ExplorationMap(self.pathfinder, self.config.map, position)
        self.map.reveal(position, heading)
        self._mark_inspected(position, heading)
        self.goal = self._kept_goal(position, heading)
        if self.goal is None:
            self.decision = self._decide(position, heading)
            if self.decision is None:
                return "stop" if self._had_goal else self._spin()
            self._had_goal, self.goal = True, self.decision.goal
        self._arrived_at = self.goal.key() if self.goal.look_at is not None and self._reached(position, self.goal) else None
        return self._action_towards(position, heading, self.goal)

    def _mark_inspected(self, position: np.ndarray, heading: float) -> None:
        """Marks seen the sectors of every viewpoint the agent stood at, at any past or current pose, facing its object:
        a sector only becomes a candidate once its object is confident/footprint-grown enough, by which point an earlier
        pose may already have covered it."""
        for viewpoint in self._object_viewpoints(position[1]):
            if any(
                self._within_reach(past_position, viewpoint) and self._turn_towards(past_position, past_heading, viewpoint.look_at) is None
                for past_position, past_heading in self.trajectory
            ):
                self.seen(viewpoint.object_id)[self._covered_sectors(viewpoint.sector)] = True

    def _kept_goal(self, position: np.ndarray, heading: float) -> Optional[Candidate]:
        """The current goal, a viewpoint refreshed from its object's points, or None once a frontier is reached, a viewpoint
        is inspected or no longer a candidate, or no path leads to it."""
        if self.goal is None or (self.goal.look_at is None and self._reached(position, self.goal)):
            return None
        goal = self.goal if self.goal.look_at is None else next(
            (viewpoint for viewpoint in self._object_viewpoints(position[1]) if viewpoint.key() == self.goal.key()), None,
        )
        return goal if goal is not None and np.isfinite(self._candidate_time(position, heading, goal)) else None

    def _decide(self, position: np.ndarray, heading: float) -> Optional[Decision]:
        frontiers = [Candidate(frontier, self.config.frontier_gain) for frontier in self.map.frontiers(position[1])]
        candidates = frontiers + self._object_viewpoints(position[1])
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
                candidates.append(Candidate(position, self.config.viewpoint_gain, object_id, sector, center))
        return candidates

    def _best_standoff_position(self, rings: List[np.ndarray], sector: int, target_standoff: float) -> Optional[np.ndarray]:
        """This sector's position, among rings' standoffs, navigable and closest to target_standoff; None if none are navigable."""
        navigable = [
            (abs(standoff - target_standoff), ring[sector]) for standoff, ring in zip(self.config.standoffs_m, rings)
            if self.map.is_revealed_navigable(self.map.to_pixel(ring[sector]))
        ]
        return min(navigable, key=lambda entry: entry[0])[1] if navigable else None

    def _reached(self, position: np.ndarray, viewpoint: Candidate) -> bool:
        """In reach, or the viewpoint the agent last arrived at: it drifts as the object's points grow, which would
        otherwise flip the agent between facing the object and walking back to it."""
        return viewpoint.key() == self._arrived_at or self._within_reach(position, viewpoint)

    def _within_reach(self, position: np.ndarray, viewpoint: Candidate) -> bool:
        return float(np.hypot(*(viewpoint.position - position)[[0, 2]])) <= self.config.reach_m

    def _candidate_time(self, position: np.ndarray, heading: float, candidate: Candidate) -> float:
        speeds = self.config.speeds
        if candidate.look_at is None:
            return path_time_cost(self.pathfinder, position, heading, candidate.position, speeds)
        if self._reached(position, candidate):
            return abs(heading_error(position, candidate.look_at, heading)) / speeds.angular_rad_per_s
        return facing_path_time_cost(self.pathfinder, position, heading, candidate.position, candidate.look_at, speeds)

    def _action_towards(self, position: np.ndarray, heading: float, goal: Candidate) -> str:
        """Turns to face the object once at its viewpoint, otherwise heads for the path's next waypoint: forward to a
        frontier, or by a step from the yaw closest to facing the object from its viewpoint."""
        if goal.look_at is not None and self._reached(position, goal):
            return self._turn_towards(position, heading, goal.look_at)
        waypoint_error = heading_error(position, np.array(find_path(self.pathfinder, position, goal.position).points[1]), heading)
        if goal.look_at is None:
            return self._turn(waypoint_error) or "move_forward"
        if self._keeps_step(goal, waypoint_error):
            return self._step[1]
        target_error = heading_error(goal.position, goal.look_at, heading)
        yaw_turns = {
            action: int(np.round(normalize_radians(waypoint_error - bearing) / self._turn_angle)) for action, bearing in _STEP_BEARINGS.items()
        }
        step = min(yaw_turns, key=lambda action: (
            max(int(np.ceil(abs(normalize_radians(target_error - yaw_turns[action] * self._turn_angle)) / self._turn_angle)) - 1, 0),
            abs(yaw_turns[action]),
        ))
        turns = yaw_turns[step]
        self._step = None if turns else (goal.key(), step)
        return step if not turns else "turn_right" if turns > 0 else "turn_left"

    def _keeps_step(self, goal: Candidate, waypoint_error: float) -> bool:
        """Whether the step last taken towards goal still heads within one turn step of the waypoint: re-picking at half a
        turn step would zigzag between two yaws as the waypoint's bearing drifts across their boundary."""
        if self._step is None or self._step[0] != goal.key():
            return False
        return self._turn(normalize_radians(waypoint_error - _STEP_BEARINGS[self._step[1]])) is None

    def _turn_towards(self, position: np.ndarray, heading: float, point: np.ndarray) -> Optional[str]:
        """The turn bringing point within one turn step of the heading, None once it already is."""
        return self._turn(heading_error(position, point, heading))

    def _turn(self, error: float) -> Optional[str]:
        if error > self._turn_angle:
            return "turn_right"
        if error < -self._turn_angle:
            return "turn_left"
        return None

    @property
    def _turn_angle(self) -> float:
        return np.deg2rad(self.config.turn_angle_deg)

    def _spin(self) -> str:
        if self._spin_action is None:
            self._spin_action = "turn_left" if self.rng.integers(2) else "turn_right"
        return self._spin_action

    @property
    def top_down_map(self) -> np.ndarray:
        """FrontierAgent's map interface, so common/utils/projection_utils.py's top-down maps draw on this agent's map too."""
        return self.map.navigable

    def meters_per_pixel(self) -> float:
        return self.map.meters_per_pixel

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        return self.map.to_pixels(positions)

    def map_image(self) -> np.ndarray:
        """BGR top-down map of the revealed area, current field of view and trajectory, each object's points and confidence,
        and every candidate colored by its score."""
        positions = np.array([position for position, _ in self.trajectory])
        image = plot_exploration_map(self.map.navigable, self.map.revealed, list(self.map.to_pixels(positions)))
        draw_in_view(image, self.map.in_view)
        objects = {object_id: (self.map.to_pixels(belief.points), belief.confidence) for object_id, belief in self.objects.items()}
        draw_objects(image, objects, self.config.confidence_thresh)
        if self.decision is not None:
            candidates = self.decision.candidates
            pixels = self.map.to_pixels(np.array([c.position for c in candidates]))
            labels = [(f"{score * time:.3f}", f"{time:.0f}") for score, time in zip(self.decision.scores, self.decision.times)]
            draw_candidates(image, pixels, [c.object_id is not None for c in candidates], self.decision.scores, labels)
        return image

    def map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB map_image) mosaic tile of the current step and decision."""
        label = f"step{len(self.trajectory) - 1:03d}" + ("" if self.decision is None else f" {self.decision.label()}")
        return label, self.map_image()[:, :, ::-1].copy()
