from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import habitat_sim
import numpy as np

from common.agents.base_exploration_agent import BaseExplorationAgent
from common.hybrid_exploration.discrete_planner import DiscretePlanner
from common.hybrid_exploration.exploration_map import EstimatedMap, MapConfig
from common.hybrid_exploration.object_viewpoints import Candidate, ObjectBelief, footprint, sector_positions
from common.hybrid_exploration.plotting import draw_candidates, draw_in_view, draw_objects
from common.utils.pose_utils import normalize_radians
from common.utils.projection_utils import DepthCamera
from common.vision.detector import Detector

_MIN_TIME_S = 1e-3
# a frontier is only revealed by facing it, so side steps are kept for object viewpoints
_STEP_BEARINGS = {"move_forward": 0.0, "move_right": np.pi / 2, "move_left": -np.pi / 2, "move_backward": np.pi}
_VISITED_COLOR = (0, 0, 255)  # BGR -- red, matching BaseExplorationAgent's trajectory color
_TRAJECTORY_COLOR = (255, 255, 0)  # BGR -- yellow, the direction arrows between visited points


@dataclass(frozen=True)
class PointTurnSpeeds:
    linear_m_per_s: float
    angular_rad_per_s: float

def _bearings(position: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.arctan2(targets[..., 2] - position[2], targets[..., 0] - position[0])

def heading_error(position: np.ndarray, target: np.ndarray, polar_heading: float) -> float:
    return normalize_radians(_bearings(position, target) - normalize_radians(np.pi / 2 - polar_heading))

def completion_time_lower_bounds(targets: np.ndarray, position: np.ndarray, polar_heading: float, speeds: PointTurnSpeeds) -> np.ndarray:
    """Obstacle-free turn-then-drive time to each target - an admissible heuristic for path_time_cost."""
    distances = np.hypot(targets[:, 2] - position[2], targets[:, 0] - position[0])
    turns = np.abs(heading_error(position, targets, polar_heading))
    return turns / speeds.angular_rad_per_s + distances / speeds.linear_m_per_s

def find_path(pathfinder: habitat_sim.PathFinder, start: np.ndarray, end: np.ndarray) -> Optional[habitat_sim.ShortestPath]:
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    return path if pathfinder.find_path(path) else None

def path_completion_time(points: np.ndarray, initial_turn: float, speeds: PointTurnSpeeds) -> float:
    """Time to follow points with point turns, starting initial_turn radians off the first segment."""
    segments = np.diff(points, axis=0)
    turns = np.concatenate([[initial_turn], np.abs(normalize_radians(np.diff(np.arctan2(segments[:, 2], segments[:, 0]))))])
    distances = np.hypot(segments[:, 2], segments[:, 0])
    return float(np.sum(distances / speeds.linear_m_per_s + turns / speeds.angular_rad_per_s))

def _path_points(pathfinder: habitat_sim.PathFinder, start: np.ndarray, end: np.ndarray) -> Optional[np.ndarray]:
    path = find_path(pathfinder, start, end)
    return None if path is None or len(path.points) < 2 else np.array(path.points)

def path_time_cost(pathfinder: habitat_sim.PathFinder, start: np.ndarray, polar_heading: float, end: np.ndarray, speeds: PointTurnSpeeds) -> float:
    points = _path_points(pathfinder, start, end)
    if points is None:
        return np.inf
    return path_completion_time(points, abs(heading_error(start, points[1], polar_heading)), speeds)

def facing_path_time_cost(pathfinder: habitat_sim.PathFinder, start: np.ndarray, polar_heading: float, end: np.ndarray, look_at: np.ndarray, speeds: PointTurnSpeeds) -> float:
    """path_time_cost plus the final point turn from the last segment to face look_at."""
    points = _path_points(pathfinder, start, end)
    if points is None:
        return np.inf
    last_segment = points[-1] - points[-2]
    final_turn = abs(normalize_radians(_bearings(points[-1], look_at) - np.arctan2(last_segment[2], last_segment[0])))
    return path_completion_time(points, abs(heading_error(start, points[1], polar_heading)), speeds) + final_turn / speeds.angular_rad_per_s


class ViewpointMode(str, Enum):
    """EXHAUSTIVE keeps every viewpoint until it is reached; COVERAGE also drops the neighbours within coverage_angle_deg of a reached one."""
    EXHAUSTIVE = "exhaustive"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class HybridConfig:
    map_config: MapConfig
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


class HybridFrontierAgent(BaseExplorationAgent):
    """Frontier exploration that also inspects confident objects (confidence_thresh and above) from per-object sector
    viewpoints, picking whichever candidate (frontier or viewpoint) has the highest gain per point-turn completion time and
    keeping it as its goal until a frontier is reached or a viewpoint inspected. Candidates (frontiers, viewpoints) come
    from its own depth-built EstimatedMap, but path costs/waypoints between the agent and a candidate use pathfinder (the
    sim's ground-truth navmesh, via path_time_cost/facing_path_time_cost in _candidate_time and _path_points in
    _action_towards) -- EstimatedMap has no path-planning of its own. The intended fix is to replace pathfinder with
    DiscretePlanner (already used, belief-grid-only, by FrontierExplorationAgent/SearchAgent): its FMM distance field
    over EstimatedMap.obstacles would give both a cost (distance) and a next waypoint (short-term goal) without ground
    truth, but that needs each candidate's cost computed from one shared multi-goal FMM pass rather than this file's
    current per-candidate pathfinder.find_path calls, so it is not a drop-in replacement.
    Uses ObjectBeliefs. see main.py for usage
    """

    map_config: MapConfig
    rng: np.random.Generator
    planner: Optional[DiscretePlanner]
    estimated_map: Optional[EstimatedMap]
    trajectory: List[Tuple[np.ndarray, float]]
    _spin_action: Optional[str]

    detector: Detector
    camera: DepthCamera
    pathfinder: habitat_sim.PathFinder

    def __init__(self, config: HybridConfig, camera: DepthCamera, rng: np.random.Generator, pathfinder: habitat_sim.PathFinder):
        super().__init__(config.map_config, rng)
        self.config = config
        self.camera = camera
        self.pathfinder = pathfinder
        self.objects: Dict[int, ObjectBelief] = {}
        self.seen_sectors: Dict[int, np.ndarray] = {}
        self.decision: Optional[Decision] = None
        self.goal: Optional[Candidate] = None
        self._had_goal = False
        self._arrived_at: Optional[Tuple] = None
        self._step: Optional[Tuple[Tuple, str]] = None
        self._successive_collisions = 0

    def reset(self, position: np.ndarray, floor_height: float) -> None:
        super().reset(position, floor_height)
        self.objects = {}
        self.seen_sectors = {}
        self.decision = None
        self.goal = None
        self._had_goal = False
        self._arrived_at = None
        self._step = None
        self._successive_collisions = 0

    def update_objects(self, objects: Dict[int, ObjectBelief]) -> None:
        self.objects = objects

    def seen(self, object_id: int) -> np.ndarray:
        return self.seen_sectors.setdefault(object_id, np.zeros(self.config.n_sectors, dtype=bool))

    def act(self, agent_state: habitat_sim.AgentState, depth: np.ndarray, collided: bool = False) -> str:
        """The next action towards the current goal, picking one only when there is none; "stop" once the last
        max_successive_collisions steps all collided."""
        self._successive_collisions = self._successive_collisions + 1 if collided else 0
        if self._successive_collisions >= self.config.max_successive_collisions:
            return "stop"
        position, heading = self.observe(agent_state, depth, self.camera)
        self._mark_inspected(position, heading)
        self.goal = self._kept_goal(position, heading)
        if self.goal is None:
            self.decision = self._decide(position, heading)
            if self.decision is None:
                return "stop" if self._had_goal else self.spin()
            self._had_goal, self.goal = True, self.decision.goal
        self._arrived_at = self.goal.key() if self.goal.look_at is not None and self._reached(position, self.goal) else None
        return self._action_towards(position, heading, self.goal)

    def _mark_inspected(self, position: np.ndarray, heading: float) -> None:
        """Marks seen the sectors of every viewpoint the agent stood at, at any past or current pose, facing its object:
        a sector only becomes a candidate once its object is confident/footprint-grown enough, by which point an earlier
        pose may already have covered it."""
        for viewpoint in self._object_viewpoints(position[1]):
            assert viewpoint.sector is not None and viewpoint.object_id is not None and viewpoint.look_at is not None
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
        assert self.estimated_map is not None, "Call reset() first"
        frontiers = [Candidate(frontier, self.config.frontier_gain) for frontier in self.estimated_map.frontiers(position[1])]
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
        """This sector's position, among rings' standoffs, free and closest to target_standoff; None if none are free."""
        free = [
            (abs(standoff - target_standoff), ring[sector]) for standoff, ring in zip(self.config.standoffs_m, rings)
            if self.estimated_map.is_free(self.estimated_map.to_pixel(ring[sector]))
        ]
        return min(free, key=lambda entry: entry[0])[1] if free else None

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
        assert self.estimated_map is not None, "Call reset() first"

        if goal.look_at is not None and self._reached(position, goal):
            action = self._turn_towards(position, heading, goal.look_at)
            assert action is not None
            return action

        points = _path_points(self.pathfinder, position, goal.position)
        if points is None:
            return "stop"
        waypoint_error = heading_error(position, points[1], heading)
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


    def map_image(self) -> np.ndarray:
        """BGR top-down map: the belief map (EstimatedMap.belief_image), current field of view, trajectory, each
        object's points and confidence, and every candidate colored by its score."""
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
        if self.decision is not None:
            pixels = self.estimated_map.to_pixels(np.array([c.position for c in self.decision.candidates]))
            draw_candidates(image, pixels, self.decision.scores)
        return image

    def map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB map_image) mosaic tile of the current step and decision."""
        label = f"step{len(self.trajectory) - 1:03d}" + ("" if self.decision is None else f" {self.decision.label()}")
        return label, self.map_image()[:, :, ::-1].copy()
