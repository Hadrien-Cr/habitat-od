"""Point-turn completion-time costs for choosing a frontier, ported from naokiyokoyama/frontier_exploration
(MIT, commit 8523aa8): utils/path_utils.py. Headings are habitat-lab polar angles (TopDownMap.get_polar_angle)."""
from dataclasses import dataclass
from typing import Callable, Optional

import habitat_sim
import numpy as np

from common.utils.pose_utils import normalize_radians


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


def path_time_cost(pathfinder: habitat_sim.PathFinder, start: np.ndarray, polar_heading: float, end: np.ndarray, speeds: PointTurnSpeeds) -> float:
    path = find_path(pathfinder, start, end)
    if path is None or len(path.points) < 2:
        return np.inf
    points = np.array(path.points)
    return path_completion_time(points, abs(heading_error(start, points[1], polar_heading)), speeds)


def best_first_search(candidates: np.ndarray, heuristics: np.ndarray, cost_fn: Callable[[np.ndarray], float]) -> Optional[int]:
    """Index of the cheapest candidate, skipping any whose heuristic already exceeds the best cost found."""
    best_idx, min_cost = None, np.inf
    for idx in np.argsort(heuristics):
        if heuristics[idx] > min_cost:
            break
        cost = cost_fn(candidates[idx])
        if cost < min_cost:
            best_idx, min_cost = int(idx), cost
    return best_idx
