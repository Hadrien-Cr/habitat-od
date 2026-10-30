"""Field-of-view reveal (ported from naokiyokoyama/frontier_exploration, MIT, commit 8523aa8:
utils/fog_of_war.py::reveal_fog_of_war) and frontier extraction on a navigable top-down map. Maps are (row,
col) uint8 masks; cv2 contours (reveal_fog_of_war only) are (x, y)."""
from typing import Tuple

import cv2
import numpy as np

from common.utils.pose_utils import normalize_radians

_MAX_VISIBLE_REGION_DIST_PX = 3
_ADJACENCY_KERNEL = np.ones((3, 3), np.uint8)
_OBSTACLE_MARGIN_PX = 2


def reveal_fog_of_war(
    top_down_map: np.ndarray, fog_of_war_mask: np.ndarray, current_point: np.ndarray, current_angle: float, fov: float, max_line_len: float,
) -> np.ndarray:
    """Adds to fog_of_war_mask the part of the fov cone at current_point (row, col) not hidden behind top_down_map's obstacles."""
    center = tuple(int(v) for v in current_point[::-1])
    angle_deg = np.rad2deg(normalize_radians(np.pi / 2 - current_angle))
    cone_mask = cv2.ellipse(
        np.zeros_like(top_down_map), center, (int(max_line_len), int(max_line_len)), 0, angle_deg - fov / 2, angle_deg + fov / 2, 1, -1,
    )
    obstacle_contours, _ = cv2.findContours(cv2.bitwise_and(cone_mask, 1 - top_down_map), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not obstacle_contours:
        return fog_of_war_mask
    visible_cone = cv2.bitwise_and(cone_mask, top_down_map)
    occluding_points = _occluding_points(center, obstacle_contours, np.deg2rad(angle_deg))
    cv2.polylines(visible_cone, _occlusion_lines(center, occluding_points, max_line_len * 1.05), isClosed=False, color=0, thickness=2)
    return _fill_closest_region(fog_of_war_mask, visible_cone, center)


def _occluding_points(center: Tuple[int, int], contours: list, angle: float) -> np.ndarray:
    points = [_extreme_angle_points(center, contour, angle) if cv2.isContourConvex(contour) else contour.reshape(-1, 2) for contour in contours]
    return np.concatenate(points)


def _extreme_angle_points(center: Tuple[int, int], contour: np.ndarray, angle: float) -> np.ndarray:
    """The two contour points with the smallest and largest bearing from center, relative to angle."""
    points = contour.reshape(-1, 2)
    rotation = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    rotated = (points - np.array(center)) @ rotation
    bearings = np.arctan2(rotated[:, 1], rotated[:, 0])
    return points[[np.argmin(bearings), np.argmax(bearings)]]


def _occlusion_lines(center: Tuple[int, int], points: np.ndarray, length: float) -> np.ndarray:
    """(N, 2, 2) segments from each point straight away from center."""
    bearings = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    endpoints = points + length * np.stack([np.cos(bearings), np.sin(bearings)], axis=-1)
    return np.stack([points, endpoints.astype(np.int32)], axis=1).astype(np.int32)


def _fill_closest_region(fog_of_war_mask: np.ndarray, visible_cone: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    contours, _ = cv2.findContours(visible_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fog_of_war_mask
    distances = [abs(cv2.pointPolygonTest(contour, center, True)) for contour in contours]
    closest = int(np.argmin(distances))
    if distances[closest] > _MAX_VISIBLE_REGION_DIST_PX:
        return fog_of_war_mask
    return cv2.drawContours(fog_of_war_mask, contours, closest, 1, -1)


def detect_frontier_waypoints(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int, obstacles: np.ndarray,
    agent_pixel: np.ndarray, min_goal_dist_px: float,
) -> np.ndarray:
    """(row, col) of every frontier cell: free, explored, adjacent to truly-open unexplored space -- free,
    unexplored, and itself at least _OBSTACLE_MARGIN_PX from any obstacle, so a frontier is never a sliver of
    unexplored space squeezed against a wall -- and at least min_goal_dist_px from agent_pixel. Without that
    last exclusion, the agent's own starting cell can itself qualify: reveal_fog_of_war's FOV cone has its
    apex at the agent's own position, explored but surrounded by unexplored space on most sides before the
    agent has moved anywhere, which would make DiscretePlanner's closest-goal collapse to the agent's current
    position (distance 0, no real target). Direct pixel-mask check, no contour/arclength geometry: cheap, and
    immune to a frontier point ever landing on or beside an obstacle cell."""
    explored_mask = np.where(full_map > 0, explored_mask, 0).astype(np.uint8)
    explored_mask = _fill_small_unexplored(full_map, explored_mask, area_thresh)
    margin = cv2.dilate(obstacles, np.ones((2 * _OBSTACLE_MARGIN_PX + 1, 2 * _OBSTACLE_MARGIN_PX + 1), np.uint8))
    open_unexplored = (full_map > 0) & (explored_mask == 0) & (margin == 0)
    frontier = (explored_mask > 0) & (obstacles == 0) & (cv2.dilate(open_unexplored.astype(np.uint8), _ADJACENCY_KERNEL) > 0)
    rows, cols = np.nonzero(frontier)
    far_enough = np.hypot(rows - agent_pixel[0], cols - agent_pixel[1]) >= min_goal_dist_px
    return np.stack([rows[far_enough], cols[far_enough]], axis=1).astype(float)


def _fill_small_unexplored(full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int) -> np.ndarray:
    """Marks enclosed unexplored pockets smaller than area_thresh as explored, so they yield no frontier."""
    unexplored = full_map.copy()
    unexplored[explored_mask > 0] = 0
    contours, _ = cv2.findContours(unexplored, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small = [contour for contour in contours if cv2.contourArea(contour) < area_thresh and _is_fully_unexplored(unexplored, contour)]
    filled = explored_mask.copy()
    cv2.drawContours(filled, small, -1, 255, -1)
    return filled


def _is_fully_unexplored(unexplored: np.ndarray, contour: np.ndarray) -> bool:
    inside = cv2.drawContours(np.zeros_like(unexplored), [contour], 0, 1, -1).astype(bool)
    return set(unexplored[inside].tolist()) == {1}
