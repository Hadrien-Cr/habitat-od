"""Field-of-view reveal and frontier extraction on a navigable top-down map, ported from
naokiyokoyama/frontier_exploration (MIT, commit 8523aa8): utils/fog_of_war.py::reveal_fog_of_war and
frontier_detection.py::detect_frontier_waypoints. Maps are (row, col) uint8 masks; cv2 contours are (x, y)."""
from typing import List, Tuple

import cv2
import numpy as np
from skimage.draw import line

from common.utils.pose_utils import normalize_radians

_MAX_VISIBLE_REGION_DIST_PX = 3


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


def detect_frontier_waypoints(full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int) -> np.ndarray:
    """(row, col) midpoint of every frontier between explored_mask and the rest of full_map's navigable area."""
    explored_mask = np.where(full_map > 0, explored_mask, 0).astype(np.uint8)
    midpoints = [_frontier_midpoint(frontier) for frontier in detect_frontiers(full_map, explored_mask, area_thresh) if len(frontier) > 1]
    return np.array(midpoints)[:, ::-1] if midpoints else np.empty((0, 2))


def detect_frontiers(full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int) -> List[np.ndarray]:
    """(N, 1, 2) contour pieces of the explored area that border unexplored navigable space."""
    explored_mask = _fill_small_unexplored(full_map, explored_mask, area_thresh)
    contours, _ = cv2.findContours(explored_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    unexplored = np.where(explored_mask > 0, 0, full_map)
    unexplored = cv2.blur(np.where(unexplored > 0, 255, unexplored).astype(np.uint8), (3, 3))
    return [frontier for contour in contours for frontier in _contour_to_frontiers(_interpolate_contour(contour), unexplored)]


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


def _interpolate_contour(contour: np.ndarray) -> np.ndarray:
    """Closed contour with every pixel between consecutive vertices filled in."""
    vertices = contour.reshape(-1, 2)
    pieces = [_segment_pixels(start, end) for start, end in zip(vertices, np.roll(vertices, -1, axis=0))]
    return np.concatenate(pieces).reshape(-1, 1, 2)


def _segment_pixels(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    xs, ys = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
    return np.stack([xs, ys], axis=1)[1:]


def _contour_to_frontiers(contour: np.ndarray, unexplored_mask: np.ndarray) -> List[np.ndarray]:
    """Splits the contour at points not touching unexplored space, merging the wrap-around piece."""
    bad = np.flatnonzero(unexplored_mask[contour[:, 0, 1], contour[:, 0, 0]] == 0)
    wraps = len(bad) > 0 and bad[0] != 0 and bad[-1] < len(contour) - 2
    pieces = np.split(contour, bad)
    frontiers = [piece if idx == 0 else piece[1:] for idx, piece in enumerate(pieces) if len(piece) > 2 or (idx == 0 and wraps)]
    if len(frontiers) > 1 and wraps:
        frontiers[0] = np.concatenate((frontiers.pop(), frontiers[0]))
    return frontiers


def _frontier_midpoint(frontier: np.ndarray) -> np.ndarray:
    """(x, y) point halfway along the frontier's length."""
    points = frontier.reshape(-1, 2).astype(float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.cumsum(lengths)
    half = cumulative[-1] / 2
    idx = int(np.argmax(cumulative > half))
    before = cumulative[idx - 1] if idx > 0 else 0.0
    return points[idx] + (half - before) / lengths[idx] * (points[idx + 1] - points[idx])
