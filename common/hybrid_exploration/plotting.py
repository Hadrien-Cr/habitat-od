"""Overlays of the hybrid agent's objects and goal map on its BGR top-down map."""
from typing import Dict, List, Tuple
from common.hybrid_exploration.object_viewpoints import ObjectBelief

import cv2
import numpy as np

_UNCERTAIN_COLOR = (0, 140, 255)
_CONFIDENT_COLOR = (255, 0, 0)
_GOAL_MAP_COLOR = (255, 144, 30)  # BGR -- dodger blue, distinct from _CONFIDENT_COLOR's pure blue
_BACKGROUND_GRAY = 100
_POINT_COLOR = (255, 255, 255)
_EXPLORED_COLOR = (127, 200, 127)
_IN_VIEW_COLOR = np.array([0, 255, 255])
_IN_VIEW_ALPHA = 0.5
_CROSS_COLOR = (0, 0, 0)
_CROSS_ARM = 3
_TEXT_COLOR = (0, 0, 0)
_TEXT_SCALE = 0.4


def draw_in_view(image: np.ndarray, in_view: np.ndarray) -> None:
    """Tints the currently viewed pixels bright yellow, keeping what's drawn under them visible."""
    mask = in_view > 0
    image[mask] = ((1 - _IN_VIEW_ALPHA) * image[mask] + _IN_VIEW_ALPHA * _IN_VIEW_COLOR).astype(np.uint8)


def draw_points(image: np.ndarray, pixels: np.ndarray, color: Tuple[int, int, int]) -> None:
    image[np.clip(pixels[:, 0], 0, image.shape[0] - 1), np.clip(pixels[:, 1], 0, image.shape[1] - 1)] = color


def plot_occupancy(explored: np.ndarray, pixel_sets: List[np.ndarray]) -> np.ndarray:
    """BGR image shaped like explored: gray, explored pixels green, every (row, col) of pixel_sets white."""
    image = np.full(explored.shape + (3,), _BACKGROUND_GRAY, dtype=np.uint8)
    image[explored > 0] = _EXPLORED_COLOR
    for pixels in pixel_sets:
        draw_points(image, pixels, _POINT_COLOR)
    return image


def draw_objects(image: np.ndarray, objects: Dict[int, ObjectBelief], confidence_thresh: float, to_pixels) -> None:
    """Each object's (row, col) points, orange while below confidence_thresh."""
    for belief in objects.values():
        pixels = to_pixels(belief.points)
        draw_points(image, pixels, _CONFIDENT_COLOR if belief.confidence >= confidence_thresh else _UNCERTAIN_COLOR)


def draw_goal_map(image: np.ndarray, goal_map: np.ndarray, radius: int = 2, goal_score: np.ndarray = None) -> None:
    """Every goal_map cell blended into a radial gradient of _GOAL_MAP_COLOR, full strength at the cell itself and
    fading to nothing by radius pixels out (matching the planner's own disk-shaped goal dilation), for visibility.
    If goal_score is given, also marks goal_map's highest-scoring pixel (ties broken by rounded mean position)
    with a black bounding cross and its score value as text."""
    distance = cv2.distanceTransform((goal_map == 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1.0 - distance / radius, 0.0, 1.0)[..., None]
    image[...] = ((1 - alpha) * image + alpha * np.array(_GOAL_MAP_COLOR)).astype(np.uint8)
    if goal_score is not None:
        _mark_peak(image, goal_map, goal_score)


def _mark_peak(image: np.ndarray, goal_map: np.ndarray, goal_score: np.ndarray) -> None:
    masked_score = np.where(goal_map != 0, goal_score, -np.inf)
    peak_value = masked_score.max()
    rows, cols = np.nonzero(masked_score == peak_value)
    row, col = int(round(rows.mean())), int(round(cols.mean()))

    image[max(row - _CROSS_ARM, 0):row + _CROSS_ARM + 1, col] = _CROSS_COLOR
    image[row, max(col - _CROSS_ARM, 0):col + _CROSS_ARM + 1] = _CROSS_COLOR
    cv2.putText(image, f"{peak_value:.2f}", (col + _CROSS_ARM + 2, row), cv2.FONT_HERSHEY_SIMPLEX,
                _TEXT_SCALE, _TEXT_COLOR, 1, cv2.LINE_AA)
