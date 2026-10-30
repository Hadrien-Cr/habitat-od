"""Overlays of the hybrid agent's objects and goal map on its BGR top-down map."""
from typing import Dict, List, Tuple

import cv2
import numpy as np

_UNCERTAIN_COLOR = (0, 140, 255)
_CONFIDENT_COLOR = (255, 0, 0)
_GOAL_MAP_COLOR = (255, 144, 30)  # BGR -- dodger blue, distinct from _CONFIDENT_COLOR's pure blue
_GOAL_MAP_DILATION_SELEM = np.ones((5, 5), dtype=np.uint8)  # goal cells are otherwise single pixels, too faint to see
_BACKGROUND_GRAY = 100
_POINT_COLOR = (255, 255, 255)
_EXPLORED_COLOR = (127, 200, 127)
_IN_VIEW_COLOR = np.array([0, 255, 255])
_IN_VIEW_ALPHA = 0.5


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


def draw_objects(image: np.ndarray, objects: Dict[int, Tuple[np.ndarray, float]], confidence_thresh: float) -> None:
    """Each object's (row, col) points, orange while below confidence_thresh."""
    for pixels, confidence in objects.values():
        draw_points(image, pixels, _CONFIDENT_COLOR if confidence >= confidence_thresh else _UNCERTAIN_COLOR)


def draw_goal_map(image: np.ndarray, goal_map: np.ndarray) -> None:
    """Every goal_map cell, dilated for visibility, drawn as a single flat color -- no per-candidate score to color
    by anymore."""
    dilated = cv2.dilate(goal_map, _GOAL_MAP_DILATION_SELEM, iterations=1)
    draw_points(image, np.argwhere(dilated), _GOAL_MAP_COLOR)
