"""Overlays of the hybrid agent's objects and scored candidates on its BGR top-down map."""
from typing import Dict, List, Tuple

import cv2
import numpy as np

_UNCERTAIN_COLOR = (0, 140, 255)
_CONFIDENT_COLOR = (255, 0, 0)
_GOAL_COLOR = (255, 0, 255)
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


def draw_candidates(image: np.ndarray, pixels: np.ndarray, scores: np.ndarray) -> None:
    """(row, col) candidates as circles colored by score (cool, relative to the best), the chosen one ringed in magenta."""
    colors = cv2.applyColorMap((scores / max(scores.max(), 1e-9) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_COOL)[:, 0]
    for (row, col), color in zip(pixels, colors):
        cv2.circle(image, (int(col), int(row)), 4, color.tolist(), -1)
    best_row, best_col = pixels[int(np.argmax(scores))]
    cv2.circle(image, (int(best_col), int(best_row)), 9, _GOAL_COLOR, 2)
