from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.kdtree import KDTree


def within_radius(tree: KDTree, points: np.ndarray, radius: float) -> np.ndarray:
    """Whether each of points lies within radius of some point of tree."""
    return tree.query(points, k=1, distance_upper_bound=radius)[0] < radius


def _bounds(points_list: List[np.ndarray]) -> np.ndarray:
    return np.array([[points.min(axis=0), points.max(axis=0)] for points in points_list]).reshape(len(points_list), 2, 3)


def bounds_intersection_matrix(points_a: List[np.ndarray], points_b: List[np.ndarray], margin: float) -> np.ndarray:
    """(i, j) = whether the axis-aligned bounds of points_a[i] and points_b[j], grown by margin, intersect."""
    bounds_a, bounds_b = _bounds(points_a), _bounds(points_b)
    return np.all((bounds_a[:, None, 0] <= bounds_b[None, :, 1] + margin) & (bounds_b[None, :, 0] <= bounds_a[:, None, 1] + margin), axis=-1)


def overlap_ratio_matrix(points_a: List[np.ndarray], points_b: List[np.ndarray], radius: float) -> np.ndarray:
    """(i, j) = fraction of points_a[i] lying within radius of some point of points_b[j] (concept-graphs' "overlap")."""
    candidates = bounds_intersection_matrix(points_a, points_b, radius)
    trees_b = {j: KDTree(points_b[j]) for j in np.flatnonzero(candidates.any(axis=0))}
    overlap = np.zeros(candidates.shape)
    for i, j in zip(*np.nonzero(candidates)):
        overlap[i, j] = within_radius(trees_b[j], points_a[i], radius).mean()
    return overlap


def _iou(points_a: np.ndarray, points_b: np.ndarray, tree_b: KDTree, radius: float) -> float:
    """IoU of two point sets, their intersection the mean count of each one's points within radius of the other's."""
    shared = (within_radius(tree_b, points_a, radius).sum() + within_radius(KDTree(points_a), points_b, radius).sum()) / 2
    return shared / (len(points_a) + len(points_b) - shared)


def view_iou_matrix(points_a: List[np.ndarray], points_b: List[np.ndarray], view_tree: KDTree, radius: float) -> np.ndarray:
    """(i, j) = IoU of points_a[i] and the part of points_b[j] seen in a view, the points within radius of view_tree's."""
    visible_b = [points[within_radius(view_tree, points, radius)] for points in points_b]
    candidates = bounds_intersection_matrix(points_a, points_b, radius) & np.array([len(points) > 0 for points in visible_b])
    trees_b = {j: KDTree(visible_b[j]) for j in np.flatnonzero(candidates.any(axis=0))}
    iou = np.zeros(candidates.shape)
    for i, j in zip(*np.nonzero(candidates)):
        iou[i, j] = _iou(points_a[i], visible_b[j], trees_b[j], radius)
    return iou


def visual_similarity_matrix(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """(i, j) = cosine similarity between features_a[i] and features_b[j]."""
    return F.normalize(features_a, dim=-1) @ F.normalize(features_b, dim=-1).T
