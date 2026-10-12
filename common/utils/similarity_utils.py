from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.kdtree import KDTree


def overlap_ratio_matrix(points_a: List[np.ndarray], points_b: List[np.ndarray], radius: float) -> np.ndarray:
    """(i, j) = fraction of points_a[i] lying within radius of some point of points_b[j] (concept-graphs' "overlap")."""
    trees_b = [KDTree(points) for points in points_b]
    return np.array([
        [(tree.query(points, k=1, distance_upper_bound=radius)[0] < radius).mean() for tree in trees_b]
        for points in points_a
    ]).reshape(len(points_a), len(points_b))


def visual_similarity_matrix(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """(i, j) = cosine similarity between features_a[i] and features_b[j]."""
    return F.normalize(features_a, dim=-1) @ F.normalize(features_b, dim=-1).T
