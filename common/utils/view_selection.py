"""k well-visible, geometrically diverse views of a resolved object: a quality filter, then greedy insertion
maximizing the chosen views' summed quality minus diversity_weight times their summed pairwise redundancy."""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from common.env_utils.visibility_utils import mesh_visibility_fraction
from common.utils.image_memory import Frame, ImageMemory
from common.utils.projection_utils import ResolvedObjects


@dataclass(frozen=True)
class ViewGeometry:
    frame_ids: List[int]
    quality: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class ViewSelection:
    k: int
    diversity_weight: float
    bandwidth: float
    quality_filter: Callable[[np.ndarray], np.ndarray]


def above_mean_quality(quality: np.ndarray) -> np.ndarray:
    """Indices of the views whose quality is at least the mean."""
    return np.flatnonzero(quality >= quality.mean())


def view_distances(camera_centers: np.ndarray, object_center: np.ndarray) -> np.ndarray:
    """(i, j) = angle between views i and j's directions from object_center, plus |log| of their ranges' ratio."""
    offsets = camera_centers - object_center
    ranges = np.linalg.norm(offsets, axis=1)
    directions = offsets / ranges[:, None]
    angles = np.arccos(np.clip(directions @ directions.T, -1.0, 1.0))
    return angles + np.abs(np.log(ranges[:, None] / ranges[None, :]))


def redundancy_matrix(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """exp(-(distance / bandwidth) ** 2), 0 on the diagonal so a view isn't redundant with itself."""
    redundancy = np.exp(-(distances / bandwidth) ** 2)
    np.fill_diagonal(redundancy, 0.0)
    return redundancy


def greedy_views(quality: np.ndarray, redundancy: np.ndarray, k: int, diversity_weight: float) -> np.ndarray:
    """Adds, k times, the view of best quality net of its redundancy with those already chosen."""
    chosen = []
    load = np.zeros(len(quality))
    for _ in range(min(k, len(quality))):
        gains = quality - diversity_weight * load
        gains[chosen] = -np.inf
        best = int(np.argmax(gains))
        chosen.append(best)
        load += redundancy[:, best]
    return np.array(chosen, dtype=int)


def object_view_geometry(fused: ResolvedObjects, memory: ImageMemory, raytraced: dict, object_id: int, hfov: float) -> ViewGeometry:
    """The frames raytraced (ImageMemory.raytrace_frames) to show the resolved object, their visibility and view_distances."""
    frame_ids = memory.frames_showing(object_id, raytraced)
    quality = np.array([_visibility(fused, memory.get_frame(frame_id), raytraced[frame_id], object_id, hfov) for frame_id in frame_ids])
    camera_centers = np.array([memory.get_frame(frame_id).pose[:3, 3] for frame_id in frame_ids])
    return ViewGeometry(frame_ids, quality, view_distances(camera_centers, fused.resolved_object_points(object_id).mean(axis=0)))


def select_views(views: ViewGeometry, selection: ViewSelection) -> List[int]:
    """Ids of the frames greedy_views picks among those selection's quality_filter keeps."""
    kept = selection.quality_filter(views.quality)
    redundancy = redundancy_matrix(views.distances[np.ix_(kept, kept)], selection.bandwidth)
    picked = kept[greedy_views(views.quality[kept], redundancy, selection.k, selection.diversity_weight)]
    return [views.frame_ids[index] for index in picked]


def _visibility(fused: ResolvedObjects, frame: Frame, frame_raytraced: tuple, object_id: int, hfov: float) -> float:
    """mesh_visibility_fraction of the object's raytraced mask, capped at 1 as its box-over-face ratio can exceed it."""
    masks, ids, _ = frame_raytraced
    mask = masks[ids == object_id][0].numpy()
    return min(mesh_visibility_fraction(fused.object_dimensions(object_id), mask, np.asarray(frame.depth), hfov), 1.0)
