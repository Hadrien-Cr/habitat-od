import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from detectron2.structures import Instances
from scipy import ndimage
from scipy.spatial.kdtree import KDTree

from common.env_utils.visibility_utils import mesh_visibility_fraction
from common.utils.plot_utils import plot_semantic_2d_map

log = logging.getLogger(__name__)

_FOREGROUND_DEPTH_PERCENTILE = 95
_MIN_WALL_GAP_M = 0.1
_FLOOR_MARGIN_M = 0.02
_PLANE_DISTANCE_M = 0.02
_PLANE_RANSAC_ITERATIONS = 1000
_MAX_VERTICAL_NORMAL_UP = 0.1
_MAX_WALLS = 3
_MIN_RAYTRACED_FRACTION = 2e-4
_FOOTPRINT_CLOSING_ITERATIONS = 2
TOPDOWN_MAP_SCALE = 8


@dataclass(frozen=True)
class DepthCamera:
    hfov: float = 90.0
    min_depth: float = 0.1
    max_depth: float = 10.0


def camera_intrinsics(width: int, height: int, hfov: float) -> Tuple[float, float, float, float]:
    """(fx, fy, xc, yc) of a pinhole camera with square pixels and horizontal field of view hfov degrees."""
    fx = width / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)
    fy = height / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)
    return fx, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0


def nearest_resized(images, shape: Tuple[int, int]) -> torch.Tensor:
    """(..., H, W) images resized to shape by nearest neighbour, so depth and masks resized alike keep their pixels aligned."""
    images = torch.as_tensor(images)
    flat = images.reshape(-1, 1, *images.shape[-2:]).float()
    return F.interpolate(flat, size=tuple(shape), mode="nearest").to(images.dtype).reshape(*images.shape[:-2], *shape)


def project_batch_masks(
    depth, pose, mask=None, hfov=79, min_depth=0.5, max_depth=15.0
):
    """World points of the masked depth pixels within [min_depth, max_depth] (every pixel without a mask)."""
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth)

    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)

    H = depth.shape[-2]
    W = depth.shape[-1]

    fx, fy, xc, yc = camera_intrinsics(W, H, hfov)

    device = pose.device
    if mask is not None:
        mask[depth <= min_depth] = False
        mask[depth >= max_depth] = False
        mask = mask.bool()

    ys, xs = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    xs = xs[mask].reshape(1, -1).to(device)
    ys = ys[mask].reshape(1, -1).to(device)
    depth = depth.squeeze()[mask].reshape(1, -1).to(device)

    xyz = (
        torch.cat(
            [
                (xs - xc) / fx * depth,
                -(ys - yc) / fy * depth,
                -depth,
                torch.ones_like(depth),
            ]
        )
        .to(device)
        .double()
    )
    xyz_world = torch.matmul(pose, xyz).permute(1, 0)

    return xyz_world[:, :3]


def object_points(depth, pose, mask: torch.Tensor, camera: DepthCamera) -> np.ndarray:
    """World points of one object's mask within the camera's depth range."""
    depth = torch.as_tensor(depth).squeeze()
    in_range = (depth > camera.min_depth) & (depth < camera.max_depth)
    points = project_batch_masks(depth, pose, torch.as_tensor(mask).bool() & in_range, hfov=camera.hfov, min_depth=camera.min_depth, max_depth=camera.max_depth)
    # open3d's Vector3dVector is ~3x slower on the non-contiguous array project_batch_masks' permute leaves
    return np.ascontiguousarray(points.numpy())


def viewing_angles(points: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
    """Horizontal angle, in radians, from each point to viewpoint: which side of the point the camera stands on."""
    offsets = viewpoint - points
    return np.arctan2(offsets[:, 0], offsets[:, 2])


def floor_mask(points: np.ndarray, floor_height: float) -> np.ndarray:
    """Points at most _FLOOR_MARGIN_M above floor_height."""
    return points[:, 1] <= floor_height + _FLOOR_MARGIN_M


def background_wall_mask(points: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    """Points of the walls behind points, peeled one RANSAC plane at a time while the largest one is a _wall_behind."""
    walls = np.zeros(len(points), dtype=bool)
    for _ in range(_MAX_WALLS):
        remaining = np.flatnonzero(~walls)
        wall = _wall_behind(points[remaining], camera_center) if len(remaining) > 3 else np.zeros(0, dtype=bool)
        if not wall.any():
            break
        walls[remaining[wall]] = True
    return walls


def _wall_behind(points: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    """Inliers of points' largest plane if it is vertical and its median camera distance is at least _MIN_WALL_GAP_M beyond
    the _FOREGROUND_DEPTH_PERCENTILE-th percentile of the other points', else none (a flat object, like a painting, hangs
    only a few centimetres off its wall)."""
    plane, inliers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).segment_plane(_PLANE_DISTANCE_M, 3, _PLANE_RANSAC_ITERATIONS)
    is_plane = np.zeros(len(points), dtype=bool)
    is_plane[inliers] = True
    if is_plane.all() or abs(plane[1]) > _MAX_VERTICAL_NORMAL_UP:
        return np.zeros(len(points), dtype=bool)
    distances = np.linalg.norm(points - camera_center, axis=1)
    depth_gate = np.percentile(distances[~is_plane], _FOREGROUND_DEPTH_PERCENTILE)
    return is_plane if np.median(distances[is_plane]) > depth_gate + _MIN_WALL_GAP_M else np.zeros(len(points), dtype=bool)


def foreground_points(points: np.ndarray, floor_height: float, camera_center: np.ndarray) -> np.ndarray:
    """points without the floor and the walls behind them."""
    points = points[~floor_mask(points, floor_height)]
    return points[~background_wall_mask(points, camera_center)]


def downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel_size).points)


def _voxel_reduce(points: np.ndarray, counts: np.ndarray, voxel_size: float, reduce: np.ufunc) -> Tuple[np.ndarray, np.ndarray]:
    """Points averaged and counts reduced per voxel of a world-anchored grid of voxel_size."""
    _, voxel_ids = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0, return_inverse=True)
    voxel_ids = voxel_ids.reshape(-1)
    voxel_points = np.zeros((voxel_ids.max() + 1, 3))
    np.add.at(voxel_points, voxel_ids, points)
    voxel_counts = np.zeros(voxel_ids.max() + 1, dtype=counts.dtype)
    reduce.at(voxel_counts, voxel_ids, counts)
    return voxel_points / np.bincount(voxel_ids)[:, None], voxel_counts


def voxel_merge(clouds: List[Tuple[np.ndarray, np.ndarray]], voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """(points, counts) clouds downsampled together, each voxel's count the sum over clouds of their max count in it."""
    per_cloud = [_voxel_reduce(points, counts, voxel_size, np.maximum) for points, counts in clouds]
    return _voxel_reduce(np.concatenate([points for points, _ in per_cloud]), np.concatenate([counts for _, counts in per_cloud]), voxel_size, np.add)


def dbscan_labels(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Open3D DBSCAN cluster id of every point, -1 for noise."""
    return np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).cluster_dbscan(eps=eps, min_points=min_points))


def largest_cluster_mask(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Points of the largest DBSCAN cluster, or all of them if it has fewer than 5 points."""
    return _largest_label_mask(dbscan_labels(points, eps, min_points))


def main_cluster_mask(points: np.ndarray, eps: float, min_points: int, attach_radius: float, max_attached_fraction: float) -> np.ndarray:
    """largest_cluster_mask plus the DBSCAN clusters under max_attached_fraction of its size within attach_radius of it."""
    labels = dbscan_labels(points, eps, min_points)
    main = _largest_label_mask(labels)
    if main.all():
        return main
    sizes = np.bincount(labels[labels >= 0])
    near_main = KDTree(points[main]).query(points, k=1, distance_upper_bound=attach_radius)[0] < attach_radius
    attached = [label for label in np.flatnonzero(sizes < max_attached_fraction * sizes.max()) if near_main[labels == label].any()]
    return main | np.isin(labels, attached)


def _largest_label_mask(labels: np.ndarray) -> np.ndarray:
    cluster_sizes = Counter(labels[labels >= 0])
    if not cluster_sizes or cluster_sizes.most_common(1)[0][1] < 5:
        return np.ones(len(labels), dtype=bool)
    return labels == cluster_sizes.most_common(1)[0][0]


def denoise(points: np.ndarray, voxel_size: float, eps: float, min_points: int) -> np.ndarray:
    """concept-graphs' process_pcd: voxel downsampling, then the largest DBSCAN cluster unless it has fewer than 5 points."""
    points = downsample_points(points, voxel_size)
    return points[largest_cluster_mask(points, eps, min_points)]


def save_resolved_ply(resolved: "ResolvedObjects", colors: list, out_path: str) -> None:
    """Writes resolved_points as a .ply, each point colored by its resolved class (colors in 0-255)."""
    save_labeled_ply(resolved.resolved_points, resolved.resolved_infos[:, 0], colors, out_path)


def save_labeled_ply(points: np.ndarray, labels: np.ndarray, colors: list, out_path: str) -> None:
    """Writes points as a .ply, each point colored by colors[its label] (colors in 0-255)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors)[labels] / 255.0)
    o3d.io.write_point_cloud(str(out_path), pcd)


def filled_footprint(pixels: np.ndarray, shape: tuple) -> np.ndarray:
    """Top-down mask of (N, 2) map pixels, with gaps between them closed and holes filled."""
    mask = np.zeros(shape, dtype=bool)
    mask[pixels[:, 0], pixels[:, 1]] = True
    return ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=_FOOTPRINT_CLOSING_ITERATIONS) | mask)


def estimated_semantic_topdown(resolved: "ResolvedObjects", estimated_map, classes: list, colors: list) -> np.ndarray:
    """RGB top-down map of resolved's objects, by class, over the agent's own explored-and-free belief map --
    the fully agent-side counterpart to gt_semantic_topdown, built from EstimatedMap alone (no ground truth)."""
    navigable = estimated_map.explored & ~estimated_map.obstacles.astype(bool)
    pixels = estimated_map.to_pixels(resolved.resolved_points)
    inside = ((pixels >= 0) & (pixels < np.array(navigable.shape))).all(axis=1)
    pixels, infos = pixels[inside], resolved.resolved_infos[inside]
    object_classes = dict(zip(infos[:, 1], infos[:, 0]))
    object_footprints = [
        (object_classes[object_id], filled_footprint(pixels[infos[:, 1] == object_id], navigable.shape))
        for object_id in object_classes
    ]
    image = plot_semantic_2d_map(
        navigable, object_footprints, dict(enumerate(colors)), dict(enumerate(classes)),
        meters_per_grid_pixel=estimated_map.meters_per_pixel, scale=TOPDOWN_MAP_SCALE, explored=estimated_map.explored,
    )
    return np.array(image)


def mesh_footprint_on_grid(mesh: trimesh.Trimesh, shape: tuple, to_pixels) -> np.ndarray:
    """Top-down silhouette of mesh on a (shape) grid, via to_pixels (world xyz -> row, col): the union of
    its triangles' floor projections."""
    triangles = np.unique(to_pixels(mesh.vertices)[mesh.faces][:, :, ::-1].astype(np.int32), axis=0)
    mask = np.zeros(shape, dtype=np.uint8)
    # one call per triangle: a multi-polygon fillPoly fills by parity, leaving overlaps empty
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 1) # type: ignore
    return mask > 0


def gt_semantic_topdown(
    meshes: List[Tuple[int, int, trimesh.Trimesh]], navigable: np.ndarray, to_pixels, meters_per_pixel: float, classes: list, colors: list,
) -> np.ndarray:
    """RGB top-down map of the silhouette of every (object id, class, world mesh) of meshes, by class, over the
    ground-truth navmesh (navigable, e.g. env_base.gt_navigable_mask) -- entirely ground truth, no agent state."""
    footprints = [(class_id, mesh_footprint_on_grid(mesh, navigable.shape, to_pixels)) for _, class_id, mesh in meshes]
    image = plot_semantic_2d_map(
        navigable, footprints, dict(enumerate(colors)), dict(enumerate(classes)), meters_per_grid_pixel=meters_per_pixel, scale=TOPDOWN_MAP_SCALE,
    )
    return np.array(image)


def instance_masks_and_classes(instances: Instances) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth masks/classes when present, predicted ones otherwise."""
    if instances.has("gt_masks"):
        return instances.gt_masks, instances.gt_classes
    return instances.pred_masks, instances.pred_classes


def instance_logits(instances: Instances) -> torch.Tensor:
    """(N, num_classes) ground-truth logits when present, else the Detector's class_probs without detectron2's trailing background column."""
    if instances.has("gt_logits"):
        return instances.gt_logits
    return instances.class_probs[:, :-1]


@dataclass(eq=False)
class ResolvedObjects:
    """Objects produced by a map's resolve(): per-point (class, object_id), per-object logits and a lookup tree."""

    resolved_points: np.ndarray = field(init=False, default_factory=lambda: np.empty((0, 3)))
    resolved_infos: np.ndarray = field(init=False, default_factory=lambda: np.empty((0, 2), dtype=int))
    resolved_logits: Dict = field(init=False, default_factory=dict)
    kdtree: Optional[KDTree] = field(init=False, default=None)

    def raytrace_radius(self) -> float:
        raise NotImplementedError

    def object_logits(self, object_id: int) -> torch.Tensor:
        """(1, num_classes) logits of one resolved object."""
        raise NotImplementedError

    def raytraced_point_indices(self, depth, pose, mask: torch.Tensor, camera: DepthCamera, object_id: int) -> np.ndarray:
        """Indices of the object's resolved points nearest, within raytrace_radius, to mask's depth pixels seen from pose."""
        nearest = self.kdtree.query(object_points(depth, pose, mask, camera), distance_upper_bound=self.raytrace_radius(), k=1, workers=-1)[1]
        nearest = nearest[nearest < len(self.resolved_points)]
        return np.unique(nearest[self.resolved_infos[nearest, 1] == object_id])

    def classified(self, class_probs: Dict[int, torch.Tensor]) -> "ResolvedObjects":
        """self restricted to the objects of class_probs, each taking its most probable class."""
        self.resolved_logits = class_probs
        return self.relabeled({object_id: int(probs.argmax()) for object_id, probs in class_probs.items()})

    def object_classes(self) -> Dict[int, int]:
        """{object id: class} of every resolved object."""
        return dict(zip(self.resolved_infos[:, 1].tolist(), self.resolved_infos[:, 0].tolist()))

    def relabeled(self, object_classes: Dict[int, int]) -> "ResolvedObjects":
        """self restricted to the objects of object_classes, each taking its class there."""
        self.subset(np.isin(self.resolved_infos[:, 1], list(object_classes)))
        self.resolved_infos[:, 0] = [object_classes[object_id] for object_id in self.resolved_infos[:, 1].tolist()]
        return self

    def subset(self, kept: np.ndarray) -> "ResolvedObjects":
        """self restricted to its kept points, its lookup tree rebuilt."""
        self.resolved_points, self.resolved_infos = self.resolved_points[kept], self.resolved_infos[kept]
        self.kdtree = KDTree(self.resolved_points) if len(self.resolved_points) else None
        return self

    def depth_raytracing(
        self, depth, pose, hfov, min_depth, max_depth, *, min_mesh_visibility_fraction: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masks, ids, classes and logits of the resolved objects seen in depth from pose, dropping those below min_mesh_visibility_fraction."""
        masks, ids, classes = self.visible_objects(depth, pose, hfov, min_depth, max_depth, min_mesh_visibility_fraction=min_mesh_visibility_fraction)
        if len(ids) == 0:
            return masks, ids, classes, torch.Tensor()
        return masks, ids, classes, torch.cat([self.object_logits(int(object_id)).cpu() for object_id in ids])

    def visible_objects(
        self, depth, pose, hfov, min_depth, max_depth, *, min_mesh_visibility_fraction: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masks, ids and classes of the resolved objects seen in depth from pose, dropping those below min_mesh_visibility_fraction."""
        if len(self.resolved_points) == 0:
            log.warning("no resolved points to raytrace")
            return torch.Tensor(), torch.Tensor(), torch.Tensor()
        projected_points = project_batch_masks(depth, pose, hfov=hfov, min_depth=min_depth, max_depth=max_depth).numpy()
        nearest_points = self.kdtree.query(projected_points, distance_upper_bound=self.raytrace_radius(), k=1, workers=-1)[1]
        resolved_masks, resolved_ids, resolved_classes = self._object_masks(nearest_points, depth.shape)
        kept = [
            idx for idx, (mask, object_id) in enumerate(zip(resolved_masks, resolved_ids))
            if mesh_visibility_fraction(self.object_dimensions(object_id), mask.numpy(), np.asarray(depth), hfov) >= min_mesh_visibility_fraction
        ]
        if not kept:
            return torch.Tensor(), torch.Tensor(), torch.Tensor()
        return (
            torch.stack([resolved_masks[idx] for idx in kept]),
            torch.tensor([resolved_ids[idx] for idx in kept], dtype=torch.int),
            torch.tensor([resolved_classes[idx] for idx in kept], dtype=torch.int),
        )

    def object_dimensions(self, object_id: int) -> np.ndarray:
        """World-axis-aligned extent of one resolved object's points, the stand-in for its mesh AABB in mesh_visibility_fraction."""
        return np.ptp(self.resolved_object_points(object_id), axis=0)

    def resolved_object_points(self, object_id: int) -> np.ndarray:
        return self.resolved_points[self.resolved_infos[:, 1] == object_id]

    def _object_masks(self, nearest_points: np.ndarray, image_shape) -> Tuple[list, list, list]:
        """Pixel mask, id and class of every resolved object hit by at least _MIN_RAYTRACED_FRACTION of the pixels."""
        hit = nearest_points != len(self.resolved_points)
        pixel_objects = np.full(len(nearest_points), -1)
        pixel_objects[hit] = self.resolved_infos[nearest_points[hit], 1]
        resolved_masks, resolved_ids, resolved_classes = [], [], []
        for object_id in np.unique(pixel_objects[hit]):
            mask = pixel_objects == object_id
            if mask.sum() < _MIN_RAYTRACED_FRACTION * mask.size:
                continue
            resolved_masks.append(torch.tensor(mask.reshape(*image_shape), dtype=torch.bool))
            resolved_ids.append(int(object_id))
            resolved_classes.append(int(self.resolved_infos[nearest_points[mask][0], 0]))
        return resolved_masks, resolved_ids, resolved_classes


def box_columns(points: np.ndarray) -> dict:
    """Center and size of the points' axis-aligned box."""
    low, high = points.min(axis=0), points.max(axis=0)
    return dict(zip(("xc", "yc", "zc", "w", "h", "d"), np.round(np.r_[(low + high) / 2, high - low], 3)))


def resolved_object_rows(stage: str, resolved: ResolvedObjects) -> List[dict]:
    """One row per resolved object: its id, class, points and box."""
    return [
        {"stage": stage, "object": object_id, "class_id": class_id, "num_points": len(points), **box_columns(points)}
        for object_id, class_id in resolved.object_classes().items() for points in [resolved.resolved_object_points(object_id)]
    ]
