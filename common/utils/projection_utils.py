import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

from common.env_utils.visibility_utils import mesh_visibility_fraction

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthCamera:
    hfov: float
    min_depth: float
    max_depth: float


def camera_intrinsics(width: int, height: int, hfov: float) -> Tuple[float, float, float, float]:
    """(fx, fy, xc, yc) of a pinhole camera with square pixels and horizontal field of view hfov degrees."""
    fx = width / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)
    fy = height / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)
    return fx, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0


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
    return points.numpy()


def downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel_size).points)


def denoise(points: np.ndarray, voxel_size: float, eps: float, min_points: int) -> np.ndarray:
    """concept-graphs' process_pcd: voxel downsampling, then the largest DBSCAN cluster unless it has fewer than 5 points."""
    points = downsample_points(points, voxel_size)

    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.array(o3d_pcd.cluster_dbscan(eps=eps, min_points=min_points))
    cluster_sizes = Counter(labels[labels >= 0])

    if not cluster_sizes or cluster_sizes.most_common(1)[0][1] < 5:
        return points
    return points[labels == cluster_sizes.most_common(1)[0][0]]


def save_resolved_ply(resolved: "ResolvedObjects", colors: list, out_path: str) -> None:
    """Writes resolved_points as a .ply, each point colored by its resolved class (colors in 0-255)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(resolved.resolved_points)
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors)[resolved.resolved_infos[:, 0]] / 255.0)
    o3d.io.write_point_cloud(str(out_path), pcd)


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

    def classified(self, class_probs: Dict[int, torch.Tensor]) -> "ResolvedObjects":
        """self restricted to the objects of class_probs, each taking its most probable class."""
        kept = np.isin(self.resolved_infos[:, 1], list(class_probs))
        self.resolved_points, self.resolved_infos = self.resolved_points[kept], self.resolved_infos[kept]
        object_classes = {object_id: int(probs.argmax()) for object_id, probs in class_probs.items()}
        self.resolved_infos[:, 0] = [object_classes[object_id] for object_id in self.resolved_infos[:, 1]]
        self.resolved_logits = class_probs
        self.kdtree = KDTree(self.resolved_points) if len(self.resolved_points) else None
        return self

    def depth_raytracing(
        self, depth, pose, hfov, min_depth, max_depth, *, min_visibility_fraction: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masks, ids, classes and logits of the resolved objects seen in depth from pose, dropping those below min_visibility_fraction."""
        masks, ids, classes = self.visible_objects(depth, pose, hfov, min_depth, max_depth, min_visibility_fraction=min_visibility_fraction)
        if len(ids) == 0:
            return masks, ids, classes, torch.Tensor()
        return masks, ids, classes, torch.cat([self.object_logits(int(object_id)).cpu() for object_id in ids])

    def visible_objects(
        self, depth, pose, hfov, min_depth, max_depth, *, min_visibility_fraction: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masks, ids and classes of the resolved objects seen in depth from pose, dropping those below min_visibility_fraction."""
        if len(self.resolved_points) == 0:
            log.warning("no resolved points to raytrace")
            return torch.Tensor(), torch.Tensor(), torch.Tensor()
        projected_points = project_batch_masks(depth, pose, hfov=hfov, min_depth=min_depth, max_depth=max_depth).numpy()
        nearest_points = self.kdtree.query(projected_points, distance_upper_bound=self.raytrace_radius(), k=1, workers=-1)[1]
        resolved_masks, resolved_ids, resolved_classes = self._object_masks(nearest_points, depth.shape)
        kept = [
            idx for idx, (mask, object_id) in enumerate(zip(resolved_masks, resolved_ids))
            if mesh_visibility_fraction(self.object_dimensions(object_id), mask.numpy(), np.asarray(depth), hfov) >= min_visibility_fraction
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

    def _object_masks(self, nearest_points: np.ndarray, image_shape, min_pixels: int = 100) -> Tuple[list, list, list]:
        """Pixel mask, id and class of every resolved object hit by at least min_pixels pixels."""
        hit = nearest_points != len(self.resolved_points)
        pixel_objects = np.full(len(nearest_points), -1)
        pixel_objects[hit] = self.resolved_infos[nearest_points[hit], 1]
        resolved_masks, resolved_ids, resolved_classes = [], [], []
        for object_id in np.unique(pixel_objects[hit]):
            mask = pixel_objects == object_id
            if mask.sum() < min_pixels:
                continue
            resolved_masks.append(torch.tensor(mask.reshape(*image_shape), dtype=torch.bool))
            resolved_ids.append(int(object_id))
            resolved_classes.append(int(self.resolved_infos[nearest_points[mask][0], 0]))
        return resolved_masks, resolved_ids, resolved_classes
