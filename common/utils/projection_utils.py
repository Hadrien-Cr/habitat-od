import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

log = logging.getLogger(__name__)


def project_batch_masks(
    depth, pose, mask=None, hfov=79, min_depth=0.5, max_depth=15.0
):
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth)

    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)

    H = depth.shape[-2]
    W = depth.shape[-1]

    fx = W / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)
    fy = H / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)

    xc = (W - 1.0) / 2.0
    yc = (H - 1.0) / 2.0

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

    # if mask is not None:
    #     xyz_world = xyz_world[_outlier_removal(depth)]

    return xyz_world[:, :3]


def _outlier_removal(points: torch.Tensor, max_deviations=1):
    mean = torch.mean(points)
    standard_deviation = torch.std(points)

    distance_from_mean = torch.norm(points - mean, dim=0)

    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return not_outlier


def instance_masks_and_classes(instances: Instances) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth masks/classes when present, predicted ones otherwise."""
    if instances.has("gt_masks"):
        return instances.gt_masks, instances.gt_classes
    return instances.pred_masks, instances.pred_classes


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

    def depth_raytracing(
        self, depth, pose, hfov, min_depth, max_depth
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.resolved_points) == 0:
            log.warning("no resolved points to raytrace")
            return torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        projected_points = project_batch_masks(depth, pose, hfov=hfov, min_depth=min_depth, max_depth=max_depth).numpy()
        nearest_points = self.kdtree.query(projected_points, distance_upper_bound=self.raytrace_radius(), k=1, workers=-1)[1]
        resolved_masks, resolved_ids, resolved_classes = self._object_masks(nearest_points, depth.shape)
        if len(resolved_masks) == 0:
            return torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        return (
            torch.stack(resolved_masks),
            torch.tensor(resolved_ids, dtype=torch.int),
            torch.tensor(resolved_classes, dtype=torch.int),
            torch.cat([self.object_logits(object_id).cpu() for object_id in resolved_ids]),
        )

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
