from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Union

import cc3d
import cv2
import numpy as np
import torch
from detectron2.structures import Instances
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.kdtree import KDTree
import open3d as o3d

from common.utils.projection_utils import ResolvedObjects, instance_masks_and_classes, project_batch_masks


def solve_inconsistency(
    update_vox_ids, object_id_to_logits, object_ids, voxel_ids, solve_function
)-> list:
    results = []
    for update_vox_id in update_vox_ids:
        objects = object_ids[voxel_ids == update_vox_id]
        if len(objects) == 0:
            continue

        ids = tuple(np.unique(objects, return_counts=True)[0])

        logits = []
        for i in ids:
            obj_logits = object_id_to_logits[i].cpu()
            if len(obj_logits.shape) == 1:
                obj_logits = obj_logits.unsqueeze(0)
            logits.append(obj_logits)

        logits = torch.cat(logits)

        resolved_class, resolved_logits = solve_function(logits)
        results.append((update_vox_id, (resolved_class, resolved_logits)))

    return results


def _seal_impl(logits) -> tuple:

    if len(logits.shape) == 1:
        resolved_class = logits.argmax()
        resolved_logits = logits
    else:
        values, indexes = logits.max(0)
        resolved_logits = logits[indexes[values.argmax()]]
        resolved_class = resolved_logits.argmax()

    return resolved_class, resolved_logits.unsqueeze(0)


def _merge_components_sharing_masks(components: np.ndarray, class_grid: np.ndarray, voxel_ids: np.ndarray, infos: np.ndarray) -> np.ndarray:
    """Relabels components so the ones holding points of a same source mask (and its class) become one object."""
    same_class = class_grid.flat[voxel_ids] - 1 == infos[:, 0]
    point_components = components.flat[voxel_ids[same_class]]
    mask_ids = np.unique(infos[same_class, 1], return_inverse=True)[1]
    mask_to_component = coo_matrix((np.ones(len(mask_ids)), (mask_ids, point_components)), shape=(len(infos), components.max() + 1))
    _, merged = connected_components(mask_to_component.T @ mask_to_component, directed=False)
    return np.where(components > 0, merged[components] + 1, 0)


def _logits_per_component(components: np.ndarray, voxel_results: dict) -> dict:
    """Unique solved logits of every voxel in each connected component, keyed by component id."""
    flat_components = components.flatten()
    return {
        int(component): torch.cat([voxel_results[voxel_id][1] for voxel_id in np.flatnonzero(flat_components == component)]).unique(dim=0)
        for component in np.unique(flat_components[flat_components > 0])
    }


def project_semantic_masks_to_3d(
    depth: torch.Tensor,
    pose: Union[torch.Tensor, np.ndarray],
    instances: Instances,
    infos,
    hfov: int,
    min_depth: float,
    max_depth: float,
) -> "SemanticPointCloud":
    """
    Creates a SemanticPointCloud from the depth, pose and instance segmentation masks.
    It iterates over the instances and projects the masks to 3D points using the depth and pose.

    depth: WxH depth tensor
    pose: 4x4 pose tensor
    instances: pred or ground-truth detectron2 Instances
    """

    n_instances = len(instances)
    if n_instances == 0:
        return SemanticPointCloud()

    instance_points, instance_infos, object_id_to_logits = [], [], {}
    masks, classes = instance_masks_and_classes(instances)
    for i in range(n_instances):
        mask, cls = masks[i], classes[i]

        kernel = np.ones((5, 5), 'uint8')

        mask = torch.tensor(
            cv2.erode(mask.cpu().numpy().astype('uint8'), kernel, iterations=1),
        )
        if mask.sum() == 0:
            continue

        points = project_batch_masks(depth, pose, mask, hfov=hfov, min_depth=min_depth, max_depth=max_depth)

        points_cls = np.ones((points.shape[0], 1)) * cls.cpu().numpy()

        points_obj_ids = np.ones((points.shape[0], 1)) * infos[i]['id_object'] # distinct id (does not need consistency across frames)
        instance_points.append(np.asarray(points))
        instance_infos.append(np.hstack((points_cls, points_obj_ids)))
        object_id_to_logits[infos[i]['id_object']] = instances[i].gt_logits

    if not instance_points:
        return SemanticPointCloud()
    return SemanticPointCloud(np.concatenate(instance_points), np.concatenate(instance_infos), object_id_to_logits)


@dataclass(eq=False)
class SemanticPointCloud(ResolvedObjects):
    """Raw labelled points fused with +; resolve() derives resolved_points/resolved_infos/resolved_logits from them."""

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    infos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    object_id_to_logits: Dict = field(default_factory=dict)
    voxel_size: float = 0.05
    solve: Callable = _seal_impl

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points).reshape(-1, 3)
        self.infos = np.asarray(self.infos).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.points)

    def raytrace_radius(self) -> float:
        return np.sqrt(2) / 2 * self.voxel_size

    def object_logits(self, object_id: int) -> torch.Tensor:
        return self.solve(self.resolved_logits[object_id])[1]

    def __add__(self, x: "SemanticPointCloud") -> "SemanticPointCloud":
        offset = int(self.infos[:, 1].max()) + 1 if len(self.infos) else 0
        self.points = np.concatenate([self.points, x.points])
        self.infos = np.concatenate([self.infos, np.stack([x.infos[:, 0], x.infos[:, 1] + offset], 1)])
        self.object_id_to_logits.update({key + offset: logits for key, logits in x.object_id_to_logits.items()})
        return self

    def _voxelize(self) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Flat voxel id of every raw point on the world-aligned thr grid, plus that grid's origin and shape."""
        world_indices = np.floor(self.points / self.voxel_size).astype(int)
        grid_origin = world_indices.min(0)
        local_indices = world_indices - grid_origin
        grid_shape = tuple(local_indices.max(0) + 1)
        return np.ravel_multi_index(local_indices.T, grid_shape), grid_origin, grid_shape

    def resolve(self) -> None:
        """Voxelizes the raw points, solves each voxel's class and splits voxels into connected objects."""
        if len(self.points) == 0:
            return
        voxel_ids, grid_origin, grid_shape = self._voxelize()
        voxel_results = dict(solve_inconsistency(
            np.unique(voxel_ids), self.object_id_to_logits, self.infos[:, 1].astype(int), voxel_ids, self.solve,
        ))
        class_grid = np.zeros(np.prod(grid_shape))
        for voxel_id, (resolved_class, _) in voxel_results.items():
            class_grid[voxel_id] = resolved_class + 1
        class_grid = class_grid.reshape(grid_shape)

        components = cc3d.connected_components(class_grid, connectivity=26)
        components = _merge_components_sharing_masks(components, class_grid, voxel_ids, self.infos)
        occupied = components > 0
        self.resolved_points = (np.argwhere(occupied) + grid_origin + 0.5) * self.voxel_size
        self.resolved_infos = np.stack([class_grid[occupied] - 1, components[occupied]], 1).astype(int)
        self.resolved_logits = _logits_per_component(components, voxel_results)
        self.kdtree = KDTree(self.resolved_points)

    def save(self, idx="0") -> None:
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(self.points)
        _pcd.colors = o3d.utility.Vector3dVector(self.infos)
        o3d.io.write_point_cloud("pcd_" + str(idx) + ".ply", _pcd)
