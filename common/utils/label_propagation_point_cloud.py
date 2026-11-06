from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cc3d
import numpy as np
import torch
from detectron2.structures import Instances
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.kdtree import KDTree
import open3d as o3d

from common.utils.concept_graph import detection_points, liftable_instances
from common.utils.image_memory import Frame
from common.utils.projection_utils import (
    DepthCamera, ResolvedObjects, denoise, instance_logits, instance_masks_and_classes, object_points,
)
from common.vision.detector import Detector

_DENOISE_VOXEL_SIZE = 0.025
_DBSCAN_EPS = 0.1
_DBSCAN_MIN_POINTS = 10


def _voxel_winning_masks(voxel_ids: np.ndarray, point_masks: np.ndarray, mask_logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """(occupied voxel ids, each one's mask with the highest logit among the masks with points in it), 3DLabelProp's "max" solution."""
    mask_scores = mask_logits.max(dim=1).values.numpy()
    order = np.lexsort((-mask_scores[point_masks], voxel_ids))
    voxels, first = np.unique(voxel_ids[order], return_index=True)
    return voxels, point_masks[order][first]


def _merge_components_sharing_masks(components: np.ndarray, class_grid: np.ndarray, voxel_ids: np.ndarray, infos: np.ndarray) -> np.ndarray:
    """Relabels components so the ones holding points of a same source mask (and its class) become one object."""
    same_class = class_grid.flat[voxel_ids] - 1 == infos[:, 0]
    point_components = components.flat[voxel_ids[same_class]]
    mask_ids = np.unique(infos[same_class, 1], return_inverse=True)[1]
    mask_to_component = coo_matrix((np.ones(len(mask_ids)), (mask_ids, point_components)), shape=(len(infos), components.max() + 1))
    _, merged = connected_components(mask_to_component.T @ mask_to_component, directed=False)
    return np.where(components > 0, merged[components] + 1, 0)


def _logits_per_component(components: np.ndarray, voxels: np.ndarray, voxel_masks: np.ndarray, mask_logits: torch.Tensor) -> dict:
    """Logits of the distinct masks winning a voxel of each connected component, keyed by component id."""
    voxel_components = components.flat[voxels]
    return {
        int(component): mask_logits[np.unique(voxel_masks[voxel_components == component])]
        for component in np.unique(voxel_components[voxel_components > 0])
    }


def labelled_point_cloud(points: List[np.ndarray], logits: torch.Tensor) -> "LabelPropagationPointCloud":
    """LabelPropagationPointCloud of each detection's denoised points, labelled with its logits' most likely class and its own object id."""
    denoised = [denoise(detection, _DENOISE_VOXEL_SIZE, _DBSCAN_EPS, _DBSCAN_MIN_POINTS) if len(detection) else detection for detection in points]
    kept = [index for index, detection in enumerate(denoised) if len(detection)]
    if not kept:
        return LabelPropagationPointCloud()
    classes = logits.argmax(dim=1).numpy()
    infos = [np.column_stack([np.full(len(denoised[index]), classes[index]), np.full(len(denoised[index]), object_id)]) for object_id, index in enumerate(kept)]
    return LabelPropagationPointCloud(
        np.concatenate([denoised[index] for index in kept]), np.concatenate(infos), {object_id: logits[index] for object_id, index in enumerate(kept)},
    )


def project_semantic_masks_to_3d(frame: Frame, instances: Instances, camera: DepthCamera) -> "LabelPropagationPointCloud":
    """LabelPropagationPointCloud of instances' masks lifted from frame, labelled with their instance_logits."""
    if len(instances) == 0:
        return LabelPropagationPointCloud()
    masks, _ = instance_masks_and_classes(instances)
    return labelled_point_cloud([object_points(frame.depth, frame.pose, mask.cpu(), camera) for mask in masks], instance_logits(instances))


def predicted_point_cloud(frame: Frame, instances: Instances, camera: DepthCamera, floor_height: float, detector: Detector) -> "LabelPropagationPointCloud":
    """LabelPropagationPointCloud of the predictions on frame lifted as ConceptGraphMemory's are, labelled with detector's class
    probabilities of their region embeddings."""
    instances = liftable_instances(frame, instances, camera)
    logits = detector.region_class_softmax(instances.region_features)
    return labelled_point_cloud(detection_points(frame, instances.pred_masks, camera, floor_height), logits)


@dataclass(eq=False)
class LabelPropagationPointCloud(ResolvedObjects):
    """Raw labelled points fused with +; resolve() derives resolved_points/resolved_infos/resolved_logits from them."""

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    infos: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    object_id_to_logits: Dict = field(default_factory=dict)
    voxel_size: float = 0.05

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points).reshape(-1, 3)
        self.infos = np.asarray(self.infos).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.points)

    def raytrace_radius(self) -> float:
        return np.sqrt(2) * self.voxel_size

    def object_logits(self, object_id: int) -> torch.Tensor:
        logits = self.resolved_logits[object_id]
        return logits[logits.max(dim=1).values.argmax()].unsqueeze(0)

    def __add__(self, x: "LabelPropagationPointCloud") -> "LabelPropagationPointCloud":
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
        """Voxelizes the raw points, gives each voxel its winning mask's class and splits voxels into connected objects."""
        if len(self.points) == 0:
            return
        voxel_ids, grid_origin, grid_shape = self._voxelize()
        mask_logits = torch.stack([self.object_id_to_logits[mask_id] for mask_id in range(len(self.object_id_to_logits))])
        voxels, voxel_masks = _voxel_winning_masks(voxel_ids, self.infos[:, 1].astype(int), mask_logits)
        class_grid = np.zeros(np.prod(grid_shape), dtype=np.int32)
        class_grid[voxels] = mask_logits[voxel_masks].argmax(dim=1).numpy() + 1
        class_grid = class_grid.reshape(grid_shape)

        components = cc3d.connected_components(class_grid, connectivity=26)
        # components = _merge_components_sharing_masks(components, class_grid, voxel_ids, self.infos)
        occupied = components > 0
        self.resolved_points = (np.argwhere(occupied) + grid_origin + 0.5) * self.voxel_size
        self.resolved_infos = np.stack([class_grid[occupied] - 1, components[occupied]], 1).astype(int)
        self.resolved_logits = _logits_per_component(components, voxels, voxel_masks, mask_logits)
        self.kdtree = KDTree(self.resolved_points)

    def save(self, idx="0") -> None:
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(self.points)
        _pcd.colors = o3d.utility.Vector3dVector(self.infos)
        o3d.io.write_point_cloud("pcd_" + str(idx) + ".ply", _pcd)
