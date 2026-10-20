from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

from common.utils.image_memory import ImageMemory
from common.utils.sam import SamMasker
from common.utils.sam_refinement import RefinementConfig, refined
from common.utils.projection_utils import (
    DepthCamera, ResolvedObjects, downsample_points, foreground_points, instance_masks_and_classes, largest_cluster_mask, object_points, voxel_merge,
)
from common.utils.similarity_utils import (
    bounds_intersection_matrix, overlap_ratio_matrix, view_iou_matrix, visual_similarity_matrix, within_radius,
)

from detector import Detector


UNCLASSIFIED = -1


@dataclass(frozen=True)
class ConceptGraphConfig:
    """concept-graphs' mapping parameters (conceptgraph/configs/slam_pipeline/base.yaml + README overlap run)."""

    downsample_voxel_size: float = 0.025
    dbscan_eps: float = 0.1
    dbscan_min_points: int = 10
    min_points_threshold: int = 16
    min_point_num_dets: int = 2
    match_iou_thresh: float = 0.1
    obj_min_points: int = 0
    obj_min_detections: int = 3
    merge_object_thresh: float = 1.2
    merge_overlap_thresh: float = 0.5
    merge_visual_sim_thresh: float = 0.5


@dataclass(eq=False)
class ConceptGraphObject:
    """One mapped object: world points with the number of detections each was in, normalized Detic region embedding
    (CLIP text space) and detection count."""

    points: np.ndarray
    point_num_dets: np.ndarray
    region_feature: torch.Tensor
    object_num_dets: int = 1

    def merged_with(self, other: "ConceptGraphObject", voxel_size: float) -> "ConceptGraphObject":
        """concept-graphs' merge_obj2_into_obj1: fused points and detection-weighted region embedding."""
        object_num_dets = self.object_num_dets + other.object_num_dets
        region_feature = (self.region_feature * self.object_num_dets + other.region_feature * other.object_num_dets) / object_num_dets
        points, point_num_dets = voxel_merge([(self.points, self.point_num_dets), (other.points, other.point_num_dets)], voxel_size)
        return ConceptGraphObject(points, point_num_dets, F.normalize(region_feature, dim=0), object_num_dets)

    def subset(self, kept: np.ndarray) -> "ConceptGraphObject":
        return replace(self, points=self.points[kept], point_num_dets=self.point_num_dets[kept])


@dataclass(frozen=True)
class FrameDetections:
    """One frame's raw ConceptGraphObjects and the world points of all its depth pixels."""

    objects: List[ConceptGraphObject]
    view_points: np.ndarray


def project_detections_to_objects(
    depth: torch.Tensor,
    pose: Union[torch.Tensor, np.ndarray],
    instances: Instances,
    camera: DepthCamera,
    floor_height: float,
) -> FrameDetections:
    """One raw ConceptGraphObject per instance, without the floor and the walls behind it; instances must carry region_features."""
    masks, _ = instance_masks_and_classes(instances)
    camera_center = np.asarray(pose)[:3, 3]
    points = [foreground_points(object_points(depth, pose, mask, camera), floor_height, camera_center) for mask in masks]
    objects = [
        ConceptGraphObject(detection_points, np.ones(len(detection_points), dtype=int), region_feature)
        for detection_points, region_feature in zip(points, instances.region_features)
    ]
    return FrameDetections(objects, object_points(depth, pose, np.ones(np.squeeze(depth).shape, dtype=bool), camera))


def mean_view_class_probs(object_probs: Dict[int, Tuple[float, Dict[int, torch.Tensor]]]) -> Dict[int, torch.Tensor]:
    """{object id: mean of its views' class probabilities} of a Detector.classify_objects result."""
    return {object_id: torch.stack(list(frame_probs.values())).mean(dim=0) for object_id, (_, frame_probs) in object_probs.items()}


@dataclass(eq=False)
class ConceptGraphResolvedObjects(ResolvedObjects):
    """ResolvedObjects of a ConceptGraphMemory, raytraced within its voxel size."""

    voxel_size: float = ConceptGraphConfig.downsample_voxel_size

    def raytrace_radius(self) -> float:
        return self.voxel_size

    def object_logits(self, object_id: int) -> torch.Tensor:
        return self.resolved_logits[object_id].unsqueeze(0)


@dataclass(eq=False)
class ConceptGraphMemory:
    """Objects fused across frames with +, the frames they were seen in, and the resolved_objects resolve() derives from them."""

    objects: List[ConceptGraphObject] = field(default_factory=list)
    config: ConceptGraphConfig = field(default_factory=ConceptGraphConfig)
    image_memory: ImageMemory = field(default_factory=ImageMemory)
    resolved_objects: ConceptGraphResolvedObjects = field(init=False)

    def __post_init__(self) -> None:
        self.resolved_objects = self._empty_resolved()

    def __len__(self) -> int:
        return len(self.objects)

    def _empty_resolved(self) -> ConceptGraphResolvedObjects:
        return ConceptGraphResolvedObjects(self.config.downsample_voxel_size)

    def __add__(self, frame: FrameDetections) -> "ConceptGraphMemory":
        """Matches each detection to its most similar object, merging it there or adding it as a new object, then merges overlapping objects."""
        processed = [self._processed(detection) for detection in frame.objects if len(detection.points) >= max(self.config.min_points_threshold, 5)]
        if not processed:
            return self
        if self.objects:
            self._match_into_objects(processed, KDTree(downsample_points(frame.view_points, self.config.downsample_voxel_size)))
        else:
            self.objects.extend(processed)
        self.objects = self._merge_overlapping(self.objects)
        return self

    def _match_into_objects(self, detections: List[ConceptGraphObject], view_tree: KDTree) -> None:
        similarities = self._match_similarities(detections, view_tree)
        for detection, detection_similarities in zip(detections, similarities):
            best_object = int(detection_similarities.argmax())
            if detection_similarities[best_object] == 0:
                self._add_object(detection)
            else:
                self._merge_into(best_object, detection)

    def _add_object(self, detection: ConceptGraphObject) -> None:
        detection = self._without_smaller_objects_points(detection, len(detection.points), excluded=-1)
        if len(detection.points) >= self.config.min_points_threshold:
            self.objects.append(detection)

    def _merge_into(self, object_id: int, detection: ConceptGraphObject) -> None:
        detection = self._without_smaller_objects_points(detection, len(self.objects[object_id].points), excluded=object_id)
        if len(detection.points):
            self.objects[object_id] = self.objects[object_id].merged_with(detection, self.config.downsample_voxel_size)

    def _without_smaller_objects_points(self, detection: ConceptGraphObject, size: int, excluded: int) -> ConceptGraphObject:
        """detection without its points within voxel size of an object, other than excluded, with fewer than size points;
        keeps a large object detected without what lies on it from absorbing it."""
        nearby = bounds_intersection_matrix([detection.points], [obj.points for obj in self.objects], self.config.downsample_voxel_size)[0]
        trees = [
            KDTree(self.objects[object_id].points) for object_id in np.flatnonzero(nearby)
            if object_id != excluded and len(self.objects[object_id].points) < size
        ]
        return detection.subset(~self._near_any(detection.points, trees))

    def _processed(self, obj: ConceptGraphObject) -> ConceptGraphObject:
        """obj voxel-downsampled, then its largest DBSCAN cluster (concept-graphs' process_pcd)."""
        points, point_num_dets = voxel_merge([(obj.points, obj.point_num_dets)], self.config.downsample_voxel_size)
        downsampled = replace(obj, points=points, point_num_dets=point_num_dets)
        return downsampled.subset(largest_cluster_mask(points, self.config.dbscan_eps, self.config.dbscan_min_points))

    def _match_similarities(self, detections: List[ConceptGraphObject], view_tree: KDTree) -> torch.Tensor:
        """(detections, objects) IoU with the objects' parts seen in this frame + region-embedding similarity, 0 where that IoU is
        under match_iou_thresh, computed before any merge of this frame."""
        spatial_sim = torch.from_numpy(view_iou_matrix(
            [detection.points for detection in detections], [obj.points for obj in self.objects], view_tree, self.config.downsample_voxel_size,
        )).float()
        visual_sim = visual_similarity_matrix(
            torch.stack([detection.region_feature for detection in detections]), torch.stack([obj.region_feature for obj in self.objects]),
        )
        return torch.where(spatial_sim >= self.config.match_iou_thresh, spatial_sim + visual_sim, torch.zeros_like(spatial_sim))

    def resolve(self) -> None:
        """concept-graphs' end-of-run post-processing (denoise, filter, overlap merge), then points in fewer than min_point_num_dets
        detections dropped, shared points kept by the smallest object and each object re-denoised, into resolved_objects,
        every object of class UNCLASSIFIED."""
        objects = [self._denoised(obj) for obj in self.objects]
        objects = [
            obj for obj in objects
            if len(obj.points) >= self.config.obj_min_points and obj.object_num_dets >= self.config.obj_min_detections
        ]
        objects = [obj.subset(obj.point_num_dets >= self.config.min_point_num_dets) for obj in self._merge_overlapping(objects)]
        objects = self._without_shared_points(self._large_enough(objects))
        self._set_resolved([self._denoised(obj) for obj in self._large_enough(objects)])

    def _large_enough(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        return [obj for obj in objects if len(obj.points) >= self.config.min_points_threshold]

    def _denoised(self, obj: ConceptGraphObject) -> ConceptGraphObject:
        denoised = self._processed(obj)
        return denoised if len(denoised.points) >= 4 else obj

    def _merge_overlapping(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        """concept-graphs' merge_overlap_objects: merge an object into another while the overlap of the first in the second
        + their region-embedding similarity exceeds merge_object_thresh, each also above its own merge threshold."""
        if len(objects) < 2:
            return objects
        overlap = overlap_ratio_matrix([obj.points for obj in objects], [obj.points for obj in objects], self.config.downsample_voxel_size)
        features = torch.stack([obj.region_feature for obj in objects])
        visual_sim = visual_similarity_matrix(features, features).numpy()
        merge_sim = np.where((overlap > self.config.merge_overlap_thresh) & (visual_sim > self.config.merge_visual_sim_thresh), overlap + visual_sim, 0)
        np.fill_diagonal(merge_sim, 0)
        kept = np.ones(len(objects), dtype=bool)
        for source, target in zip(*np.unravel_index(np.argsort(-merge_sim, axis=None), merge_sim.shape)):
            if merge_sim[source, target] <= self.config.merge_object_thresh:
                break
            if kept[source] and kept[target]:
                objects[target] = self._processed(objects[target].merged_with(objects[source], self.config.downsample_voxel_size))
                kept[source] = False
        return [obj for obj, keep in zip(objects, kept) if keep]

    def _without_shared_points(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        """objects, each without its points within voxel size of a smaller object's."""
        objects = sorted(objects, key=lambda obj: len(obj.points))
        points = [obj.points for obj in objects]
        smaller_neighbors = np.tril(bounds_intersection_matrix(points, points, self.config.downsample_voxel_size), k=-1)
        trees = [KDTree(cloud) for cloud in points]
        return [
            obj.subset(~self._near_any(obj.points, [trees[j] for j in np.flatnonzero(neighbors)]))
            for obj, neighbors in zip(objects, smaller_neighbors)
        ]

    def _near_any(self, points: np.ndarray, trees: List[KDTree]) -> np.ndarray:
        near = np.zeros(len(points), dtype=bool)
        for tree in trees:
            near |= within_radius(tree, points, self.config.downsample_voxel_size)
        return near

    def _set_resolved(self, objects: List[ConceptGraphObject]) -> None:
        self.resolved_objects = self._empty_resolved()
        if not objects:
            return
        self.resolved_objects.resolved_points = np.concatenate([obj.points for obj in objects])
        self.resolved_objects.resolved_infos = np.concatenate([np.tile([UNCLASSIFIED, object_id], (len(obj.points), 1)) for object_id, obj in enumerate(objects)])
        self.resolved_objects.kdtree = KDTree(self.resolved_objects.resolved_points)

    def refine(self, masker: SamMasker, camera: DepthCamera, config: RefinementConfig) -> None:
        """Carves resolved_objects with SAM masks of a few diverse views of each object (sam_refinement.refined)."""
        self.resolved_objects = refined(self.resolved_objects, self.image_memory, masker, camera, config)

    def classify(self, detector: "Detector", camera: DepthCamera, min_visibility_fraction: float) -> None:
        """Gives each resolved object the mean class probabilities of its views raytraced in image_memory, dropping objects without any."""
        views = self.image_memory.object_views(self.resolved_objects, camera, min_visibility_fraction)
        self.resolved_objects = self.resolved_objects.classified(mean_view_class_probs(detector.classify_objects(views)))
