from dataclasses import dataclass, field, replace
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

from common.utils.projection_utils import (
    DepthCamera, ResolvedObjects, denoise, downsample_points, foreground_mask, instance_masks_and_classes, object_points, surface_labels,
)
from common.utils.similarity_utils import overlap_ratio_matrix, visual_similarity_matrix


UNCLASSIFIED = -1


@dataclass(frozen=True)
class ConceptGraphConfig:
    """concept-graphs' mapping parameters (conceptgraph/configs/slam_pipeline/base.yaml + README overlap run)."""

    downsample_voxel_size: float = 0.025
    dbscan_eps: float = 0.1
    dbscan_min_points: int = 10
    min_points_threshold: int = 16
    sim_threshold: float = 1.2
    obj_min_points: int = 0
    obj_min_detections: int = 3
    merge_overlap_thresh: float = 0.7
    merge_visual_sim_thresh: float = 0.7


@dataclass(eq=False)
class ConceptGraphObject:
    """One mapped object: world points, normalized Detic region embedding (CLIP text space) and detection count."""

    points: np.ndarray
    region_feature: torch.Tensor
    num_detections: int = 1

    def merged_with(self, other: "ConceptGraphObject", voxel_size: float) -> "ConceptGraphObject":
        """concept-graphs' merge_obj2_into_obj1: fused points and detection-weighted region embedding."""
        num_detections = self.num_detections + other.num_detections
        region_feature = (self.region_feature * self.num_detections + other.region_feature * other.num_detections) / num_detections
        return ConceptGraphObject(
            points=downsample_points(np.concatenate([self.points, other.points]), voxel_size),
            region_feature=F.normalize(region_feature, dim=0),
            num_detections=num_detections,
        )


def project_detections_to_objects(
    depth: torch.Tensor,
    pose: Union[torch.Tensor, np.ndarray],
    instances: Instances,
    hfov: int,
    min_depth: float,
    max_depth: float,
) -> List[ConceptGraphObject]:
    """One raw ConceptGraphObject per instance; instances must carry region_features."""
    masks, _ = instance_masks_and_classes(instances)
    camera = DepthCamera(hfov, min_depth, max_depth)
    surfaces = surface_labels(depth, camera)
    return [
        ConceptGraphObject(
            points=object_points(depth, pose, foreground_mask(depth, mask, surfaces), camera),
            region_feature=region_feature,
        )
        for mask, region_feature in zip(masks, instances.region_features)
    ]


@dataclass(eq=False)
class ConceptGraphObjectSet(ResolvedObjects):
    """Objects fused across frames with +; resolve() derives resolved_points/resolved_infos from them, left unclassified for Detector.classify_objects."""

    objects: List[ConceptGraphObject] = field(default_factory=list)
    config: ConceptGraphConfig = field(default_factory=ConceptGraphConfig)

    def __len__(self) -> int:
        return len(self.objects)

    def raytrace_radius(self) -> float:
        return self.config.downsample_voxel_size

    def object_logits(self, object_id: int) -> torch.Tensor:
        return self.resolved_logits[object_id].unsqueeze(0)

    def __add__(self, detections: List[ConceptGraphObject]) -> "ConceptGraphObjectSet":
        """Matches each detection to its most similar object, merging it there or adding it as a new object."""
        processed = [self._processed(detection) for detection in detections if len(detection.points) >= max(self.config.min_points_threshold, 5)]
        if not processed or not self.objects:
            self.objects.extend(processed)
            return self
        similarities = self._match_similarities(processed)
        for detection, detection_similarities in zip(processed, similarities):
            best_object = int(detection_similarities.argmax())
            if detection_similarities[best_object] < self.config.sim_threshold:
                self.objects.append(detection)
                continue
            self.objects[best_object] = self.objects[best_object].merged_with(detection, self.config.downsample_voxel_size)
        return self

    def _processed(self, detection: ConceptGraphObject) -> ConceptGraphObject:
        return replace(detection, points=denoise(detection.points, self.config.downsample_voxel_size, self.config.dbscan_eps, self.config.dbscan_min_points))

    def _match_similarities(self, detections: List[ConceptGraphObject]) -> torch.Tensor:
        """(detections, objects) spatial overlap, in whichever direction is larger, + region-embedding similarity, computed before any merge of this frame."""
        detection_points, object_points = [detection.points for detection in detections], [obj.points for obj in self.objects]
        radius = self.config.downsample_voxel_size
        spatial_sim = torch.from_numpy(np.maximum(
            overlap_ratio_matrix(detection_points, object_points, radius), overlap_ratio_matrix(object_points, detection_points, radius).T,
        )).float()
        visual_sim = visual_similarity_matrix(
            torch.stack([detection.region_feature for detection in detections]), torch.stack([obj.region_feature for obj in self.objects]),
        )
        return spatial_sim + visual_sim

    def resolve(self) -> None:
        """concept-graphs' end-of-run post-processing (denoise, filter, overlap merge) into the resolved_* outputs, every object of class UNCLASSIFIED."""
        objects = [self._denoised(obj) for obj in self.objects]
        objects = [
            obj for obj in objects
            if len(obj.points) >= self.config.obj_min_points and obj.num_detections >= self.config.obj_min_detections
        ]
        self._set_resolved(self._merge_overlapping(objects))

    def _denoised(self, obj: ConceptGraphObject) -> ConceptGraphObject:
        denoised = self._processed(obj)
        return denoised if len(denoised.points) >= 4 else obj

    def _merge_overlapping(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        """concept-graphs' merge_overlap_objects: merge an object into another it mostly lies within, if they look alike."""
        if len(objects) < 2:
            return objects
        overlap = overlap_ratio_matrix([obj.points for obj in objects], [obj.points for obj in objects], self.config.downsample_voxel_size)
        np.fill_diagonal(overlap, 0)
        visual_sim = visual_similarity_matrix(torch.stack([obj.region_feature for obj in objects]), torch.stack([obj.region_feature for obj in objects]))
        kept = np.ones(len(objects), dtype=bool)
        for source, target in zip(*np.unravel_index(np.argsort(-overlap, axis=None), overlap.shape)):
            if overlap[source, target] <= self.config.merge_overlap_thresh:
                break
            if visual_sim[source, target] > self.config.merge_visual_sim_thresh and kept[source] and kept[target]:
                objects[target] = self._processed(objects[target].merged_with(objects[source], self.config.downsample_voxel_size))
                kept[source] = False
        return [obj for obj, keep in zip(objects, kept) if keep]

    def _set_resolved(self, objects: List[ConceptGraphObject]) -> None:
        if not objects:
            return
        self.resolved_points = np.concatenate([obj.points for obj in objects])
        self.resolved_infos = np.concatenate([np.tile([UNCLASSIFIED, object_id], (len(obj.points), 1)) for object_id, obj in enumerate(objects)])
        self.kdtree = KDTree(self.resolved_points)
