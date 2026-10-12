from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

from common.utils.projection_utils import (
    DepthCamera, ResolvedObjects, denoise, downsample_points, instance_logits, instance_masks_and_classes, object_points,
)
from common.utils.similarity_utils import overlap_ratio_matrix, visual_similarity_matrix


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
    """One mapped object: world points, normalized CLIP feature, summed class logits, detection count and (frame id, instance index) of each detection."""

    points: np.ndarray
    clip_feature: torch.Tensor
    class_logits: torch.Tensor
    num_detections: int = 1
    sources: List[Tuple[int, int]] = field(default_factory=list)

    def merged_with(self, other: "ConceptGraphObject", voxel_size: float) -> "ConceptGraphObject":
        """concept-graphs' merge_obj2_into_obj1: fused points, detection-weighted CLIP feature, summed logits."""
        num_detections = self.num_detections + other.num_detections
        clip_feature = (self.clip_feature * self.num_detections + other.clip_feature * other.num_detections) / num_detections
        return ConceptGraphObject(
            points=downsample_points(np.concatenate([self.points, other.points]), voxel_size),
            clip_feature=F.normalize(clip_feature, dim=0),
            class_logits=self.class_logits + other.class_logits,
            num_detections=num_detections,
            sources=self.sources + other.sources,
        )


def project_detections_to_objects(
    depth: torch.Tensor,
    pose: Union[torch.Tensor, np.ndarray],
    instances: Instances,
    hfov: int,
    min_depth: float,
    max_depth: float,
) -> "ConceptGraphObjectSet":
    """One raw ConceptGraphObject per instance; instances must carry clip_features, gt_logits or class_probs, and infos with a frame_id."""
    masks, _ = instance_masks_and_classes(instances)
    camera = DepthCamera(hfov, min_depth, max_depth)
    objects = [
        ConceptGraphObject(
            points=object_points(depth, pose, mask, camera),
            clip_feature=clip_feature,
            class_logits=logits,
            sources=[(info["frame_id"], instance_idx)],
        )
        for instance_idx, (mask, clip_feature, logits, info) in enumerate(zip(masks, instances.clip_features, instance_logits(instances), instances.infos))
    ]
    return ConceptGraphObjectSet(objects)


@dataclass(eq=False)
class ConceptGraphObjectSet(ResolvedObjects):
    """Objects fused across frames with +; resolve() derives resolved_points/resolved_infos/resolved_logits from them."""

    objects: List[ConceptGraphObject] = field(default_factory=list)
    config: ConceptGraphConfig = field(default_factory=ConceptGraphConfig)
    resolved_sources: Dict[int, Dict[int, int]] = field(init=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.objects)

    def raytrace_radius(self) -> float:
        return self.config.downsample_voxel_size

    def object_logits(self, object_id: int) -> torch.Tensor:
        return self.resolved_logits[object_id].unsqueeze(0)

    def source_instances(self, object_id: int) -> Dict[int, int]:
        return self.resolved_sources[object_id]

    def __add__(self, detections: "ConceptGraphObjectSet") -> "ConceptGraphObjectSet":
        """Matches each detection to its most similar object, merging it there or adding it as a new object."""
        processed = [self._processed(detection) for detection in detections.objects if len(detection.points) >= max(self.config.min_points_threshold, 5)]
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
        """(detections, objects) spatial overlap + CLIP similarity, computed before any merge of this frame."""
        spatial_sim = torch.from_numpy(overlap_ratio_matrix(
            [detection.points for detection in detections], [obj.points for obj in self.objects], self.config.downsample_voxel_size,
        )).float()
        visual_sim = visual_similarity_matrix(
            torch.stack([detection.clip_feature for detection in detections]), torch.stack([obj.clip_feature for obj in self.objects]),
        )
        return spatial_sim + visual_sim

    def resolve(self) -> None:
        """concept-graphs' end-of-run post-processing (denoise, filter, overlap merge) into the resolved_* outputs."""
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
        visual_sim = visual_similarity_matrix(torch.stack([obj.clip_feature for obj in objects]), torch.stack([obj.clip_feature for obj in objects]))
        kept = np.ones(len(objects), dtype=bool)
        for source, target in zip(*np.unravel_index(np.argsort(-overlap, axis=None), overlap.shape)):
            if overlap[source, target] <= self.config.merge_overlap_thresh:
                break
            if visual_sim[source, target] > self.config.merge_visual_sim_thresh and kept[source] and kept[target]:
                objects[target] = self._processed(objects[target].merged_with(objects[source], self.config.downsample_voxel_size))
                kept[source] = False
        return [obj for obj, keep in zip(objects, kept) if keep]

    def _set_resolved(self, objects: List[ConceptGraphObject]) -> None:
        self.resolved_points = np.concatenate([obj.points for obj in objects]) if objects else np.empty((0, 3))
        self.resolved_infos = np.concatenate([
            np.tile([int(obj.class_logits.argmax()), object_id], (len(obj.points), 1)) for object_id, obj in enumerate(objects)
        ]) if objects else np.empty((0, 2), dtype=int)
        self.resolved_logits = {object_id: obj.class_logits / obj.num_detections for object_id, obj in enumerate(objects)}
        self.resolved_sources = {object_id: dict(obj.sources) for object_id, obj in enumerate(objects)}
        self.kdtree = KDTree(self.resolved_points) if objects else None
