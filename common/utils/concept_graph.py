from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

from common.env_utils.visibility_utils import within_size_ranges
from common.hybrid_exploration.object_viewpoints import ObjectBelief
from common.utils.image_memory import Frame, ImageMemory
from common.utils.projection_utils import (
    DepthCamera, ResolvedObjects, box_columns, downsample_points, foreground_points, instance_masks_and_classes, largest_cluster_mask, main_cluster_mask,
    nearest_resized, object_points, voxel_merge,
)
from common.utils.similarity_utils import (
    bounds_intersection_matrix, overlap_ratio_matrix, view_iou_matrix, visual_similarity_matrix, within_radius,
)
from detector import Detector


UNCLASSIFIED = -1
_MIN_LIFT_FRACTION = 1e-3
_FULL_CONFIDENCE_DETECTIONS = 5
_SIZE_MARGIN = 0.5


@dataclass(frozen=True)
class ConceptGraphConfig:
    """concept-graphs' mapping parameters (conceptgraph/configs/slam_pipeline/base.yaml + README overlap run)."""

    downsample_voxel_size: float = 0.025
    dbscan_eps: float = 0.1
    dbscan_min_points: int = 10
    min_points_threshold: int = 16
    min_point_num_dets: int = 2
    detection_dbscan_eps: float = 0.05
    detection_dbscan_min_points: int = 5
    detection_max_attached_fraction: float = 0.15
    match_iou_thresh: float = 0.1
    obj_min_points: int = 0
    obj_min_detections: int = 3
    merge_object_thresh: float = 1.2
    merge_overlap_thresh: float = 0.5
    merge_visual_sim_thresh: float = 0.5
    min_detection_rate: float = 0.3
    adjacent_distance: float = 0.05
    adjacent_visual_sim_thresh: float = 0.8


@dataclass(eq=False)
class ConceptGraphObject:
    """One mapped object: world points with the number of detections each was in, normalized Detic region embedding
    (CLIP text space), detection-weighted objectness, ids of the frames it was detected in and detection count."""

    points: np.ndarray
    point_num_dets: np.ndarray
    region_feature: torch.Tensor
    objectness: float
    detected_frames: frozenset
    object_num_dets: int = 1

    def merged_with(self, other: "ConceptGraphObject", voxel_size: float) -> "ConceptGraphObject":
        """concept-graphs' merge_obj2_into_obj1: fused points, detection-weighted region embedding and objectness."""
        object_num_dets = self.object_num_dets + other.object_num_dets
        region_feature = (self.region_feature * self.object_num_dets + other.region_feature * other.object_num_dets) / object_num_dets
        objectness = (self.objectness * self.object_num_dets + other.objectness * other.object_num_dets) / object_num_dets
        points, point_num_dets = voxel_merge([(self.points, self.point_num_dets), (other.points, other.point_num_dets)], voxel_size)
        return ConceptGraphObject(
            points, point_num_dets, F.normalize(region_feature, dim=0), objectness, self.detected_frames | other.detected_frames, object_num_dets,
        )

    def subset(self, kept: np.ndarray) -> "ConceptGraphObject":
        return replace(self, points=self.points[kept], point_num_dets=self.point_num_dets[kept])


@dataclass(frozen=True)
class FrameDetections:
    """One frame's raw ConceptGraphObjects and the world points of all its depth pixels."""

    objects: List[ConceptGraphObject]
    view_points: np.ndarray


def project_detections_to_objects(
    frame_id: int,
    frame: Frame,
    instances: Instances,
    camera: DepthCamera,
    floor_height: float,
) -> FrameDetections:
    """One raw ConceptGraphObject per instance of frame frame_id, without the floor and the walls behind it; instances must
    carry region_features."""
    masks, _ = instance_masks_and_classes(instances)
    camera_center = np.asarray(frame.pose)[:3, 3]
    points = [foreground_points(object_points(frame.depth, frame.pose, mask, camera), floor_height, camera_center) for mask in masks]
    objects = [
        ConceptGraphObject(detection_points, np.ones(len(detection_points), dtype=int), region_feature, float(score), frozenset({frame_id}))
        for detection_points, region_feature, score in zip(points, instances.region_features, instances.scores)
    ]
    all_pixels = np.ones(np.squeeze(frame.depth).shape, dtype=bool)
    return FrameDetections(objects, object_points(frame.depth, frame.pose, all_pixels, camera))


def has_enough_depth_points(frame: Frame, instances: Instances, camera: DepthCamera) -> torch.Tensor:
    """Whether each of instances' masks covers at least _MIN_LIFT_FRACTION of the frame's pixels with valid depth (within the camera's range)."""
    depth = torch.as_tensor(np.asarray(frame.depth)).squeeze()
    valid = (depth > camera.min_depth) & (depth < camera.max_depth)
    return (instances.pred_masks & valid).flatten(1).sum(dim=1) >= _MIN_LIFT_FRACTION * valid.numel()


def lifted_instances(instances: Instances, shape: Tuple[int, int]) -> Instances:
    """instances with their masks and boxes resized to shape, the resolution of the depth they are lifted from."""
    lifted = Instances(shape, **instances.get_fields())
    lifted.pred_masks = nearest_resized(instances.pred_masks, shape)
    lifted.pred_boxes = instances.pred_boxes.clone()
    lifted.pred_boxes.scale(shape[1] / instances.image_size[1], shape[0] / instances.image_size[0])
    return lifted


def predicted_objects(frame_id: int, frame: Frame, instances: Instances, camera: DepthCamera, floor_height: float) -> FrameDetections:
    """One ConceptGraphObject per prediction on frame frame_id whose mask, resized to the frame's depth, covers at least
    _MIN_LIFT_FRACTION of it with valid depth, with its Detic region embedding and objectness."""
    instances = lifted_instances(instances, np.squeeze(frame.depth).shape)
    instances = instances[has_enough_depth_points(frame, instances, camera)]
    if len(instances) == 0:
        return FrameDetections([], np.empty((0, 3)))
    return project_detections_to_objects(frame_id, frame, instances, camera, floor_height)


def object_size_ranges(classes: list, class_ranges: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """{class id: its class_ranges entry, each (min, max) widened to (_SIZE_MARGIN min, max / _SIZE_MARGIN)} of every class of classes having one."""
    return {
        class_id: {measure: (_SIZE_MARGIN * low, high / _SIZE_MARGIN) for measure, (low, high) in class_ranges[name].items()}
        for class_id, name in enumerate(classes) if name in class_ranges
    }


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
    resolved_sources: List[ConceptGraphObject] = field(init=False, default_factory=list)
    foreground_scores: Dict[int, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.resolved_objects = self._empty_resolved()

    def __len__(self) -> int:
        return len(self.objects)

    def _empty_resolved(self) -> ConceptGraphResolvedObjects:
        return ConceptGraphResolvedObjects(self.config.downsample_voxel_size)

    def __add__(self, frame: FrameDetections) -> "ConceptGraphMemory":
        """Matches each detection to its most similar object, merging it there or adding it as a new object, then merges overlapping objects."""
        processed = [self._detection_processed(detection) for detection in frame.objects if len(detection.points) >= max(self.config.min_points_threshold, 5)]
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

    def _detection_processed(self, detection: ConceptGraphObject) -> ConceptGraphObject:
        """detection voxel-downsampled, then its largest tight DBSCAN cluster with the small ones attached to it: its thin parts
        stay, while its mask's spill onto a neighbouring object, which only a tight cluster separates, goes."""
        downsampled = self._downsampled(detection)
        return downsampled.subset(main_cluster_mask(
            downsampled.points, self.config.detection_dbscan_eps, self.config.detection_dbscan_min_points,
            self.config.dbscan_eps, self.config.detection_max_attached_fraction,
        ))

    def _downsampled(self, obj: ConceptGraphObject) -> ConceptGraphObject:
        points, point_num_dets = voxel_merge([(obj.points, obj.point_num_dets)], self.config.downsample_voxel_size)
        return replace(obj, points=points, point_num_dets=point_num_dets)

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
        """concept-graphs' end-of-run filter, then points in fewer than min_point_num_dets detections dropped, each object
        denoised, shared points kept by the object detected there most often and the adjacent parts of a single object
        merged, into resolved_objects, every object of class UNCLASSIFIED."""
        objects = [
            obj for obj in self.objects
            if len(obj.points) >= self.config.obj_min_points and obj.object_num_dets >= self.config.obj_min_detections
        ]
        objects = [self._denoised(obj.subset(obj.point_num_dets >= self.config.min_point_num_dets)) for obj in objects]
        objects = self._without_shared_points(self._large_enough(objects))
        self._set_resolved(self._merge_adjacent(self._large_enough(objects)))

    def _large_enough(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        return [obj for obj in objects if len(obj.points) >= self.config.min_points_threshold]

    def _denoised(self, obj: ConceptGraphObject) -> ConceptGraphObject:
        """obj voxel-downsampled, then its largest DBSCAN cluster (concept-graphs' process_pcd), obj itself if under 4 points survive."""
        downsampled = self._downsampled(obj)
        denoised = downsampled.subset(largest_cluster_mask(downsampled.points, self.config.dbscan_eps, self.config.dbscan_min_points))
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
                objects[target] = objects[target].merged_with(objects[source], self.config.downsample_voxel_size)
                kept[source] = False
        return [obj for obj, keep in zip(objects, kept) if keep]

    def _merge_adjacent(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        """objects with each _adjacent_parts pair merged, most similar first, until none is left."""
        pair = self._adjacent_parts(objects)
        while pair is not None:
            merged = objects[pair[0]].merged_with(objects[pair[1]], self.config.downsample_voxel_size)
            objects = [obj for index, obj in enumerate(objects) if index not in pair] + [merged]
            pair = self._adjacent_parts(objects)
        return objects

    def _adjacent_parts(self, objects: List[ConceptGraphObject]) -> Optional[Tuple[int, int]]:
        """The most visually similar pair of objects at least adjacent_visual_sim_thresh similar whose points come within
        adjacent_distance of each other, None if there is none."""
        if len(objects) < 2:
            return None
        points, features = [obj.points for obj in objects], torch.stack([obj.region_feature for obj in objects])
        visual_sim = visual_similarity_matrix(features, features).numpy()
        similar = visual_sim >= self.config.adjacent_visual_sim_thresh
        candidates = bounds_intersection_matrix(points, points, self.config.adjacent_distance) & similar
        pairs = sorted(zip(*np.nonzero(np.triu(candidates, k=1))), key=lambda pair: -visual_sim[pair])
        return next((
            (int(a), int(b)) for a, b in pairs if within_radius(KDTree(points[b]), points[a], self.config.adjacent_distance).any()
        ), None)

    def _without_shared_points(self, objects: List[ConceptGraphObject]) -> List[ConceptGraphObject]:
        """objects, each without its points another object's detections included more often: a part keeps its points
        only where it was detected more often than the whole it belongs to."""
        points = [obj.points for obj in objects]
        neighbors = bounds_intersection_matrix(points, points, self.config.downsample_voxel_size)
        np.fill_diagonal(neighbors, False)
        trees = [KDTree(cloud) for cloud in points]
        return [
            obj.subset(~self._outvoted(obj, [(trees[j], objects[j].point_num_dets) for j in np.flatnonzero(row)]))
            for obj, row in zip(objects, neighbors)
        ]

    def _outvoted(self, obj: ConceptGraphObject, neighbors: List[tuple]) -> np.ndarray:
        """Whether each of obj's points has, within voxel size, a point of a (tree, point_num_dets) neighbor detected more often."""
        outvoted = np.zeros(len(obj.points), dtype=bool)
        for tree, point_num_dets in neighbors:
            distances, nearest = tree.query(obj.points, distance_upper_bound=self.config.downsample_voxel_size)
            found = np.isfinite(distances)
            outvoted[found] |= point_num_dets[nearest[found]] > obj.point_num_dets[found]
        return outvoted

    def _near_any(self, points: np.ndarray, trees: List[KDTree]) -> np.ndarray:
        near = np.zeros(len(points), dtype=bool)
        for tree in trees:
            near |= within_radius(tree, points, self.config.downsample_voxel_size)
        return near

    def _set_resolved(self, objects: List[ConceptGraphObject]) -> None:
        self.resolved_objects, self.resolved_sources = self._empty_resolved(), objects
        if not objects:
            return
        self.resolved_objects.resolved_points = np.concatenate([obj.points for obj in objects])
        self.resolved_objects.resolved_infos = np.concatenate([np.tile([UNCLASSIFIED, object_id], (len(obj.points), 1)) for object_id, obj in enumerate(objects)])
        self.resolved_objects.kdtree = KDTree(self.resolved_objects.resolved_points)

    def keep_foreground(self, camera: DepthCamera, min_mesh_visibility_fraction: float) -> None:
        """Keeps the resolved objects foreground_score accepts, remembering their scores in foreground_scores."""
        raytraced = self.image_memory.raytrace_frames(self.resolved_objects, camera, min_mesh_visibility_fraction)
        scores = {
            object_id: self.foreground_score(obj, set(self.image_memory.frames_showing(object_id, raytraced)))
            for object_id, obj in enumerate(self.resolved_sources)
        }
        self.foreground_scores = {object_id: score for object_id, score in scores.items() if score is not None}
        self.resolved_objects.relabeled(dict.fromkeys(self.foreground_scores, UNCLASSIFIED))

    def foreground_score(self, obj: ConceptGraphObject, visible_frames: set) -> Optional[float]:
        """obj's detection rate over the frames it is visible in times its objectness, None if it is rarely detected when visible."""
        if not visible_frames:
            return None
        detection_rate = len(obj.detected_frames & visible_frames) / len(visible_frames)
        return detection_rate * obj.objectness if detection_rate >= self.config.min_detection_rate else None

    def classify(self, detector: Detector, camera: DepthCamera, min_mesh_visibility_fraction: float, size_ranges: Dict[int, Dict[str, Tuple[float, float]]]) -> None:
        """Gives each resolved object the mean class probabilities of its views raytraced in image_memory times its
        foreground score, p(object) p(class | object), zero for the classes whose size_ranges its extent falls outside,
        dropping objects without any view or plausible class."""
        views = self.image_memory.object_views(self.resolved_objects, camera, min_mesh_visibility_fraction)
        scores = {}
        for object_id, frame_probs in detector.classify_objects(views).items():
            probs = torch.stack(list(frame_probs.values())).mean(dim=0)
            dimensions = self.resolved_objects.object_dimensions(object_id)
            plausible = torch.tensor([within_size_ranges(dimensions, size_ranges.get(class_id, {})) for class_id in range(len(probs))])
            scores[object_id] = self.foreground_scores[object_id] * probs * plausible
        self.resolved_objects.classified({object_id: object_scores for object_id, object_scores in scores.items() if object_scores.any()})


def object_beliefs(fused: ConceptGraphMemory) -> Dict[int, ObjectBelief]:
    """Every fused object's points, its confidence growing linearly with its detections up to _FULL_CONFIDENCE_DETECTIONS."""
    return {
        object_id: ObjectBelief(min(obj.object_num_dets, _FULL_CONFIDENCE_DETECTIONS) / _FULL_CONFIDENCE_DETECTIONS, obj.points)
        for object_id, obj in enumerate(fused.objects)
    }


def fused_object_rows(stage: str, fused: ConceptGraphMemory) -> List[dict]:
    """One row per fused object: its index (which merges shift), first detection frame (which they don't), detections,
    detecting frames, objectness, points and box."""
    return [
        {
            "stage": stage, "object": index, "first_frame": min(obj.detected_frames), "num_dets": obj.object_num_dets,
            "num_frames": len(obj.detected_frames), "objectness": round(obj.objectness, 3), "num_points": len(obj.points),
            **box_columns(obj.points),
        }
        for index, obj in enumerate(fused.objects)
    ]
