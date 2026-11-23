from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Instances
from scipy.spatial.kdtree import KDTree

from common.env_utils.visibility_utils import within_size_ranges
from common.utils.image_memory import Frame, frames_showing
from common.utils.projection_utils import (
    DepthCamera, ResolvedObjects, downsample_points, foreground_points, instance_masks_and_classes, largest_cluster_mask, main_cluster_mask,
    nearest_resized, object_points, voxel_merge,
)
from common.utils.similarity_utils import (
    bounds_intersection_matrix, overlap_ratio_matrix, view_iou_matrix, visual_similarity_matrix, within_radius,
)
from common.vision.detector import Detector, ObjectViews


UNCLASSIFIED = -1
_MIN_LIFT_FRACTION = 5e-4
_SIZE_MARGIN = 0.5


@dataclass(frozen=True)
class ConceptGraphConfig:
    """concept-graphs' fusion parameters (conceptgraph/configs/slam_pipeline/base.yaml + README overlap run), used while
    exploring (add_processed, the agent's ObjectTracker), overridden by the collection config's fusion section and recorded
    in full in its raw run's config.yaml, plus absorb_covered_fraction; min_points_threshold
    lowered from 16 so small objects (an egg spans about 15 voxels) are kept."""

    downsample_voxel_size: float = 0.025
    dbscan_eps: float = 0.1
    min_points_threshold: int = 10
    detection_dbscan_eps: float = 0.05
    detection_dbscan_min_points: int = 5
    detection_max_attached_fraction: float = 0.15
    match_iou_thresh: float = 0.1
    merge_object_thresh: float = 1.2
    merge_overlap_thresh: float = 0.5
    merge_visual_sim_thresh: float = 0.5
    absorb_covered_fraction: float = 0.5


@dataclass(frozen=True)
class ResolveConfig:
    """Parameters of the fused objects' end-of-run resolve (concept-graphs' own, plus the adjacent-part merge) and of
    keep_foreground, applied after exploration, without the simulator."""

    min_points_threshold: int = 10
    min_point_num_dets: int = 2
    obj_min_points: int = 0
    obj_min_detections: int = 3
    dbscan_eps: float = 0.1
    dbscan_min_points: int = 10
    adjacent_distance: float = 0.05
    adjacent_visual_sim_thresh: float = 0.5
    min_detection_rate: float = 0.3


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
    points = detection_points(frame, masks, camera, floor_height)
    objects = [
        ConceptGraphObject(lifted, np.ones(len(lifted), dtype=int), region_feature, float(score), frozenset({frame_id}))
        for lifted, region_feature, score in zip(points, instances.region_features, instances.scores)
    ]
    all_pixels = np.ones(np.squeeze(frame.depth).shape, dtype=bool)
    return FrameDetections(objects, object_points(frame.depth, frame.pose, all_pixels, camera))


def detection_points(frame: Frame, masks: torch.Tensor, camera: DepthCamera, floor_height: float) -> List[np.ndarray]:
    """World points of each of masks on frame, without the floor and the walls behind them."""
    camera_center = np.asarray(frame.pose)[:3, 3]
    return [foreground_points(object_points(frame.depth, frame.pose, mask, camera), floor_height, camera_center) for mask in masks]


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


def liftable_instances(frame: Frame, instances: Instances, camera: DepthCamera) -> Instances:
    """instances resized to the frame's depth, keeping those covering at least _MIN_LIFT_FRACTION of it with valid depth."""
    instances = lifted_instances(instances, np.squeeze(frame.depth).shape)
    return instances[has_enough_depth_points(frame, instances, camera)]


def predicted_objects(frame_id: int, frame: Frame, instances: Instances, camera: DepthCamera, floor_height: float) -> FrameDetections:
    """One ConceptGraphObject per prediction on frame frame_id whose mask, resized to the frame's depth, covers at least
    _MIN_LIFT_FRACTION of it with valid depth, with its Detic region embedding and objectness."""
    instances = liftable_instances(frame, instances, camera)
    if len(instances) == 0:
        return FrameDetections([], np.empty((0, 3)))
    return project_detections_to_objects(frame_id, frame, instances, camera, floor_height)


def object_size_ranges(classes: list, class_ranges: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """{class id: its class_ranges entry, each (min, max) widened to (_SIZE_MARGIN min, max / _SIZE_MARGIN)} of every class of classes having one."""
    return {
        class_id: {measure: (_SIZE_MARGIN * low, high / _SIZE_MARGIN) for measure, (low, high) in class_ranges[name].items()}
        for class_id, name in enumerate(classes) if name in class_ranges
    }


def plausible_class_probs(frame_probs: Dict[int, torch.Tensor], dimensions: np.ndarray, size_ranges: Dict[int, Dict[str, Tuple[float, float]]]) -> torch.Tensor:
    """Mean of an object's per-view class probabilities, zero for the classes whose size_ranges its dimensions fall outside."""
    probs = torch.stack(list(frame_probs.values())).mean(dim=0)
    plausible = torch.tensor([within_size_ranges(dimensions, size_ranges.get(class_id, {})) for class_id in range(len(probs))])
    return probs * plausible


def downsampled_object(obj: ConceptGraphObject, voxel_size: float) -> ConceptGraphObject:
    points, point_num_dets = voxel_merge([(obj.points, obj.point_num_dets)], voxel_size)
    return replace(obj, points=points, point_num_dets=point_num_dets)


def processed_detections(frame: FrameDetections, config: ConceptGraphConfig) -> List[ConceptGraphObject]:
    """frame's detections with at least min_points_threshold (and 5) points, each voxel-downsampled, then its largest tight
    DBSCAN cluster with the small ones attached to it: its thin parts stay, while its mask's spill onto a neighbouring
    object, which only a tight cluster separates, goes."""
    downsampled = [
        downsampled_object(detection, config.downsample_voxel_size)
        for detection in frame.objects if len(detection.points) >= max(config.min_points_threshold, 5)
    ]
    return [
        detection.subset(main_cluster_mask(
            detection.points, config.detection_dbscan_eps, config.detection_dbscan_min_points, config.dbscan_eps, config.detection_max_attached_fraction,
        ))
        for detection in downsampled
    ]


def view_tree(frame: FrameDetections, voxel_size: float) -> KDTree:
    return KDTree(downsample_points(frame.view_points, voxel_size))


@dataclass(frozen=True)
class ProcessedFrame:
    """A frame's processed_detections and its view_tree, computed once for every consumer of the frame (add_processed,
    the hybrid agent's ObjectTracker)."""

    detections: List[ConceptGraphObject]
    view_tree: KDTree


def processed_frame(frame: FrameDetections, config: ConceptGraphConfig) -> ProcessedFrame:
    return ProcessedFrame(processed_detections(frame, config), view_tree(frame, config.downsample_voxel_size))


def match_similarities(
    detections: List[ConceptGraphObject], objects: List[ConceptGraphObject], frame_view_tree: KDTree, config: ConceptGraphConfig,
) -> torch.Tensor:
    """(detections, objects) IoU with the objects' parts seen in this frame (frame_view_tree) + region-embedding similarity,
    0 where that IoU is under match_iou_thresh."""
    spatial_sim = torch.from_numpy(view_iou_matrix(
        [detection.points for detection in detections], [obj.points for obj in objects], frame_view_tree, config.downsample_voxel_size,
    )).float()
    visual_sim = visual_similarity_matrix(
        torch.stack([detection.region_feature for detection in detections]), torch.stack([obj.region_feature for obj in objects]),
    )
    return torch.where(spatial_sim >= config.match_iou_thresh, spatial_sim + visual_sim, torch.zeros_like(spatial_sim))


def add_processed(objects: List[ConceptGraphObject], frame: ProcessedFrame, config: ConceptGraphConfig) -> List[ConceptGraphObject]:
    """objects with frame's detections fused in: each matched to its most similar object, merged there or added as a new
    object, the smaller objects a merged detection mostly covers absorbed into its object, then overlapping objects merged."""
    if not frame.detections:
        return objects
    objects = list(objects)
    if objects:
        covered = _match_into_objects(objects, frame.detections, frame.view_tree, config)
        objects = _absorbed(objects, covered, config.downsample_voxel_size)
    else:
        objects.extend(frame.detections)
    return _merge_overlapping(objects, config)


def _match_into_objects(
    objects: List[ConceptGraphObject], detections: List[ConceptGraphObject], frame_view_tree: KDTree, config: ConceptGraphConfig,
) -> List[Tuple[int, int]]:
    """Merges each detection into its most similar object, or adds it as a new one; the (object, covered part) pairs of
    the merged detections (_covered_parts)."""
    similarities = match_similarities(detections, objects, frame_view_tree, config)
    covered = []
    for detection, detection_similarities in zip(detections, similarities):
        best_object = int(detection_similarities.argmax())
        if detection_similarities[best_object] == 0:
            _add_object(objects, detection, config)
        else:
            covered += [(best_object, part) for part in _covered_parts(objects, best_object, detection, config)]
            _merge_into(objects, best_object, detection, config)
    return covered


def _add_object(objects: List[ConceptGraphObject], detection: ConceptGraphObject, config: ConceptGraphConfig) -> None:
    detection = _without_smaller_objects_points(objects, detection, len(detection.points), -1, config)
    if len(detection.points) >= config.min_points_threshold:
        objects.append(detection)


def _merge_into(objects: List[ConceptGraphObject], object_id: int, detection: ConceptGraphObject, config: ConceptGraphConfig) -> None:
    detection = _without_smaller_objects_points(objects, detection, len(objects[object_id].points), object_id, config)
    if len(detection.points):
        objects[object_id] = objects[object_id].merged_with(detection, config.downsample_voxel_size)


def _without_smaller_objects_points(
    objects: List[ConceptGraphObject], detection: ConceptGraphObject, size: int, excluded: int, config: ConceptGraphConfig,
) -> ConceptGraphObject:
    """detection without its points within voxel size of an object, other than excluded, with fewer than size points;
    keeps a large object detected without what lies on it from absorbing it."""
    trees = [KDTree(objects[object_id].points) for object_id in _smaller_nearby(objects, detection, size, excluded, config.downsample_voxel_size)]
    return detection.subset(~_near_any(detection.points, trees, config.downsample_voxel_size))


def _smaller_nearby(objects: List[ConceptGraphObject], detection: ConceptGraphObject, size: int, excluded: int, voxel_size: float) -> List[int]:
    """Indices of the objects, other than excluded, with fewer than size points whose bounds come within voxel_size of detection's."""
    nearby = bounds_intersection_matrix([detection.points], [obj.points for obj in objects], voxel_size)[0]
    return [object_id for object_id in np.flatnonzero(nearby) if object_id != excluded and len(objects[object_id].points) < size]


def _covered_parts(objects: List[ConceptGraphObject], object_id: int, detection: ConceptGraphObject, config: ConceptGraphConfig) -> List[int]:
    """Indices of the objects smaller than objects[object_id] at least absorb_covered_fraction of whose points detection
    covers: parts that object was split into, while what lies on it is rarely inside its detections' masks."""
    tree, voxel_size = KDTree(detection.points), config.downsample_voxel_size
    return [
        part for part in _smaller_nearby(objects, detection, len(objects[object_id].points), object_id, voxel_size)
        if within_radius(tree, objects[part].points, voxel_size).mean() >= config.absorb_covered_fraction
    ]


def _absorbed(objects: List[ConceptGraphObject], covered: List[Tuple[int, int]], voxel_size: float) -> List[ConceptGraphObject]:
    """objects with each (object, part) pair of covered merged, part into object, wherever earlier pairs already moved them."""
    owner = np.arange(len(objects))
    for object_id, part in covered:
        object_id, part = owner[object_id], owner[part]
        if object_id != part:
            objects[object_id] = objects[object_id].merged_with(objects[part], voxel_size)
            owner[owner == part] = object_id
    return [obj for index, obj in enumerate(objects) if owner[index] == index]


def _near_any(points: np.ndarray, trees: List[KDTree], radius: float) -> np.ndarray:
    near = np.zeros(len(points), dtype=bool)
    for tree in trees:
        near |= within_radius(tree, points, radius)
    return near


def _merge_overlapping(objects: List[ConceptGraphObject], config: ConceptGraphConfig) -> List[ConceptGraphObject]:
    """concept-graphs' merge_overlap_objects: merge an object into another while the overlap of the first in the second
    + their region-embedding similarity exceeds merge_object_thresh, each also above its own merge threshold."""
    if len(objects) < 2:
        return objects
    overlap = overlap_ratio_matrix([obj.points for obj in objects], [obj.points for obj in objects], config.downsample_voxel_size)
    features = torch.stack([obj.region_feature for obj in objects])
    visual_sim = visual_similarity_matrix(features, features).numpy()
    merge_sim = np.where((overlap > config.merge_overlap_thresh) & (visual_sim > config.merge_visual_sim_thresh), overlap + visual_sim, 0)
    np.fill_diagonal(merge_sim, 0)
    kept = np.ones(len(objects), dtype=bool)
    for source, target in zip(*np.unravel_index(np.argsort(-merge_sim, axis=None), merge_sim.shape)):
        if merge_sim[source, target] <= config.merge_object_thresh:
            break
        if kept[source] and kept[target]:
            objects[target] = objects[target].merged_with(objects[source], config.downsample_voxel_size)
            kept[source] = False
    return [obj for obj, keep in zip(objects, kept) if keep]


def resolve(objects: List[ConceptGraphObject], voxel_size: float, config: ResolveConfig) -> Tuple[ResolvedObjects, List[ConceptGraphObject]]:
    """concept-graphs' end-of-run filter of the fused objects (fused at voxel_size), then points in fewer than
    min_point_num_dets detections dropped, each object denoised, shared points kept by the object detected there most
    often and the adjacent parts of a single object merged: (the resolved objects, every one of class UNCLASSIFIED, the
    fused object each resolved object id indexes)."""
    kept = [obj for obj in objects if len(obj.points) >= config.obj_min_points and obj.object_num_dets >= config.obj_min_detections]
    kept = [_denoised(obj.subset(obj.point_num_dets >= config.min_point_num_dets), voxel_size, config) for obj in kept]
    kept = _without_shared_points(_large_enough(kept, config), voxel_size)
    sources = _merge_adjacent(_large_enough(kept, config), voxel_size, config)
    return _resolved(sources, voxel_size), sources


def _large_enough(objects: List[ConceptGraphObject], config: ResolveConfig) -> List[ConceptGraphObject]:
    return [obj for obj in objects if len(obj.points) >= config.min_points_threshold]


def _denoised(obj: ConceptGraphObject, voxel_size: float, config: ResolveConfig) -> ConceptGraphObject:
    """obj voxel-downsampled, then its largest DBSCAN cluster (concept-graphs' process_pcd), obj itself if under 4 points survive."""
    downsampled = downsampled_object(obj, voxel_size)
    denoised = downsampled.subset(largest_cluster_mask(downsampled.points, config.dbscan_eps, config.dbscan_min_points))
    return denoised if len(denoised.points) >= 4 else obj


def _merge_adjacent(objects: List[ConceptGraphObject], voxel_size: float, config: ResolveConfig) -> List[ConceptGraphObject]:
    """objects with each _adjacent_parts pair merged, most similar first, until none is left."""
    pair = _adjacent_parts(objects, config)
    while pair is not None:
        merged = objects[pair[0]].merged_with(objects[pair[1]], voxel_size)
        objects = [obj for index, obj in enumerate(objects) if index not in pair] + [merged]
        pair = _adjacent_parts(objects, config)
    return objects


def _adjacent_parts(objects: List[ConceptGraphObject], config: ResolveConfig) -> Optional[Tuple[int, int]]:
    """The most visually similar pair of objects at least adjacent_visual_sim_thresh similar whose points come within
    adjacent_distance of each other, None if there is none."""
    if len(objects) < 2:
        return None
    points, features = [obj.points for obj in objects], torch.stack([obj.region_feature for obj in objects])
    visual_sim = visual_similarity_matrix(features, features).numpy()
    similar = visual_sim >= config.adjacent_visual_sim_thresh
    candidates = bounds_intersection_matrix(points, points, config.adjacent_distance) & similar
    pairs = sorted(zip(*np.nonzero(np.triu(candidates, k=1))), key=lambda pair: -visual_sim[pair])
    return next(((int(a), int(b)) for a, b in pairs if within_radius(KDTree(points[b]), points[a], config.adjacent_distance).any()), None)


def _without_shared_points(objects: List[ConceptGraphObject], voxel_size: float) -> List[ConceptGraphObject]:
    """objects, each without its points another object's detections included more often: a part keeps its points
    only where it was detected more often than the whole it belongs to."""
    points = [obj.points for obj in objects]
    neighbors = bounds_intersection_matrix(points, points, voxel_size)
    np.fill_diagonal(neighbors, False)
    trees = [KDTree(cloud) for cloud in points]
    return [
        obj.subset(~_outvoted(obj, [(trees[j], objects[j].point_num_dets) for j in np.flatnonzero(row)], voxel_size))
        for obj, row in zip(objects, neighbors)
    ]


def _outvoted(obj: ConceptGraphObject, neighbors: List[tuple], radius: float) -> np.ndarray:
    """Whether each of obj's points has, within radius, a point of a (tree, point_num_dets) neighbor detected more often."""
    outvoted = np.zeros(len(obj.points), dtype=bool)
    for tree, point_num_dets in neighbors:
        distances, nearest = tree.query(obj.points, distance_upper_bound=radius)
        found = np.isfinite(distances)
        outvoted[found] |= point_num_dets[nearest[found]] > obj.point_num_dets[found]
    return outvoted


def _resolved(objects: List[ConceptGraphObject], voxel_size: float) -> ResolvedObjects:
    """ResolvedObjects of objects, each its index as object id and of class UNCLASSIFIED, raytraced within voxel_size."""
    resolved = ResolvedObjects(voxel_size)
    if not objects:
        return resolved
    resolved.resolved_points = np.concatenate([obj.points for obj in objects])
    resolved.resolved_infos = np.concatenate([np.tile([UNCLASSIFIED, object_id], (len(obj.points), 1)) for object_id, obj in enumerate(objects)])
    resolved.kdtree = KDTree(resolved.resolved_points)
    return resolved


def keep_foreground(resolved: ResolvedObjects, sources: List[ConceptGraphObject], raytraced: dict, min_detection_rate: float) -> Dict[int, float]:
    """{object id: foreground_score} of the resolved objects it accepts, sources their fused objects and raytraced the
    frames raytraced with them (raytrace_frames); resolved is restricted to those objects."""
    scores = {object_id: foreground_score(obj, set(frames_showing(raytraced, object_id)), min_detection_rate) for object_id, obj in enumerate(sources)}
    scores = {object_id: score for object_id, score in scores.items() if score is not None}
    resolved.relabeled(dict.fromkeys(scores, UNCLASSIFIED))
    return scores


def foreground_score(obj: ConceptGraphObject, visible_frames: set, min_detection_rate: float) -> Optional[float]:
    """obj's detection rate over the frames it is visible in times its objectness, None if it is rarely detected when visible."""
    if not visible_frames:
        return None
    detection_rate = len(obj.detected_frames & visible_frames) / len(visible_frames)
    return detection_rate * obj.objectness if detection_rate >= min_detection_rate else None


def classify(
    resolved: ResolvedObjects, foreground_scores: Dict[int, float], views: ObjectViews, detector: Detector, size_ranges: Dict[int, Dict[str, Tuple[float, float]]],
) -> None:
    """Gives each resolved object the mean class probabilities of its views (object_views) times its foreground score,
    p(object) p(class | object), zero for the classes whose size_ranges its extent falls outside, dropping objects
    without any view or plausible class."""
    scores = {}
    for object_id, frame_probs in detector.classify_objects(views).items():
        dimensions = resolved.object_dimensions(object_id)
        scores[object_id] = foreground_scores[object_id] * plausible_class_probs(frame_probs, dimensions, size_ranges)
    resolved.classified({object_id: object_scores for object_id, object_scores in scores.items() if object_scores.any()})
