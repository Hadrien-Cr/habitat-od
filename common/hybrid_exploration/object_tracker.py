"""The objects HybridFrontierAgent inspects, tracked across frames from lifted detections (concept_graph.py's
FrameDetections) with ConceptGraphMemory's own per-detection denoising and match score (3D IoU with the object's part in
view + region-embedding similarity). Unlike ConceptGraphMemory, tracked objects are never merged with each other or
dropped, so each keeps its id, the key of the agent's seen sectors."""
from dataclasses import dataclass, field
from typing import Dict, List

from common.hybrid_exploration.object_viewpoints import ObjectBelief
from common.utils.concept_graph import ConceptGraphConfig, ConceptGraphObject, FrameDetections, match_similarities, processed_detections, view_tree

_FULL_CONFIDENCE_DETECTIONS = 5


@dataclass
class ObjectTracker:
    config: ConceptGraphConfig = field(default_factory=ConceptGraphConfig)
    objects: List[ConceptGraphObject] = field(default_factory=list)

    def update(self, frame: FrameDetections) -> None:
        """Merges each of frame's detections into its best-matching tracked object, tracking it as a new one (of at least
        min_points_threshold points) when none matches."""
        detections = processed_detections(frame, self.config)
        if not detections:
            return
        if not self.objects:
            self.objects.extend(detection for detection in detections if len(detection.points) >= self.config.min_points_threshold)
            return
        similarities = match_similarities(detections, self.objects, view_tree(frame, self.config.downsample_voxel_size), self.config)
        for detection, detection_similarities in zip(detections, similarities):
            best_object = int(detection_similarities.argmax())
            if detection_similarities[best_object] > 0:
                self.objects[best_object] = self.objects[best_object].merged_with(detection, self.config.downsample_voxel_size)
            elif len(detection.points) >= self.config.min_points_threshold:
                self.objects.append(detection)

    def beliefs(self) -> Dict[int, ObjectBelief]:
        """Every tracked object's points, its confidence growing linearly with its detections up to _FULL_CONFIDENCE_DETECTIONS."""
        return {
            object_id: ObjectBelief(min(obj.object_num_dets, _FULL_CONFIDENCE_DETECTIONS) / _FULL_CONFIDENCE_DETECTIONS, obj.points)
            for object_id, obj in enumerate(self.objects)
        }
