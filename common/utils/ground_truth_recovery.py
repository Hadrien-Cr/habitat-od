"""Recovery of a resolved map's retrieved views (retrieval.py's retrieved_views) from an oracle annotation, the ground truth's 3D boxes
and its RLE-encoded masks in every frame, without any mesh: dropping false positives, labelling matches with their
ground-truth class, giving matches their ground truth's masks and adding views of false negatives, labelled with their
ground-truth class or classified as the resolved objects are."""
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Set

import numpy as np
import torch
from detectron2.structures import BitMasks, Instances
from pycocotools import mask as mask_util

from common.env_utils.object_detector_sensors import FILTER_KEYS
from common.utils.coco_utils import encoded_masks, stacked_masks
from common.utils.concept_graph import plausible_class_probs
from common.utils.eval_utils import GroundTruthObjects
from common.utils.image_memory import Frame
from common.utils.projection_utils import viewing_angles
from common.utils.retrieval import RetrievedView, ViewSelection, greedy_coverage
from common.vision.detector import Detector


@dataclass(frozen=True)
class GroundTruthView:
    frame_id: int
    object_id: int
    class_id: int
    visibility: float
    rle: dict


@dataclass(frozen=True)
class GroundTruthAnnotation:
    """The ground truth's 3D boxes and its views in every frame."""

    objects: GroundTruthObjects
    views: List[GroundTruthView]


def encoded_views(frame_id: int, instances: Instances) -> List[GroundTruthView]:
    """One GroundTruthView per ground-truth instance of the frame (bbsgt infos), its mask RLE-encoded."""
    rles = encoded_masks(instances.gt_masks)
    return [
        GroundTruthView(frame_id, info["object_id"], class_id, info["visibility_fraction"], rle)
        for info, class_id, rle in zip(instances.infos, instances.gt_classes.tolist(), rles)
    ]


def views_by_frame(ground_truth_views: List[GroundTruthView]) -> Dict[int, List[GroundTruthView]]:
    frame_views = defaultdict(list)
    for view in ground_truth_views:
        frame_views[view.frame_id].append(view)
    return frame_views


def ground_truth_frame(views: List[GroundTruthView], image_size: tuple) -> Instances:
    """One frame's ground truth from its views (encoded_views), as sense.ground_truth_instances gave it: masks, their
    boxes, classes and object ids, every view valid (no FILTER_KEYS set); empty for a frame without views."""
    masks = stacked_masks([view.rle for view in views], image_size)
    ground_truth = Instances(image_size, gt_masks=masks, gt_boxes=BitMasks(masks).get_bounding_boxes())
    ground_truth.gt_classes = torch.tensor([view.class_id for view in views], dtype=torch.long)
    ground_truth.infos = [{"object_id": view.object_id, **dict.fromkeys(FILTER_KEYS, False)} for view in views]
    return ground_truth


def recover_drop_fp(views: List[RetrievedView], matches: Dict[int, int]) -> List[RetrievedView]:
    """views of the resolved objects matching some ground truth (matched_ground_truth)."""
    return [view for view in views if view.object_id in matches]


def recover_gt_classes(views: List[RetrievedView], matches: Dict[int, int], ground_truth: GroundTruthObjects) -> List[RetrievedView]:
    """views, those of a resolved object matching some ground truth (matched_ground_truth) taking its class."""
    return [
        replace(view, class_id=int(ground_truth.class_ids[matches[view.object_id]])) if view.object_id in matches else view
        for view in views
    ]


def recover_gt_masks(views: List[RetrievedView], matches: Dict[int, int], annotation: GroundTruthAnnotation) -> List[RetrievedView]:
    """views, those of a resolved object matching some ground truth (matched_ground_truth) taking that ground truth's mask
    in their frame, so its box too; a match's view in whose frame its ground truth shows no mask is dropped."""
    rles = {(view.frame_id, view.object_id): view.rle for view in annotation.views}
    ground_truth_ids = {object_id: int(annotation.objects.object_ids[index]) for object_id, index in matches.items()}
    keyed = [(view, (view.frame_id, ground_truth_ids.get(view.object_id))) for view in views]
    return [
        replace(view, mask=torch.from_numpy(mask_util.decode(rles[key])).bool()) if key in rles else view
        for view, key in keyed if view.object_id not in matches or key in rles
    ]


def missed_ground_truth(annotation: GroundTruthAnnotation, matches: Dict[int, int]) -> np.ndarray:
    """Indices of the ground truths no resolved object matches (matched_ground_truth)."""
    return np.setdiff1d(np.arange(len(annotation.objects.object_ids)), list(matches.values()))


def pseudo_classes(
    annotation: GroundTruthAnnotation, indices: np.ndarray, frames: Dict[int, Frame], detector: Detector, size_ranges: Dict[int, Dict[str, tuple]],
) -> Dict[int, int]:
    """{index: class} of the ground truths at indices classified as concept_graph.classify classifies resolved objects,
    from every view of theirs and their box's extent in place of their points', those without a plausible class left out."""
    indices_by_object = {int(annotation.objects.object_ids[index]): int(index) for index in indices}
    views = defaultdict(dict)
    for view in annotation.views:
        if view.object_id in indices_by_object:
            views[indices_by_object[view.object_id]][view.frame_id] = (frames[view.frame_id].rgb, mask_util.decode(view.rle))
    scores = {
        index: plausible_class_probs(frame_probs, np.ptp(annotation.objects.boxes[index], axis=0), size_ranges)
        for index, frame_probs in detector.classify_objects(views).items()
    }
    return {index: int(object_scores.argmax()) for index, object_scores in scores.items() if object_scores.any()}


def recover_add_fn(
    views: List[RetrievedView], annotation: GroundTruthAnnotation, classes: Dict[int, int], frames: Dict[int, Frame], selection: ViewSelection,
) -> List[RetrievedView]:
    """views plus, under a negative object id (never a resolved object's), the views selection picks of every ground truth
    in classes ({index: class}), labelled with its class there."""
    return views + [
        _retrieved_view(frames, view, -(index + 1), class_id)
        for index, class_id in classes.items() for view in _selected_views(annotation, index, frames, selection)
    ]


def recovered_objects(pseudo: List[RetrievedView], views: List[RetrievedView]) -> Set[int]:
    """Ids of the objects of views whose class differs from theirs in pseudo, relabelled by recover_gt_classes, or absent
    from pseudo, added by recover_add_fn."""
    pseudo_classes = {view.object_id: view.class_id for view in pseudo}
    return {view.object_id for view in views if pseudo_classes.get(view.object_id) != view.class_id}


def _selected_views(annotation: GroundTruthAnnotation, index: int, frames: Dict[int, Frame], selection: ViewSelection) -> List[GroundTruthView]:
    """The ground truth's views greedy_coverage picks, each covering the viewing angle of its camera around the box center."""
    object_id, center = annotation.objects.object_ids[index], annotation.objects.boxes[index].mean(axis=0)
    candidates = [view for view in annotation.views if view.object_id == object_id]
    elements = [
        (np.zeros(1, dtype=int), viewing_angles(center[None], frames[view.frame_id].pose[:3, 3]))
        for view in candidates
    ]
    weights = np.array([view.visibility for view in candidates]) ** selection.alpha
    return [candidates[chosen] for chosen in greedy_coverage(elements, weights, selection)]


def _retrieved_view(frames: Dict[int, Frame], view: GroundTruthView, object_id: int, class_id: int) -> RetrievedView:
    """view's frame rgb with its decoded mask."""
    mask = torch.from_numpy(mask_util.decode(view.rle)).bool()
    return RetrievedView(frames[view.frame_id].rgb, mask, class_id, object_id, view.frame_id)
