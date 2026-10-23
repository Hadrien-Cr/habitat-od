"""Recovery of a resolved map's retrieved views (retrieval.py's retrieved_views) from an oracle annotation, the ground truth's 3D boxes
and its RLE-encoded masks in every frame, without any mesh: dropping false positives, labelling matches with their
ground-truth class and adding views of false negatives."""
from dataclasses import dataclass, replace
from typing import Dict, List, Set

import numpy as np
import torch
from detectron2.structures import Instances
from pycocotools import mask as mask_util

from common.utils.eval_utils import GroundTruthObjects
from common.utils.image_memory import ImageMemory
from common.utils.projection_utils import yaw_bins
from common.utils.retrieval import RetrievedView, ViewSelection, greedy_coverage


@dataclass(frozen=True)
class GroundTruthView:
    frame_id: int
    object_id: int
    visibility: float
    rle: dict


@dataclass(frozen=True)
class GroundTruthAnnotation:
    """The ground truth's 3D boxes and its views in every frame."""

    objects: GroundTruthObjects
    views: List[GroundTruthView]


def encoded_masks(masks: torch.Tensor) -> List[dict]:
    """RLE of each of the (N, H, W) masks."""
    return mask_util.encode(np.asfortranarray(masks.permute(1, 2, 0).numpy().astype(np.uint8)))


def encoded_views(frame_id: int, instances: Instances) -> List[GroundTruthView]:
    """One GroundTruthView per ground-truth instance of the frame (bbsgt infos), its mask RLE-encoded."""
    rles = encoded_masks(instances.gt_masks)
    return [GroundTruthView(frame_id, info["object_id"], info["visibility_fraction"], rle) for info, rle in zip(instances.infos, rles)]


def recover_drop_fp(views: List[RetrievedView], matches: Dict[int, int]) -> List[RetrievedView]:
    """views of the resolved objects matching some ground truth (matched_ground_truth)."""
    return [view for view in views if view.object_id in matches]


def recover_gt_classes(views: List[RetrievedView], matches: Dict[int, int], ground_truth: GroundTruthObjects) -> List[RetrievedView]:
    """views, those of a resolved object matching some ground truth (matched_ground_truth) taking its class."""
    return [
        replace(view, class_id=int(ground_truth.class_ids[matches[view.object_id]])) if view.object_id in matches else view
        for view in views
    ]


def recover_add_fn(
    views: List[RetrievedView], annotation: GroundTruthAnnotation, matches: Dict[int, int], memory: ImageMemory, selection: ViewSelection,
) -> List[RetrievedView]:
    """views plus, under a negative object id (never a resolved object's), the views selection picks of every ground
    truth no resolved object matches (matched_ground_truth), labelled with its class."""
    unmatched = np.setdiff1d(np.arange(len(annotation.objects.object_ids)), list(matches.values()))
    return views + [
        _retrieved_view(memory, view, -(rank + 1), int(annotation.objects.class_ids[index]))
        for rank, index in enumerate(unmatched) for view in _selected_views(annotation, index, memory, selection)
    ]


def recovered_objects(pseudo: List[RetrievedView], views: List[RetrievedView]) -> Set[int]:
    """Ids of the objects of views whose class differs from theirs in pseudo, relabelled by recover_gt_classes, or absent
    from pseudo, added by recover_add_fn."""
    pseudo_classes = {view.object_id: view.class_id for view in pseudo}
    return {view.object_id for view in views if pseudo_classes.get(view.object_id) != view.class_id}


def _selected_views(annotation: GroundTruthAnnotation, index: int, memory: ImageMemory, selection: ViewSelection) -> List[GroundTruthView]:
    """The ground truth's views greedy_coverage picks, each covering the yaw bin of its camera around the box center."""
    object_id, center = annotation.objects.object_ids[index], annotation.objects.boxes[index].mean(axis=0)
    candidates = [view for view in annotation.views if view.object_id == object_id]
    bins = [yaw_bins(center[None], memory.get_frame(view.frame_id).pose[:3, 3], selection.num_yaw_bins) for view in candidates]
    weights = np.array([view.visibility for view in candidates]) ** selection.alpha
    return [candidates[chosen] for chosen in greedy_coverage(bins, weights, selection)]


def _retrieved_view(memory: ImageMemory, view: GroundTruthView, object_id: int, class_id: int) -> RetrievedView:
    """view's frame rgb with its decoded mask."""
    mask = torch.from_numpy(mask_util.decode(view.rle)).bool()
    return RetrievedView(memory.get_frame(view.frame_id).rgb, mask, class_id, object_id, view.frame_id)
