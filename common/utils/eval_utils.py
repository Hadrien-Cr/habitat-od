import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from detectron2.utils.logger import create_small_table
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.utils.projection_utils import ResolvedObjects, ground_truth_topdown, save_resolved_ply, semantic_topdown
from common.utils.retrieval import SegmentationStyle

_IOU_THRESHOLDS = np.round(np.arange(0.25, 0.951, 0.05), 2)
_SMALL_MAX_VOLUME_M3 = 0.01
_MEDIUM_MAX_VOLUME_M3 = 0.25
_WIREFRAME_RADIUS_M = 0.005
_PREDICTED_BOX_GRID_M = 0.02
_MIN_MATCHED_EXTENT_M = 0.05
_VOLUME_RANGES = [[0.0, np.inf], [0.0, _SMALL_MAX_VOLUME_M3], [_SMALL_MAX_VOLUME_M3, _MEDIUM_MAX_VOLUME_M3], [_MEDIUM_MAX_VOLUME_M3, np.inf]]


def _xywh_to_xyxy(box: list) -> list:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _box_volumes(boxes: np.ndarray) -> np.ndarray:
    """Area (2D) or volume (3D) of (N, 2, D) axis-aligned (min, max) boxes."""
    return np.prod(boxes[:, 1] - boxes[:, 0], axis=-1)


def _snap_boxes_outward(boxes: np.ndarray, grid: float) -> np.ndarray:
    """(N, 2, D) (min, max) boxes with min corners floored and max corners ceiled to multiples of grid."""
    return np.stack([np.floor(boxes[:, 0] / grid), np.ceil(boxes[:, 1] / grid)], axis=1) * grid


def _box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(len(boxes_a), len(boxes_b)) IoU of (N, 2, D) axis-aligned (min, max) boxes."""
    lower = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    upper = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter = np.clip(upper - lower, 0.0, None).prod(axis=-1)
    union = _box_volumes(boxes_a)[:, None] + _box_volumes(boxes_b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _with_min_extent(boxes: np.ndarray, min_extent: float) -> np.ndarray:
    """(N, 2, D) (min, max) boxes grown about their centers to at least min_extent along every axis."""
    padding = np.clip(min_extent - (boxes[:, 1] - boxes[:, 0]), 0.0, None) / 2
    return np.stack([boxes[:, 0] - padding, boxes[:, 1] + padding], axis=1)


def _box_iou_3d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """_box_iou of 3D boxes grown to _MIN_MATCHED_EXTENT_M, so thin objects (a pencil, a painting) can still overlap."""
    return _box_iou(_with_min_extent(boxes_a, _MIN_MATCHED_EXTENT_M), _with_min_extent(boxes_b, _MIN_MATCHED_EXTENT_M))


def _greedy_matches(ious: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Ground truth (column of ious) each score-sorted prediction (row) greedily, class-agnostically matches, -1 for none."""
    matches = np.full(len(ious), -1)
    matched = np.zeros(ious.shape[1], dtype=bool)
    for pred_index, pred_ious in enumerate(ious):
        unmatched_ious = np.where(matched, 0.0, pred_ious)
        best = int(unmatched_ious.argmax()) if len(unmatched_ious) else -1
        if best >= 0 and unmatched_ious[best] >= iou_threshold:
            matched[best] = True
            matches[pred_index] = best
    return matches


def _add_matches(matrix: np.ndarray, gt_classes: np.ndarray, pred_classes: np.ndarray, matches: np.ndarray) -> None:
    """Adds to matrix each prediction under its _greedy_matches ground truth's class, or background, and each unmatched ground truth."""
    background = len(matrix) - 1
    for pred_class, match in zip(pred_classes, matches):
        matrix[background if match < 0 else gt_classes[match], pred_class] += 1
    np.add.at(matrix[:, background], np.delete(gt_classes, matches[matches >= 0]), 1)


def _save_confusion_heatmap(matrix: np.ndarray, labels: list, out_path: Path) -> None:
    """Row-normalized heatmap of matrix, annotated with its raw counts."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 2, 0.5 * len(labels) + 2))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i, j in zip(*np.nonzero(matrix)):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6,
                color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="fraction of GT class's instances (row-normalized)")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_confusion_matrix(
    predictions_json: Path,
    gt_json: Path,
    out_path: Path,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    skip_unseen_classes: bool = True,
) -> np.ndarray:
    """Builds a class-confusion matrix from a COCOEvaluator `coco_instances_results.json`
    (predictions, `category_id`s already unmapped to match the GT json's own ids -- see
    `COCOEvaluator._eval_predictions`) against the GT `coco_writer.build_coco_dataset` json
    it was scored against. Saves a heatmap PNG to `out_path` and returns the raw
    (num_classes + 1, num_classes + 1) count matrix (last row/col is "background": an
    unmatched GT is a false negative in its class's background column, an unmatched
    prediction is a false positive in the background row).

    Matching is class-agnostic (IoU-only, greedy by descending score) so misclassifications
    land off-diagonal instead of being silently treated as a missed detection plus an
    unrelated false positive.

    `skip_unseen_classes` drops categories with zero GT annotations from both axes (e.g. a
    per-scene val set that only covers a handful of the full vocab -- see
    dataset.py::build_dataset's mode="separate") instead of leaving them as empty rows/columns;
    predictions of a dropped class are then out of scope too (there's no row to attribute them
    to) and are excluded rather than folded into "background"."""
    with open(gt_json) as f:
        gt = json.load(f)
    with open(predictions_json) as f:
        predictions = json.load(f)

    categories = sorted(gt["categories"], key=lambda c: c["name"])

    if skip_unseen_classes:
        seen_ids = {ann["category_id"] for ann in gt["annotations"]}
        categories = [c for c in categories if c["id"] in seen_ids]
        
    class_names = [c["name"] for c in categories]
    id_to_index = {c["id"]: i for i, c in enumerate(categories)}
    n = len(class_names)

    gt_by_image: dict[int, list[tuple[int, list]]] = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append((id_to_index[ann["category_id"]], _xywh_to_xyxy(ann["bbox"])))

    preds_by_image: dict[int, list[tuple[int, list, float]]] = {}
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        if pred["category_id"] not in id_to_index:
            continue  # a class dropped by skip_unseen_classes -- no row to attribute this to
        preds_by_image.setdefault(pred["image_id"], []).append(
            (id_to_index[pred["category_id"]], _xywh_to_xyxy(pred["bbox"]), pred["score"])
        )

    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)

    for image_id in set(gt_by_image) | set(preds_by_image):
        gts = gt_by_image.get(image_id, [])
        preds = sorted(preds_by_image.get(image_id, []), key=lambda p: -p[2])
        ious = _box_iou(np.reshape([box for _, box, _ in preds], (-1, 2, 2)), np.reshape([box for _, box in gts], (-1, 2, 2)))
        _add_matches(matrix, np.array([c for c, _ in gts], dtype=int), [c for c, _, _ in preds], _greedy_matches(ious, iou_threshold))

    _save_confusion_heatmap(matrix, class_names + ["background"], out_path)
    return matrix


@dataclass(frozen=True)
class SceneObjects:
    """Scene objects as class ids and world-axis-aligned (N, 2, 3) (min, max) boxes."""

    class_ids: np.ndarray
    boxes: np.ndarray

    def coco_annotations(self) -> List[dict]:
        """One COCO annotation per object, its box under "box3d" and its box volume as "area"."""
        return [
            {"id": idx + 1, "image_id": 0, "category_id": int(class_id), "box3d": box, "area": float(volume), "iscrowd": 0}
            for idx, (class_id, box, volume) in enumerate(zip(self.class_ids, self.boxes, _box_volumes(self.boxes)))
        ]


@dataclass(frozen=True)
class GroundTruthObjects(SceneObjects):
    """Annotated objects of the ground-truth scene, with their simulator object ids."""

    object_ids: np.ndarray

    @classmethod
    def from_meshes(cls, meshes: List[Tuple[int, int, trimesh.Trimesh]]) -> "GroundTruthObjects":
        """Objects of (object id, class, world mesh) triples, each boxed by its mesh bounds."""
        return cls(
            class_ids=np.array([class_id for _, class_id, _ in meshes], dtype=int),
            boxes=np.reshape([mesh.bounds for _, _, mesh in meshes], (-1, 2, 3)),
            object_ids=np.array([object_id for object_id, _, _ in meshes], dtype=int),
        )


@dataclass(frozen=True)
class ResolvedDetections(SceneObjects):
    """Resolved objects scored by their highest class probability."""

    scores: np.ndarray
    object_ids: np.ndarray

    @classmethod
    def from_resolved(cls, resolved: ResolvedObjects) -> "ResolvedDetections":
        """Every resolved object, boxed by the extent of its points snapped outward to _PREDICTED_BOX_GRID_M."""
        object_classes = resolved.object_classes()
        points =[resolved.resolved_object_points(object_id) for object_id in object_classes]
        extents = np.reshape([[object_points.min(axis=0), object_points.max(axis=0)] for object_points in points], (-1, 2, 3))
        return cls(
            class_ids=np.array(list(object_classes.values()), dtype=int),
            boxes=_snap_boxes_outward(extents, _PREDICTED_BOX_GRID_M),
            scores=np.array([float(resolved.object_logits(int(object_id)).max()) for object_id in object_classes]),
            object_ids=np.array(list(object_classes), dtype=int),
        )

    def coco_annotations(self) -> List[dict]:
        return [dict(annotation, score=float(score)) for annotation, score in zip(super().coco_annotations(), self.scores)]

    def by_score(self) -> "ResolvedDetections":
        order = np.argsort(-self.scores, kind="stable")
        return ResolvedDetections(self.class_ids[order], self.boxes[order], self.scores[order], self.object_ids[order])


class _SceneCOCOeval(COCOeval):
    """COCOeval matching "box3d" boxes by 3D IoU."""

    def computeIoU(self, imgId, catId):
        gts, dts = self._gts[imgId, catId], self._dts[imgId, catId]
        dts = sorted(dts, key=lambda d: -d["score"])[: self.params.maxDets[-1]]
        if not gts or not dts:
            return []
        return _box_iou_3d(np.array([d["box3d"] for d in dts]), np.array([g["box3d"] for g in gts]))


def _box_wireframe(box: np.ndarray, color: list) -> trimesh.Trimesh:
    """Thin cylinders of color (0-255) along the edges of a (2, 3) (min, max) box."""
    corners = np.unique(list(itertools.product(*box.T)), axis=0)
    edges = [(start, end) for start, end in itertools.combinations(corners, 2) if np.count_nonzero(start != end) == 1]
    wireframe = trimesh.util.concatenate([trimesh.creation.cylinder(_WIREFRAME_RADIUS_M, segment=edge) for edge in edges])
    wireframe.visual.face_colors = [*color, 255]
    return wireframe


def save_resolved_wireframes(resolved: ResolvedObjects, classes: list, colors: list, out_path: Path) -> None:
    """Writes a .glb with one node per resolved object, its point-extent box as a class-colored wireframe."""
    detections = ResolvedDetections.from_resolved(resolved)
    trimesh.Scene({
        f"{idx}_{classes[class_id]}": _box_wireframe(box, colors[class_id])
        for idx, (class_id, box) in enumerate(zip(detections.class_ids, detections.boxes))
    }).export(out_path)


def _scene_coco(objects: SceneObjects, classes: list) -> COCO:
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 0}],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(classes)],
        "annotations": objects.coco_annotations(),
    }
    coco.createIndex()
    return coco


def _scene_coco_eval(detections: ResolvedDetections, ground_truth: GroundTruthObjects, classes: list) -> COCOeval:
    coco_eval = _SceneCOCOeval(_scene_coco(ground_truth, classes), _scene_coco(detections, classes), iouType="bbox")
    coco_eval.params.iouThrs = _IOU_THRESHOLDS
    coco_eval.params.areaRng = _VOLUME_RANGES
    coco_eval.params.maxDets = [max(len(detections.class_ids), 1)]
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval


def _mean_precision(precision: np.ndarray) -> float:
    """Mean of the defined (> -1) entries of a slice of COCOeval's precision, in percent (nan when none are)."""
    defined = precision[precision > -1]
    return float(defined.mean() * 100) if defined.size else float("nan")


def _ap(precision: np.ndarray, iou_mask: np.ndarray, area_index: int = 0, class_index=slice(None)) -> float:
    """_mean_precision of COCOeval's precision at the iou_mask thresholds, one volume range and class_index's classes."""
    return _mean_precision(precision[iou_mask, :, class_index, area_index, -1])


def _summary(precision: np.ndarray) -> Dict[str, float]:
    """AP averaged over IoU 0.5:0.95 (overall and per volume range), and at IoU 0.25 and 0.5."""
    coco_range = _IOU_THRESHOLDS >= 0.5
    return {
        "AP": _ap(precision, coco_range),
        "AP25": _ap(precision, _IOU_THRESHOLDS == 0.25),
        "AP50": _ap(precision, _IOU_THRESHOLDS == 0.5),
        "APs": _ap(precision, coco_range, area_index=1),
        "APm": _ap(precision, coco_range, area_index=2),
        "APl": _ap(precision, coco_range, area_index=3),
    }


def _per_class_table(precision: np.ndarray, classes: list, class_ids: np.ndarray) -> str:
    """AP25, AP50 and AP (IoU 0.5:0.95) of each of class_ids."""
    rows = [
        (classes[class_id], _ap(precision, _IOU_THRESHOLDS == 0.25, class_index=class_id),
         _ap(precision, _IOU_THRESHOLDS == 0.5, class_index=class_id), _ap(precision, _IOU_THRESHOLDS >= 0.5, class_index=class_id))
        for class_id in class_ids
    ]
    return tabulate(rows, tablefmt="pipe", floatfmt=".3f", headers=["category", "AP25", "AP50", "AP"], numalign="left")


def evaluate_scene(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, classes: list) -> Dict[str, float]:
    """LVIS-style 3D box AP of resolved's objects against ground_truth, printed overall and per GT class."""
    precision = _scene_coco_eval(ResolvedDetections.from_resolved(resolved), ground_truth, classes).eval["precision"]
    summary = _summary(precision)
    print(f"3D box AP (small < {_SMALL_MAX_VOLUME_M3} m3 <= medium < {_MEDIUM_MAX_VOLUME_M3} m3 <= large):\n{create_small_table(summary)}")
    print(f"Per-category 3D box AP:\n{_per_class_table(precision, classes, np.unique(ground_truth.class_ids))}")
    return summary


def _scene_matches(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, iou_threshold: float) -> Tuple[ResolvedDetections, np.ndarray, np.ndarray]:
    """resolved's score-sorted detections, their 3D box IoU with ground_truth and their _greedy_matches."""
    detections = ResolvedDetections.from_resolved(resolved).by_score()
    ious = _box_iou_3d(detections.boxes, ground_truth.boxes)
    return detections, ious, _greedy_matches(ious, iou_threshold)


def matched_ground_truth(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, iou_threshold: float = 0.25) -> Dict[int, int]:
    """{object id: index in ground_truth of the object it class-agnostically matches (3D box IoU >= iou_threshold)} of every matching resolved object."""
    detections, _, matches = _scene_matches(resolved, ground_truth, iou_threshold)
    return {int(object_id): int(match) for object_id, match in zip(detections.object_ids, matches) if match >= 0}


def matched_ground_truth_classes(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, iou_threshold: float = 0.25) -> Dict[int, int]:
    """{object id: class of the ground truth it class-agnostically matches (3D box IoU >= iou_threshold)} of every matching resolved object."""
    return {object_id: int(ground_truth.class_ids[match]) for object_id, match in matched_ground_truth(resolved, ground_truth, iou_threshold).items()}


def _matched_percent(matched: np.ndarray) -> float:
    return float(100 * matched.mean()) if len(matched) else float("nan")


def scene_matching(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, classes: list, iou_threshold: float = 0.25) -> Dict[str, float]:
    """Percent of ground truth class-agnostically matched by a resolved object (3D box IoU >= iou_threshold), overall and by
    volume, and percent of resolved objects matching one; printed with the unmatched ground truth."""
    detections, _, matches = _scene_matches(resolved, ground_truth, iou_threshold)
    matched = np.isin(np.arange(len(ground_truth.boxes)), matches)
    volumes = _box_volumes(ground_truth.boxes)
    summary = {
        name: _matched_percent(matched[(volumes >= low) & (volumes < high)])
        for name, (low, high) in zip(("GT matched", "GT matched s", "GT matched m", "GT matched l"), _VOLUME_RANGES)
    }
    summary["resolved matching"] = _matched_percent(matches >= 0)
    print(f"Class-agnostic matching (3D box IoU >= {iou_threshold}): {matched.sum()}/{len(matched)} ground truth matched, "
          f"{(matches >= 0).sum()}/{len(matches)} resolved objects matching\n{create_small_table(summary)}")
    unmatched = [(classes[ground_truth.class_ids[idx]], _box_center_size(ground_truth.boxes[idx])) for idx in np.flatnonzero(~matched)]
    print(f"Unmatched ground truth:\n{tabulate(unmatched, headers=['class', '(xc, yc, zc) (w, h, d)'], tablefmt='pipe')}")
    return summary


def scene_confusion_matrix(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, classes: list, out_path: Path, iou_threshold: float = 0.25) -> np.ndarray:
    """compute_confusion_matrix for resolved's objects against ground_truth by 3D box IoU, saved as a heatmap only showing the
    classes present in either; prints each object's match (_object_matches_table)."""
    detections, ious, matches = _scene_matches(resolved, ground_truth, iou_threshold)
    print(f"Resolved objects matched to ground truth (3D box IoU >= {iou_threshold}):\n{_object_matches_table(detections, ground_truth, classes, matches, ious)}")
    matrix = np.zeros((len(classes) + 1, len(classes) + 1), dtype=np.int64)
    _add_matches(matrix, ground_truth.class_ids, detections.class_ids, matches)
    present = (matrix.sum(axis=0) + matrix.sum(axis=1)) > 0
    present[-1] = True
    present_matrix = matrix[np.ix_(present, present)]
    labels = [label for label, kept in zip(list(classes) + ["background"], present) if kept]
    # print(f"3D box confusion matrix (rows ground truth, columns predicted, IoU >= {iou_threshold}):\n{_confusion_table(present_matrix, labels)}")
    _save_confusion_heatmap(present_matrix, labels, out_path)
    return matrix


def _box_center_size(box: np.ndarray) -> str:
    """(xc, yc, zc) (w, h, d) of a (2, 3) (min, max) box, in meters."""
    center, size = box.mean(axis=0), box[1] - box[0]
    return "({:.2f}, {:.2f}, {:.2f}) ({:.2f}, {:.2f}, {:.2f})".format(*center, *size)


def _matched_label(ground_truth: GroundTruthObjects, classes: list, match: int, object_ious: np.ndarray) -> str:
    return "background" if match < 0 else f"{classes[ground_truth.class_ids[match]]} ({object_ious[match]:.2f})"


def _object_matches_table(detections: ResolvedDetections, ground_truth: GroundTruthObjects, classes: list, matches: np.ndarray, ious: np.ndarray) -> str:
    """One row per resolved object, by id: its predicted class with its score, its box's center and size, and its matched
    ground truth's class with their IoU and its box's center and size, or background."""
    rows = sorted(
        (
            int(object_id), f"{classes[class_id]} ({score:.2f})", _box_center_size(box), _matched_label(ground_truth, classes, match, object_ious),
            "" if match < 0 else _box_center_size(ground_truth.boxes[match]),
        )
        for object_id, class_id, score, box, match, object_ious in zip(
            detections.object_ids, detections.class_ids, detections.scores, detections.boxes, matches, ious,
        )
    )
    headers = ["obj_id", "predicted class", "(xc, yc, zc) (w, h, d)", "matched class (IoU)", "matched (xc, yc, zc) (w, h, d)"]
    return tabulate(rows, headers=headers, tablefmt="pipe")


def _confusion_table(matrix: np.ndarray, labels: list) -> str:
    return tabulate([[label, *row] for label, row in zip(labels, matrix.tolist())], headers=["gt \\ pred", *labels], tablefmt="pipe")


def save_maps(resolved: ResolvedObjects, meshes: List[Tuple[int, int, trimesh.Trimesh]], agent: HybridFrontierAgent, style: SegmentationStyle, out_dir: Path) -> None:
    """semantic_point_cloud.ply, semantic_wireframes.glb and semantic_topdown.png of resolved's objects, and
    ground_truth_topdown.png of the (object id, class, world mesh) meshes."""
    save_resolved_ply(resolved, style.colors, out_dir / "semantic_point_cloud.ply")
    save_resolved_wireframes(resolved, style.classes, style.colors, out_dir / "semantic_wireframes.glb")
    cv2.imwrite(str(out_dir / "semantic_topdown.png"), semantic_topdown(resolved, agent, style.classes, style.colors)[:, :, ::-1])
    cv2.imwrite(str(out_dir / "ground_truth_topdown.png"), ground_truth_topdown(meshes, agent, style.classes, style.colors)[:, :, ::-1])


def evaluate_resolved(resolved: ResolvedObjects, meshes: List[Tuple[int, int, trimesh.Trimesh]], classes: list, out_dir: Path) -> None:
    """3D box AP, confusion_matrix.png and class-agnostic matching of resolved's objects against the (object id, class, world mesh) meshes."""
    ground_truth = GroundTruthObjects.from_meshes(meshes)
    evaluate_scene(resolved, ground_truth, classes)
    scene_confusion_matrix(resolved, ground_truth, classes, out_dir / "confusion_matrix.png")
    scene_matching(resolved, ground_truth, classes)
