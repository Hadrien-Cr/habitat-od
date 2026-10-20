import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from detectron2.utils.logger import create_small_table
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

from common.utils.projection_utils import ResolvedObjects

_IOU_THRESHOLDS = np.round(np.arange(0.25, 0.951, 0.05), 2)
_SMALL_MAX_VOLUME_M3 = 0.01
_MEDIUM_MAX_VOLUME_M3 = 0.25
_WIREFRAME_RADIUS_M = 0.005
_VOLUME_RANGES = [[0.0, np.inf], [0.0, _SMALL_MAX_VOLUME_M3], [_SMALL_MAX_VOLUME_M3, _MEDIUM_MAX_VOLUME_M3], [_MEDIUM_MAX_VOLUME_M3, np.inf]]


def _xywh_to_xyxy(box: list) -> list:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _box_volumes(boxes: np.ndarray) -> np.ndarray:
    """Area (2D) or volume (3D) of (N, 2, D) axis-aligned (min, max) boxes."""
    return np.prod(boxes[:, 1] - boxes[:, 0], axis=-1)


def _box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(len(boxes_a), len(boxes_b)) IoU of (N, 2, D) axis-aligned (min, max) boxes."""
    lower = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    upper = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter = np.clip(upper - lower, 0.0, None).prod(axis=-1)
    union = _box_volumes(boxes_a)[:, None] + _box_volumes(boxes_b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _add_matches(matrix: np.ndarray, gt_classes: np.ndarray, pred_classes: np.ndarray, ious: np.ndarray, iou_threshold: float) -> None:
    """Adds to matrix a greedy class-agnostic matching of score-sorted predictions (rows of ious) to ground truths (columns)."""
    background = len(matrix) - 1
    matched = np.zeros(len(gt_classes), dtype=bool)
    for pred_class, pred_ious in zip(pred_classes, ious):
        unmatched_ious = np.where(matched, 0.0, pred_ious)
        best = int(unmatched_ious.argmax()) if len(unmatched_ious) else -1
        if best >= 0 and unmatched_ious[best] >= iou_threshold:
            matched[best] = True
            matrix[gt_classes[best], pred_class] += 1
        else:
            matrix[background, pred_class] += 1
    np.add.at(matrix[:, background], gt_classes[~matched], 1)


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
        _add_matches(matrix, np.array([c for c, _ in gts], dtype=int), [c for c, _, _ in preds], ious, iou_threshold)

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
    """Annotated objects of the ground-truth scene."""

    @classmethod
    def from_meshes(cls, meshes: List[Tuple[int, trimesh.Trimesh]]) -> "GroundTruthObjects":
        """Objects of (class, world mesh) pairs, each boxed by its mesh bounds."""
        return cls(np.array([class_id for class_id, _ in meshes], dtype=int), np.reshape([mesh.bounds for _, mesh in meshes], (-1, 2, 3)))


@dataclass(frozen=True)
class ResolvedDetections(SceneObjects):
    """Resolved objects scored by their highest class probability."""

    scores: np.ndarray

    @classmethod
    def from_resolved(cls, resolved: ResolvedObjects) -> "ResolvedDetections":
        """Every resolved object, boxed by the extent of its points."""
        object_classes = dict(zip(resolved.resolved_infos[:, 1], resolved.resolved_infos[:, 0]))
        points = [resolved.resolved_object_points(object_id) for object_id in object_classes]
        return cls(
            class_ids=np.array(list(object_classes.values()), dtype=int),
            boxes=np.reshape([[object_points.min(axis=0), object_points.max(axis=0)] for object_points in points], (-1, 2, 3)),
            scores=np.array([float(resolved.object_logits(int(object_id)).max()) for object_id in object_classes]),
        )

    def coco_annotations(self) -> List[dict]:
        return [dict(annotation, score=float(score)) for annotation, score in zip(super().coco_annotations(), self.scores)]

    def by_score(self) -> "ResolvedDetections":
        order = np.argsort(-self.scores, kind="stable")
        return ResolvedDetections(self.class_ids[order], self.boxes[order], self.scores[order])


class _SceneCOCOeval(COCOeval):
    """COCOeval matching "box3d" boxes by 3D IoU."""

    def computeIoU(self, imgId, catId):
        gts, dts = self._gts[imgId, catId], self._dts[imgId, catId]
        dts = sorted(dts, key=lambda d: -d["score"])[: self.params.maxDets[-1]]
        if not gts or not dts:
            return []
        return _box_iou(np.array([d["box3d"] for d in dts]), np.array([g["box3d"] for g in gts]))


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


def scene_confusion_matrix(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, classes: list, out_path: Path, iou_threshold: float = 0.25) -> np.ndarray:
    """compute_confusion_matrix for resolved's objects against ground_truth by 3D box IoU, printed and saved as a heatmap,
    both only showing the classes present in either."""
    detections = ResolvedDetections.from_resolved(resolved).by_score()
    matrix = np.zeros((len(classes) + 1, len(classes) + 1), dtype=np.int64)
    _add_matches(matrix, ground_truth.class_ids, detections.class_ids, _box_iou(detections.boxes, ground_truth.boxes), iou_threshold)
    present = (matrix.sum(axis=0) + matrix.sum(axis=1)) > 0
    present[-1] = True
    present_matrix = matrix[np.ix_(present, present)]
    labels = [label for label, kept in zip(list(classes) + ["background"], present) if kept]
    # print(f"3D box confusion matrix (rows ground truth, columns predicted, IoU >= {iou_threshold}):\n{_confusion_table(present_matrix, labels)}")
    _save_confusion_heatmap(present_matrix, labels, out_path)
    return matrix


def _confusion_table(matrix: np.ndarray, labels: list) -> str:
    return tabulate([[label, *row] for label, row in zip(labels, matrix.tolist())], headers=["gt \\ pred", *labels], tablefmt="pipe")
