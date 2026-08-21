import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _xywh_to_xyxy(box: list) -> list:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _iou_xyxy(box1: list, box2: list) -> float:
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def compute_confusion_matrix(
    predictions_json: Path,
    gt_json: Path,
    out_path: Path,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
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
    unrelated false positive."""
    with open(gt_json) as f:
        gt = json.load(f)
    with open(predictions_json) as f:
        predictions = json.load(f)

    categories = sorted(gt["categories"], key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    id_to_index = {c["id"]: i for i, c in enumerate(categories)}
    n = len(class_names)
    background = n

    gt_by_image: dict[int, list[tuple[int, list]]] = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append((id_to_index[ann["category_id"]], _xywh_to_xyxy(ann["bbox"])))

    preds_by_image: dict[int, list[tuple[int, list, float]]] = {}
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        preds_by_image.setdefault(pred["image_id"], []).append(
            (id_to_index[pred["category_id"]], _xywh_to_xyxy(pred["bbox"]), pred["score"])
        )

    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)

    for image_id in set(gt_by_image) | set(preds_by_image):
        gt_boxes = list(gt_by_image.get(image_id, []))
        matched = [False] * len(gt_boxes)

        for pred_class, pred_box, _score in sorted(preds_by_image.get(image_id, []), key=lambda p: -p[2]):
            best_iou, best_idx = 0.0, -1
            for i, (_gt_class, gt_box) in enumerate(gt_boxes):
                if matched[i]:
                    continue
                iou = _iou_xyxy(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= iou_threshold:
                matched[best_idx] = True
                matrix[gt_boxes[best_idx][0], pred_class] += 1
            else:
                matrix[background, pred_class] += 1

        for i, (gt_class, _gt_box) in enumerate(gt_boxes):
            if not matched[i]:
                matrix[gt_class, background] += 1

    labels = class_names + ["background"]
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
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="fraction of GT class's instances (row-normalized)")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return matrix
