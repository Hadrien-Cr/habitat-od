import contextlib
import copy
import io
import itertools
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Set, Tuple

import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.hybrid_exploration.exploration_map import MapGrid
from common.utils.coco_utils import decoded_mask, encoded_masks, json_rle, write_json
from common.utils.plot_utils import PNG_COMPRESS_LEVEL, plot_semantic_2d_map, plot_tide_errors
from common.utils.projection_utils import TOPDOWN_MAP_SCALE, ResolvedObjects, filled_footprint, mesh_footprint_on_grid, save_labeled_ply
from common.utils.retrieval import SegmentationStyle

_IOU_THRESHOLDS = np.round(np.arange(0.25, 0.951, 0.05), 2)
_SMALL_MAX_VOLUME_M3 = 0.01
_MEDIUM_MAX_VOLUME_M3 = 0.25
_WIREFRAME_RADIUS_M = 0.005
_VIEWPOINT_SPACING_M = 0.5
_BOX_SAMPLE_FRACTIONS = np.array(list(itertools.product((0.25, 0.5, 0.75), repeat=3)))
_HIT_TOLERANCE_M = 0.02
_GROUND_TRUTH_POINTS_PER_M2 = 1600  # one point per downsample voxel face (0.025 m), as dense as the fused objects
_PREDICTED_BOX_GRID_M = 0.02
_MIN_MATCHED_EXTENT_M = 0.05
GROUND_TRUTH_JSON = "ground_truth.json"
PREDICTIONS_JSON = "predictions.json"
_VOLUME_RANGES = [[0.0, np.inf], [0.0, _SMALL_MAX_VOLUME_M3], [_SMALL_MAX_VOLUME_M3, _MEDIUM_MAX_VOLUME_M3], [_MEDIUM_MAX_VOLUME_M3, np.inf]]
_VOLUME_RANGE_SIZES = ("", "s", "m", "l")
_TOP_K = 5
_TIDE_BACKGROUND_IOU = 0.1
_TIDE_ORACLE_FIELDS = {"Cls": ("category_id",), "Loc": ("bbox", "box3d", "area")}
_MAX_HEATMAP_CLASSES = 60  # more is unreadable and can exceed matplotlib's 2^16-pixel figure limit (NYU's 890 classes)


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
    """Row-normalized heatmap of matrix, annotated with its raw counts, drawn for its _MAX_HEATMAP_CLASSES rows (and their
    columns) with the most counts."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    kept = np.sort(np.argsort(-row_sums[:, 0], kind="stable")[:_MAX_HEATMAP_CLASSES])
    matrix, normalized, labels = matrix[np.ix_(kept, kept)], normalized[np.ix_(kept, kept)], [labels[index] for index in kept]

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


def save_classification_confusion(true_classes: np.ndarray, predicted_classes: np.ndarray, classes: list, out_path: Path) -> None:
    """Heatmap (_save_confusion_heatmap) of the confusion matrix of predicted_classes against true_classes, indices into classes."""
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (true_classes, predicted_classes), 1)
    _save_confusion_heatmap(matrix, classes, out_path)


def one_class_per_proposal(detections: List[dict]) -> List[dict]:
    """Copies of COCO detections with each image's box kept once, under its top-scored class: per-class NMS emits a box of
    class-agnostic regression once per class over the score threshold."""
    top_scored = {(detection["image_id"], tuple(detection["bbox"])): detection for detection in sorted(detections, key=lambda detection: detection["score"])}
    return [dict(detection) for detection in top_scored.values()]


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
    unrelated false positive. Predictions are kept one_class_per_proposal, so a box's lower-scored classes don't
    fill the background row.

    `skip_unseen_classes` drops categories with zero GT annotations from both axes (e.g. a
    per-scene val set that only covers a handful of the full vocab -- see
    dataset.py::build_dataset's mode="separate") instead of leaving them as empty rows/columns;
    predictions of a dropped class are then out of scope too (there's no row to attribute them
    to) and are excluded rather than folded into "background"."""
    with open(gt_json) as f:
        gt = json.load(f)
    with open(predictions_json) as f:
        predictions = one_class_per_proposal(json.load(f))

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


def _annotation_fields(annotations: List[dict]) -> dict:
    """SceneObjects fields of COCO annotations (SceneObjects.coco_annotations)."""
    return {
        "class_ids": np.array([annotation["category_id"] for annotation in annotations], dtype=int),
        "boxes": np.reshape([annotation["box3d"] for annotation in annotations], (-1, 2, 3)),
        "object_ids": np.array([annotation["object_id"] for annotation in annotations], dtype=int),
    }


@dataclass(frozen=True)
class SceneObjects:
    """Scene objects as class ids, world-axis-aligned (N, 2, 3) (min, max) boxes and object ids."""

    class_ids: np.ndarray
    boxes: np.ndarray
    object_ids: np.ndarray

    @classmethod
    def from_annotations(cls, annotations: List[dict]) -> "SceneObjects":
        return cls(**_annotation_fields(annotations))

    def coco_annotations(self) -> List[dict]:
        """One COCO annotation per object, its box under "box3d", its box volume as "area" and its object id."""
        return [
            {"id": idx + 1, "image_id": 0, "category_id": int(class_id), "box3d": box.tolist(), "area": float(volume), "iscrowd": 0, "object_id": int(object_id)}
            for idx, (class_id, box, volume, object_id) in enumerate(zip(self.class_ids, self.boxes, _box_volumes(self.boxes), self.object_ids))
        ]


@dataclass(frozen=True)
class GroundTruthObjects(SceneObjects):
    """Annotated objects of the ground-truth scene, with their simulator object ids."""

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
    """Resolved objects scored by their highest class probability, with their (N, classes) class_scores."""

    scores: np.ndarray
    class_scores: np.ndarray

    @classmethod
    def from_resolved(cls, resolved: ResolvedObjects) -> "ResolvedDetections":
        """Every resolved object, boxed by the extent of its points snapped outward to _PREDICTED_BOX_GRID_M."""
        object_classes = resolved.object_classes()
        points =[resolved.resolved_object_points(object_id) for object_id in object_classes]
        extents = np.reshape([[object_points.min(axis=0), object_points.max(axis=0)] for object_points in points], (-1, 2, 3))
        class_scores = [resolved.object_logits(int(object_id)).float().cpu().numpy().reshape(-1) for object_id in object_classes]
        return cls(
            class_ids=np.array(list(object_classes.values()), dtype=int),
            boxes=_snap_boxes_outward(extents, _PREDICTED_BOX_GRID_M),
            object_ids=np.array(list(object_classes), dtype=int),
            scores=np.array([float(object_scores.max()) for object_scores in class_scores]),
            class_scores=np.array(class_scores),
        )

    @classmethod
    def from_annotations(cls, annotations: List[dict]) -> "ResolvedDetections":
        return cls(
            **_annotation_fields(annotations),
            scores=np.array([annotation["score"] for annotation in annotations], dtype=float),
            class_scores=np.array([annotation["class_scores"] for annotation in annotations], dtype=float),
        )

    def coco_annotations(self) -> List[dict]:
        return [
            dict(annotation, score=float(score), class_scores=object_scores.tolist())
            for annotation, score, object_scores in zip(super().coco_annotations(), self.scores, self.class_scores)
        ]

    def by_score(self) -> "ResolvedDetections":
        order = np.argsort(-self.scores, kind="stable")
        return ResolvedDetections(**{field.name: getattr(self, field.name)[order] for field in fields(self)})


class _SceneCOCOeval(COCOeval):
    """COCOeval matching "box3d" boxes by 3D IoU, of catId or of every category when they are ignored (params.useCats 0)."""

    def computeIoU(self, imgId, catId):
        cat_ids = [catId] if self.params.useCats else self.params.catIds
        gts = [gt for cat_id in cat_ids for gt in self._gts[imgId, cat_id]]
        dts = [dt for cat_id in cat_ids for dt in self._dts[imgId, cat_id]]
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


def save_box_wireframes(objects: SceneObjects, classes: list, colors: list, out_path: Path) -> None:
    """Writes a .glb with one node per object, its box as a class-colored wireframe."""
    trimesh.Scene({
        f"{idx}_{classes[class_id]}": _box_wireframe(box, colors[class_id])
        for idx, (class_id, box) in enumerate(zip(objects.class_ids, objects.boxes))
    }).export(out_path)


def save_ground_truth_ply(meshes: List[Tuple[int, int, trimesh.Trimesh]], colors: list, out_path: Path) -> None:
    """Writes the (object id, class, world mesh) meshes' surfaces as a .ply, _GROUND_TRUTH_POINTS_PER_M2 points per square
    meter each colored by its class, comparable with save_resolved_ply's."""
    samples = [(trimesh.sample.sample_surface(mesh, max(1, round(mesh.area * _GROUND_TRUTH_POINTS_PER_M2)), seed=0)[0], class_id) for _, class_id, mesh in meshes]
    labels = np.concatenate([np.full(len(points), class_id) for points, class_id in samples])
    save_labeled_ply(np.concatenate([points for points, _ in samples]), labels, colors, out_path)


def _resolved_footprints(resolved: ResolvedObjects, grid: MapGrid, object_ids: np.ndarray) -> np.ndarray:
    """(N, H, W) filled_footprint of each of object_ids' resolved points on grid, empty for an object off it."""
    shape = grid.explored.shape
    pixels = grid.pixels(resolved.resolved_points)
    inside = ((pixels >= 0) & (pixels < np.array(shape))).all(axis=1)
    point_object_ids = resolved.resolved_infos[:, 1]
    footprints = [filled_footprint(pixels[inside & (point_object_ids == object_id)], shape) for object_id in object_ids]
    return np.array(footprints, dtype=bool).reshape(-1, *shape)


def _ground_truth_navigable(grid: MapGrid, pathfinder: habitat_sim.PathFinder) -> np.ndarray:
    """The ground-truth navmesh's navigable cells (GroundTruthMapTracker's, on the world axes) resampled onto grid."""
    ground_truth = GroundTruthMapTracker(pathfinder, grid.ref_position, fov_deg=0.0, visibility_dist_m=0.0, resolution=grid.explored.shape[0])
    cells = np.indices(grid.explored.shape).reshape(2, -1).T
    pixels = ground_truth.to_pixels(grid.world_points(cells))
    inside = ((pixels >= 0) & (pixels < np.array(ground_truth.navigable.shape))).all(axis=1)
    navigable = np.zeros(len(cells), dtype=bool)
    navigable[inside] = ground_truth.navigable[tuple(pixels[inside].T)]
    return navigable.reshape(grid.explored.shape)


def _grid_image(scene: str, meters_per_pixel: float, masks: Dict[str, np.ndarray]) -> dict:
    """COCO image of a scene's top-down grid: its size, cell size and its named (H, W) masks, RLE-encoded."""
    height, width = next(iter(masks.values())).shape
    rles = encoded_masks(np.stack(list(masks.values())))
    return {"id": 0, "file_name": scene, "height": height, "width": width, "meters_per_pixel": meters_per_pixel, **{name: json_rle(rle) for name, rle in zip(masks, rles)}}


def _scene_dataset(image: dict, objects: SceneObjects, footprints: np.ndarray, classes: list) -> dict:
    """Single-scene COCO dataset of objects on image's grid, each with its (H, W) top-down footprint as its RLE "segmentation"."""
    return {
        "images": [image],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(classes)],
        "annotations": [dict(annotation, segmentation=json_rle(rle)) for annotation, rle in zip(objects.coco_annotations(), encoded_masks(footprints))],
    }


def restricted_scene_dataset(dataset: dict, class_ids: Dict[int, int], classes: list) -> dict:
    """A single-scene dataset (_scene_dataset) with only its objects of the classes class_ids maps, their category ids
    mapped, its categories classes."""
    return {
        **dataset, "categories": [{"id": idx, "name": name} for idx, name in enumerate(classes)],
        "annotations": [dict(annotation, category_id=class_ids[annotation["category_id"]]) for annotation in dataset["annotations"] if annotation["category_id"] in class_ids],
    }


def navigable_viewpoints(dataset: dict, grid: MapGrid, height: float) -> np.ndarray:
    """(N, 3) points at height above the cells of dataset's "navigable" mask (a ground_truth_dataset's, the navmesh's) on
    grid, one every _VIEWPOINT_SPACING_M along each axis."""
    stride = max(1, round(_VIEWPOINT_SPACING_M / grid.meters_per_pixel))
    cells = np.argwhere(decoded_mask(dataset["images"][0]["navigable"]))
    points = grid.world_points(cells[np.all(cells % stride == 0, axis=1)])
    points[:, 1] = height
    return points


def observable_ground_truth(dataset: dict, scene_mesh: trimesh.Trimesh, viewpoints: np.ndarray, max_depth: float) -> dict:
    """dataset without the objects none of viewpoints has in line of sight within max_depth (_in_line_of_sight), like a
    closed fridge's contents: what no camera on the navmesh could see is not counted as missed."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(scene_mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(scene_mesh.faces, np.uint32)))
    boxes = GroundTruthObjects.from_annotations(dataset["annotations"]).boxes
    observable = [_in_line_of_sight(scene, box, viewpoints, max_depth) for box in boxes]
    return {**dataset, "annotations": [annotation for annotation, seen in zip(dataset["annotations"], observable) if seen]}


def _in_line_of_sight(scene: o3d.t.geometry.RaycastingScene, box: np.ndarray, viewpoints: np.ndarray, max_depth: float) -> bool:
    """Whether a ray from a viewpoint within max_depth of the (2, 3) box to one of its _BOX_SAMPLE_FRACTIONS points first
    hits scene inside the box, or nothing."""
    near = viewpoints[np.linalg.norm(viewpoints - box.mean(axis=0), axis=1) < max_depth]
    targets = box[0] + _BOX_SAMPLE_FRACTIONS * (box[1] - box[0])
    origins, ends = np.repeat(near, len(targets), axis=0), np.tile(targets, (len(near), 1))
    directions = (ends - origins) / np.linalg.norm(ends - origins, axis=1, keepdims=True)
    t_hit = scene.cast_rays(o3d.core.Tensor(np.hstack([origins, directions]).astype(np.float32)))["t_hit"].numpy()
    missed = ~np.isfinite(t_hit)
    hits = origins + directions * np.where(missed, 0, t_hit)[:, None]
    on_box = np.all((hits >= box[0] - _HIT_TOLERANCE_M) & (hits <= box[1] + _HIT_TOLERANCE_M), axis=1)
    return bool((missed | on_box).any())


def ground_truth_dataset(
    scene: str, meshes: List[Tuple[int, int, trimesh.Trimesh]], grid: MapGrid, pathfinder: habitat_sim.PathFinder, classes: list,
) -> dict:
    """_scene_dataset of the (object id, class, world mesh) meshes, boxed by their mesh bounds, their footprints their mesh
    silhouettes on grid, whose "navigable" mask is the ground-truth navmesh's (_ground_truth_navigable)."""
    shape = grid.explored.shape
    footprints = np.array([mesh_footprint_on_grid(mesh, shape, grid.pixels) for _, _, mesh in meshes], dtype=bool).reshape(-1, *shape)
    image = _grid_image(scene, grid.meters_per_pixel, {"navigable": _ground_truth_navigable(grid, pathfinder)})
    return _scene_dataset(image, GroundTruthObjects.from_meshes(meshes), footprints, classes)


def predictions_dataset(scene: str, resolved: ResolvedObjects, grid: MapGrid, classes: list) -> dict:
    """_scene_dataset of resolved's objects (ResolvedDetections), their footprints their points' (_resolved_footprints), on
    grid with its "navigable" (explored and free) and "explored" masks."""
    detections = ResolvedDetections.from_resolved(resolved)
    masks = {"navigable": grid.explored & ~grid.obstacles.astype(bool), "explored": grid.explored}
    image = _grid_image(scene, grid.meters_per_pixel, masks)
    return _scene_dataset(image, detections, _resolved_footprints(resolved, grid, detections.object_ids), classes)


def semantic_topdown(dataset: dict, style: SegmentationStyle) -> Image.Image:
    """A single-scene dataset's object footprints by class (plot_semantic_2d_map) over its grid's navigable cells; with an
    "explored" mask (an agent's belief map), its unexplored cells white and its labels black."""
    image = dataset["images"][0]
    explored = decoded_mask(image["explored"]) if "explored" in image else None
    footprints = [(annotation["category_id"], decoded_mask(annotation["segmentation"])) for annotation in dataset["annotations"]]
    return plot_semantic_2d_map(
        decoded_mask(image["navigable"]), [(class_id, mask) for class_id, mask in footprints if mask.any()],
        dict(enumerate(style.colors)), dict(enumerate(style.classes)), meters_per_grid_pixel=image["meters_per_pixel"],
        scale=TOPDOWN_MAP_SCALE, explored=explored, text_color=(255, 255, 255) if explored is None else (0, 0, 0),
    )


def save_topdowns(scene_dir: Path, style: SegmentationStyle) -> None:
    """estimated_topdown.png and ground_truth_topdown.png in scene_dir, the semantic_topdown of its predictions.json and ground_truth.json."""
    for json_name, png_name in ((PREDICTIONS_JSON, "estimated_topdown.png"), (GROUND_TRUTH_JSON, "ground_truth_topdown.png")):
        semantic_topdown(json.loads((scene_dir / json_name).read_text()), style).save(scene_dir / png_name, compress_level=PNG_COMPRESS_LEVEL)


def _merged_dataset(paths: List[Path]) -> dict:
    """The single-scene datasets at paths pooled, each scene its own image; annotation ids restart at 1, as COCOeval treats id 0 as unmatched."""
    datasets = [json.loads(path.read_text()) for path in paths]
    annotations = [dict(annotation, image_id=image_id) for image_id, dataset in enumerate(datasets) for annotation in dataset["annotations"]]
    return {
        "images": [dict(dataset["images"][0], id=image_id) for image_id, dataset in enumerate(datasets)],
        "categories": datasets[0]["categories"],
        "annotations": [dict(annotation, id=idx + 1) for idx, annotation in enumerate(annotations)],
    }


def _coco(dataset: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def _precision(coco_eval: COCOeval) -> np.ndarray:
    """coco_eval's precision at _IOU_THRESHOLDS and by _VOLUME_RANGES, every detection of a scene kept."""
    coco_eval.params.iouThrs = _IOU_THRESHOLDS
    coco_eval.params.areaRng = _VOLUME_RANGES
    coco_eval.params.maxDets = _every_detection(coco_eval.cocoDt)
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval.eval["precision"]


def _every_detection(predictions: COCO) -> List[int]:
    """COCOeval's maxDets keeping every detection of predictions' scenes."""
    return [max([1, *map(len, predictions.imgToAnns.values())])]


def _scene_tide_daps(ground_truth: COCO, predictions: COCO, iou_threshold: float) -> Dict[str, float]:
    """tide_daps of predictions' 3D boxes (each object's top-1 class), TIDE's foreground iou_threshold, every volume and detection."""
    tide = _SceneCOCOeval(ground_truth, predictions, iouType="bbox")
    tide.params.iouThrs, tide.params.areaRng, tide.params.maxDets = np.array([iou_threshold]), [[0.0, np.inf]], _every_detection(predictions)
    return tide_daps(tide)


def mean_precision(precision: np.ndarray) -> float:
    """Mean of the defined (> -1) entries of a slice of COCOeval's precision, in percent (nan when none are)."""
    defined = precision[precision > -1]
    return float(defined.mean() * 100) if defined.size else float("nan")


def _ap(precision: np.ndarray, iou_mask: np.ndarray, area_index: int = 0, class_index=slice(None)) -> float:
    """mean_precision of COCOeval's precision at the iou_mask thresholds, one volume range and class_index's classes."""
    return mean_precision(precision[iou_mask, :, class_index, area_index, -1])


def _summary(precision: np.ndarray) -> Dict[str, float]:
    """AP averaged over IoU 0.5:0.95, and at IoU 0.25 (overall and per volume range) and 0.5."""
    iou_25 = _IOU_THRESHOLDS == 0.25
    return {
        "AP": _ap(precision, _IOU_THRESHOLDS >= 0.5),
        "AP25": _ap(precision, iou_25),
        "AP50": _ap(precision, _IOU_THRESHOLDS == 0.5),
        "AP25s": _ap(precision, iou_25, area_index=1),
        "AP25m": _ap(precision, iou_25, area_index=2),
        "AP25l": _ap(precision, iou_25, area_index=3),
    }


def _per_class_table(precision: np.ndarray, classes: list, class_ids: np.ndarray) -> str:
    """AP25, AP50 and AP (IoU 0.5:0.95) of each of class_ids."""
    rows = [
        (classes[class_id], _ap(precision, _IOU_THRESHOLDS == 0.25, class_index=class_id),
         _ap(precision, _IOU_THRESHOLDS == 0.5, class_index=class_id), _ap(precision, _IOU_THRESHOLDS >= 0.5, class_index=class_id))
        for class_id in class_ids
    ]
    return tabulate(rows, tablefmt="pipe", floatfmt=".3f", headers=["category", "AP25", "AP50", "AP"], numalign="left")


def _ap25_by_volume(precision: np.ndarray, suffix: str) -> Dict[str, float]:
    """"AP25[s|m|l]<suffix>": AP25 of COCOeval's precision, overall and by volume."""
    return {f"AP25{size}{suffix}": _ap(precision, _IOU_THRESHOLDS == 0.25, area_index=idx) for idx, size in enumerate(_VOLUME_RANGE_SIZES)}


def _top_k_dataset(predictions: dict, k: int) -> dict:
    """predictions with each object submitted once per class among its k highest class_scores, scored by that class's score."""
    annotations = [
        dict(annotation, category_id=int(class_id), score=annotation["class_scores"][class_id])
        for annotation in predictions["annotations"]
        for class_id in np.argsort(annotation["class_scores"])[::-1][:k]
    ]
    return dict(predictions, annotations=[dict(annotation, id=idx + 1) for idx, annotation in enumerate(annotations)])


def _ap_metrics(ground_truth: COCO, predictions: COCO, classes: list) -> Dict[str, float]:
    """LVIS-style _summary of the 3D box AP of predictions (each object's top-1 class), and AP25, overall and by volume, of
    each object's _TOP_K classes; prints the per-class AP of ground_truth's classes."""
    precision = _precision(_SceneCOCOeval(ground_truth, predictions, iouType="bbox"))
    class_ids = np.unique([annotation["category_id"] for annotation in ground_truth.dataset["annotations"]])
    print(f"Per-category 3D box AP:\n{_per_class_table(precision, classes, class_ids)}")
    top_k = _SceneCOCOeval(ground_truth, _coco(_top_k_dataset(predictions.dataset, _TOP_K)), iouType="bbox")
    return {**_summary(precision), **_ap25_by_volume(_precision(top_k), f" top{_TOP_K}")}


@dataclass(frozen=True)
class TideErrors:
    """TIDE's error of each false-positive detection id, with the ground truth id a Cls or Loc error targets (-1 for none),
    and the ground truth ids true positives match and no detection matches."""

    detections: Dict[int, Tuple[str, int]]
    matched: Set[int]
    unmatched: Set[int]

    def missed(self) -> Set[int]:
        """Unmatched ground truth no Cls or Loc error targets."""
        return self.unmatched - {target for _, target in self.detections.values()}


def single_threshold_ap(coco_eval: COCOeval) -> float:
    """mean_precision of coco_eval, set to one IoU threshold, area range and detection cap, once evaluated."""
    coco_eval.evaluate()
    coco_eval.accumulate()
    return mean_precision(coco_eval.eval["precision"][..., 0, -1])


def _detection_matches(coco_eval: COCOeval) -> Dict[int, int]:
    """{detection id: id of the ground truth it matches, 0 for none} of every unignored detection coco_eval evaluated."""
    return {
        int(detection_id): int(match)
        for image in filter(None, coco_eval.evalImgs)
        for detection_id, match, ignored in zip(image["dtIds"], image["dtMatches"][0], image["dtIgnore"][0]) if not ignored
    }


def _image_objects(coco_eval: COCOeval, image_id: int) -> Tuple[List[dict], List[dict]]:
    """(ground truth, detections) of image_id, every category's, in the order computeIoU, classes ignored and detections uncapped, pairs them."""
    ground_truth = [annotation for cat_id in coco_eval.params.catIds for annotation in coco_eval._gts[image_id, cat_id]]
    detections = [annotation for cat_id in coco_eval.params.catIds for annotation in coco_eval._dts[image_id, cat_id]]
    return ground_truth, sorted(detections, key=lambda detection: -detection["score"])


def _detection_error(ious: np.ndarray, same_class: np.ndarray, iou_threshold: float) -> Tuple[str, int]:
    """TIDE's error of a false-positive detection of ious with each ground truth (same_class as it or not), checked in
    TIDE's order, and the index of the ground truth a Cls or Loc error targets (-1 for none)."""
    same_class_ious, other_class_ious = np.where(same_class, ious, 0.0), np.where(same_class, 0.0, ious)
    if _TIDE_BACKGROUND_IOU <= same_class_ious.max(initial=0.0) < iou_threshold:
        return "Loc", int(same_class_ious.argmax())
    if other_class_ious.max(initial=0.0) >= iou_threshold:
        return "Cls", int(other_class_ious.argmax())
    if same_class_ious.max(initial=0.0) >= iou_threshold:
        return "Dupe", -1
    if ious.max(initial=0.0) < _TIDE_BACKGROUND_IOU:
        return "Bkg", -1
    return "Both", -1


def _image_errors(agnostic: COCOeval, image_id: int, matches: Dict[int, int]) -> Dict[int, Tuple[str, int]]:
    """_detection_error of each of image_id's detections matches maps to no ground truth, as {id: (error, target id)},
    by agnostic's IoU of every category's detections with every category's unignored ground truth."""
    ground_truth, detections = _image_objects(agnostic, image_id)
    ious = np.reshape(agnostic.computeIoU(image_id, None), (len(detections), len(ground_truth)))
    ious = np.where([annotation["ignore"] for annotation in ground_truth], 0.0, ious)
    ground_truth_ids = [annotation["id"] for annotation in ground_truth] + [-1]
    ground_truth_classes = np.array([annotation["category_id"] for annotation in ground_truth])
    errors = {
        detection["id"]: _detection_error(detection_ious, ground_truth_classes == detection["category_id"], agnostic.params.iouThrs[0])
        for detection, detection_ious in zip(detections, ious) if matches.get(detection["id"]) == 0
    }
    return {detection_id: (error, ground_truth_ids[target]) for detection_id, (error, target) in errors.items()}


def tide_errors(coco_eval: COCOeval) -> TideErrors:
    """TideErrors of coco_eval's evaluated class-aware matching (TIDE's foreground threshold its single IoU threshold)."""
    matches = _detection_matches(coco_eval)
    agnostic = copy.copy(coco_eval)
    agnostic.params = copy.copy(coco_eval.params)
    agnostic.params.useCats, agnostic.params.maxDets = 0, [sys.maxsize]  # COCOeval caps detections per class, not per image
    detections = {}
    for image_id in coco_eval.params.imgIds:
        detections.update(_image_errors(agnostic, image_id, matches))
    matched = set(matches.values()) - {0}
    ground_truth = coco_eval.cocoGt.loadAnns(coco_eval.cocoGt.getAnnIds(imgIds=coco_eval.params.imgIds, catIds=coco_eval.params.catIds))
    return TideErrors(detections, matched, {annotation["id"] for annotation in ground_truth if not annotation["ignore"]} - matched)


def _retargeted(detections: List[dict], ground_truth: COCO, errors: TideErrors, kind: str) -> List[dict]:
    """detections with each `kind` (Cls or Loc) error given its target's _TIDE_ORACLE_FIELDS, by descending score, or
    dropped when its target is already matched or taken by a higher-scored one (TIDE's duplicate suppression)."""
    taken, fixed = set(errors.matched), []
    for detection in sorted(detections, key=lambda detection: -detection["score"]):
        error, target = errors.detections.get(detection["id"], ("", -1))
        if error != kind:
            fixed.append(detection)
        elif target not in taken:
            taken.add(target)
            oracle = ground_truth.anns[target]
            fixed.append(dict(detection, **{key: oracle[key] for key in _TIDE_ORACLE_FIELDS[kind] if key in oracle}))
    return fixed


def _without(annotations: List[dict], ids: Set[int]) -> List[dict]:
    return [annotation for annotation in annotations if annotation["id"] not in ids]


def _oracle_datasets(coco_eval: COCOeval, errors: TideErrors) -> Dict[str, Tuple[List[dict], List[dict]]]:
    """{error type, FP or FN: (ground truth, detection) annotations with every error of it fixed}."""
    ground_truth, detections = coco_eval.cocoGt.dataset["annotations"], coco_eval.cocoDt.dataset["annotations"]
    of_kind = {kind: {id_ for id_, (error, _) in errors.detections.items() if error == kind} for kind in ("Both", "Dupe", "Bkg")}
    return {
        **{kind: (ground_truth, _retargeted(detections, coco_eval.cocoGt, errors, kind)) for kind in ("Cls", "Loc")},
        **{kind: (ground_truth, _without(detections, ids)) for kind, ids in of_kind.items()},
        "Miss": (_without(ground_truth, errors.missed()), detections),
        "FP": (ground_truth, _without(detections, set(errors.detections))),
        "FN": (_without(ground_truth, errors.unmatched), detections),
    }


def _oracle_ap(coco_eval: COCOeval, ground_truth: List[dict], detections: List[dict]) -> float:
    """single_threshold_ap, under coco_eval's class and params, of ground_truth's and detections' annotations."""
    oracle = type(coco_eval)(
        _coco(dict(coco_eval.cocoGt.dataset, annotations=ground_truth)), _coco(dict(coco_eval.cocoDt.dataset, annotations=detections)), coco_eval.params.iouType,
    )
    oracle.params = copy.deepcopy(coco_eval.params)
    return single_threshold_ap(oracle)


def tide_daps(coco_eval: COCOeval) -> Dict[str, float]:
    """TIDE's dAP (arXiv 2008.08115) of each error type (Cls, Loc, Both, Dupe, Bkg, Miss) and of every false positive (FP)
    and negative (FN): the AP gained once an oracle fixes them, coco_eval set to one IoU threshold (TIDE's foreground),
    area range and detection cap. Negative dAPs, interpolation noise, are clamped to 0 as TIDE does."""
    with contextlib.redirect_stdout(io.StringIO()):
        base_ap = single_threshold_ap(coco_eval)
        oracles = _oracle_datasets(coco_eval, tide_errors(coco_eval))
        return {name: max(_oracle_ap(coco_eval, *oracle) - base_ap, 0.0) for name, oracle in oracles.items()}


@dataclass(frozen=True)
class SceneMatches:
    """One scene's score-sorted detections, their 3D box IoU with its ground truth and the ground truth each greedily,
    class-agnostically matches (-1 for none)."""

    ground_truth: SceneObjects
    detections: ResolvedDetections
    ious: np.ndarray
    matches: np.ndarray

    @classmethod
    def of(cls, ground_truth: SceneObjects, detections: ResolvedDetections, iou_threshold: float) -> "SceneMatches":
        detections = detections.by_score()
        ious = _box_iou_3d(detections.boxes, ground_truth.boxes)
        return cls(ground_truth, detections, ious, _greedy_matches(ious, iou_threshold))

    def ground_truth_matched(self) -> np.ndarray:
        return np.isin(np.arange(len(self.ground_truth.boxes)), self.matches)


def matched_ground_truth(resolved: ResolvedObjects, ground_truth: GroundTruthObjects, iou_threshold: float = 0.25) -> Dict[int, int]:
    """{object id: index in ground_truth of the object it class-agnostically matches (3D box IoU >= iou_threshold)} of every matching resolved object."""
    scene = SceneMatches.of(ground_truth, ResolvedDetections.from_resolved(resolved), iou_threshold)
    return {int(object_id): int(match) for object_id, match in zip(scene.detections.object_ids, scene.matches) if match >= 0}


def _scenes_matches(ground_truth: COCO, predictions: COCO, iou_threshold: float) -> Dict[str, SceneMatches]:
    """{scene: SceneMatches} of every scene (image) of the pooled ground truth and predictions."""
    return {
        image["file_name"]: SceneMatches.of(
            GroundTruthObjects.from_annotations(ground_truth.imgToAnns[image["id"]]),
            ResolvedDetections.from_annotations(predictions.imgToAnns[image["id"]]), iou_threshold,
        )
        for image in ground_truth.dataset["images"]
    }


def _save_confusion_matrix(scenes: Dict[str, SceneMatches], classes: list, out_path: Path) -> None:
    """The (classes + background) confusion matrix of every scene's matches pooled, saved as a heatmap only showing the
    classes present in either."""
    matrix = np.zeros((len(classes) + 1, len(classes) + 1), dtype=np.int64)
    for scene in scenes.values():
        _add_matches(matrix, scene.ground_truth.class_ids, scene.detections.class_ids, scene.matches)
    present = (matrix.sum(axis=0) + matrix.sum(axis=1)) > 0
    present[-1] = True
    labels = [label for label, kept in zip(list(classes) + ["background"], present) if kept]
    _save_confusion_heatmap(matrix[np.ix_(present, present)], labels, out_path)


def _box_center_size(box: np.ndarray) -> str:
    """(xc, yc, zc) (w, h, d) of a (2, 3) (min, max) box, in meters."""
    center, size = box.mean(axis=0), box[1] - box[0]
    return "({:.2f}, {:.2f}, {:.2f}) ({:.2f}, {:.2f}, {:.2f})".format(*center, *size)


def _matched_label(ground_truth: SceneObjects, classes: list, match: int, object_ious: np.ndarray) -> str:
    return "background" if match < 0 else f"{classes[ground_truth.class_ids[match]]} ({object_ious[match]:.2f})"


def _object_matches_table(scene: SceneMatches, classes: list) -> str:
    """One row per resolved object, by id: its predicted class with its score, its box's center and size, and its matched
    ground truth's class with their IoU and its box's center and size, or background."""
    detections, ground_truth = scene.detections, scene.ground_truth
    rows = sorted(
        (
            int(object_id), f"{classes[class_id]} ({score:.2f})", _box_center_size(box), _matched_label(ground_truth, classes, match, object_ious),
            "" if match < 0 else _box_center_size(ground_truth.boxes[match]),
        )
        for object_id, class_id, score, box, match, object_ious in zip(
            detections.object_ids, detections.class_ids, detections.scores, detections.boxes, scene.matches, scene.ious,
        )
    )
    headers = ["obj_id", "predicted class", "(xc, yc, zc) (w, h, d)", "matched class (IoU)", "matched (xc, yc, zc) (w, h, d)"]
    return tabulate(rows, headers=headers, tablefmt="pipe")


def _print_scene_matches(scene_name: str, scene: SceneMatches, classes: list, iou_threshold: float) -> None:
    """scene's resolved objects with their match (_object_matches_table), then its unmatched ground truth."""
    print(f"{scene_name}: resolved objects matched to ground truth (3D box IoU >= {iou_threshold}):\n{_object_matches_table(scene, classes)}")
    ground_truth = scene.ground_truth
    unmatched = [(classes[ground_truth.class_ids[idx]], _box_center_size(ground_truth.boxes[idx])) for idx in np.flatnonzero(~scene.ground_truth_matched())]
    print(f"{scene_name}: unmatched ground truth:\n{tabulate(unmatched, headers=['class', '(xc, yc, zc) (w, h, d)'], tablefmt='pipe')}")


def evaluate_scene_jsons(scene_dirs: List[Path], out_dir: Path, iou_threshold: float = 0.25) -> None:
    """3D box AP, TIDE's dAP at 3D box IoU >= iou_threshold (also drawn to out_dir/tide.png), and confusion_matrix.png of
    the class-agnostic matching at it, of every scene_dirs' predictions.json against its ground_truth.json, pooled; printed,
    and written as one dict to out_dir/metrics.json."""
    ground_truth = _coco(_merged_dataset([scene_dir / GROUND_TRUTH_JSON for scene_dir in scene_dirs]))
    predictions = _coco(_merged_dataset([scene_dir / PREDICTIONS_JSON for scene_dir in scene_dirs]))
    classes = [category["name"] for category in ground_truth.dataset["categories"]]
    scenes = _scenes_matches(ground_truth, predictions, iou_threshold)
    for scene_name, scene in scenes.items():
        _print_scene_matches(scene_name, scene, classes, iou_threshold)
    daps = _scene_tide_daps(ground_truth, predictions, iou_threshold)
    metrics = {**_ap_metrics(ground_truth, predictions, classes), **{f"dAP{round(iou_threshold * 100)}/{name}": dap for name, dap in daps.items()}}
    print(f"3D box metrics (small < {_SMALL_MAX_VOLUME_M3} m3 <= medium < {_MEDIUM_MAX_VOLUME_M3} m3 <= large; TIDE dAP "
          f"at IoU >= {iou_threshold}):\n{tabulate(list(metrics.items()), headers=['metric', 'value'], tablefmt='pipe', floatfmt='.3f')}")
    _save_confusion_matrix(scenes, classes, out_dir / "confusion_matrix.png")
    plot_tide_errors(daps, f"TIDE, 3D box IoU >= {iou_threshold}", out_dir / "tide.png")
    write_json(out_dir / "metrics.json", metrics)
