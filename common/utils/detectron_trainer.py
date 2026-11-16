"""Training/eval infra shared by pretrain.py (COCO training, e.g. BoxSup-C2_L_CLIP_SwinB_896b32_4x
from scratch) and habitat_embodied_sl/main.py (CustomFT_Detic fine-tuning on ObjectViewsDataset): dataset
registration, the evaluator-wired Trainer, periodic TensorBoard image logging, and
final-checkpoint reporting."""
import contextlib
import io
import json
from collections import defaultdict
from itertools import chain, islice, zip_longest
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data import detection_utils  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore
from detectron2.engine import DefaultTrainer, HookBase  # type: ignore
from detectron2.evaluation import COCOEvaluator  # type: ignore
from detectron2.evaluation.testing import flatten_results_dict  # type: ignore
from detectron2.structures import Boxes, BoxMode, Instances  # type: ignore
from detectron2.utils.events import get_event_storage  # type: ignore
from PIL import Image
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from torchvision.ops import batched_nms, nms

from common.env_utils.vocab_constants import make_colors
from common.env_utils.dataset_registry import register_dataset
from common.utils.eval_utils import compute_confusion_matrix, mean_precision
from common.utils.plot_utils import make_mosaic, plot_boxes_overlay, plot_class_counts

COCO_ROOT = Path(__file__).resolve().parents[2] / "coco_testbench" / "datasets" / "coco"
IMAGE_LOG_SAMPLES = 16
IMAGE_LOG_HEIGHT = 512
DISPLAY_SCORE_THRESH = 0.3  # predictions kept in logged images and confusion matrices; AP uses SCORE_THRESH_TEST
DISPLAY_NMS_IOU = 0.5
AGNOSTIC_NMS_IOU = 0.5  # class-agnostic scoring keeps one detection, the top-scored, of boxes overlapping more


def agnostic_ap50_recall50(coco_eval: COCOeval) -> dict:
    """Class-agnostic (pycocotools' useCats=0) AP50 and recall at IoU 0.5, up to 100 detections an image, of coco_eval's
    detections, of each image's boxes overlapping more than AGNOSTIC_NMS_IOU only the top-scored."""
    detections = coco_eval.cocoDt.dataset["annotations"]
    boxes = BoxMode.convert(torch.tensor([detection["bbox"] for detection in detections], dtype=torch.float), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    scores = torch.tensor([detection["score"] for detection in detections], dtype=torch.float)
    kept = batched_nms(boxes, scores, torch.tensor([detection["image_id"] for detection in detections]), AGNOSTIC_NMS_IOU).tolist()
    kept_detections = [{key: detections[index][key] for key in ("image_id", "category_id", "bbox", "score")} for index in kept]
    with contextlib.redirect_stdout(io.StringIO()):
        agnostic = COCOeval(coco_eval.cocoGt, coco_eval.cocoGt.loadRes(kept_detections), "bbox")
        agnostic.params.useCats, agnostic.params.imgIds, agnostic.params.iouThrs = 0, coco_eval.params.imgIds, np.array([0.5])
        agnostic.params.areaRng, agnostic.params.areaRngLbl, agnostic.params.maxDets = [[0, 1e10]], ["all"], [100]
        agnostic.evaluate()
        agnostic.accumulate()
    return {"AP50 agnostic": mean_precision(agnostic.eval["precision"][0, :, 0, 0, 0]), "recall50 agnostic": float(agnostic.eval["recall"][0, 0, 0, 0] * 100)}


def displayed_predictions(predictions: Instances) -> Instances:
    """predictions scored over DISPLAY_SCORE_THRESH, one class per box: class-agnostic NMS at DISPLAY_NMS_IOU keeps the top-scored class."""
    predictions = predictions[predictions.scores > DISPLAY_SCORE_THRESH]
    return predictions[nms(predictions.pred_boxes.tensor, predictions.scores, DISPLAY_NMS_IOU)]


def sample_overlay(sample: dict, rgb: np.ndarray, predictions: Instances, metadata) -> Image.Image:
    """sample's rgb IMAGE_LOG_HEIGHT pixels high, its annotations' boxes and its displayed_predictions drawn on it (plot_boxes_overlay)."""
    ground_truth = detection_utils.annotations_to_instances(sample["annotations"], rgb.shape[:2], mask_format="bitmask")
    return plot_boxes_overlay(rgb, ground_truth, displayed_predictions(predictions), metadata, IMAGE_LOG_HEIGHT / rgb.shape[0])


def _predicted_instances(results: list, image_size: tuple, dataset_to_contiguous_id: dict) -> Instances:
    """One image's COCO json detections (dataset category ids) as detectron2 predictions."""
    predictions = Instances(image_size)
    boxes = torch.tensor([result["bbox"] for result in results], dtype=torch.float).reshape(-1, 4)
    predictions.pred_boxes = Boxes(BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
    predictions.scores = torch.tensor([result["score"] for result in results], dtype=torch.float)
    predictions.pred_classes = torch.tensor([dataset_to_contiguous_id[result["category_id"]] for result in results], dtype=torch.long)
    return predictions


def _evaluation_iteration() -> int:
    """The training iteration an evaluation runs at, 0 outside training (no event storage)."""
    try:
        return get_event_storage().iter
    except AssertionError:
        return 0


class ClassAP50Evaluator(COCOEvaluator):
    """COCOEvaluator reporting and logging each class's AP50 (IoU 0.5) as AP50-<class>, instead of its AP averaged over
    IoU 0.5 to 0.95, whose stricter thresholds mostly measure box tightness (frozen when fine-tuning CustomFT_Detic), and
    the class-agnostic "AP50 agnostic" and "recall50 agnostic" (agnostic_ap50_recall50, named as eval_utils' 3D metrics),
    without the NaNs of classes or sizes the dataset has no box of. Each evaluation also saves, over DISPLAY_SCORE_THRESH,
    its confusion matrix <output_dir>/confusion_matrix_iter<iteration>.png and inference_samples_iter<iteration>.jpg
    (_save_inference_samples)."""

    def evaluate(self, img_ids=None):
        results = super().evaluate(img_ids)
        if results and self._output_dir:
            out_dir, iteration = Path(self._output_dir), _evaluation_iteration()
            predictions_json = out_dir / "coco_instances_results.json"
            compute_confusion_matrix(predictions_json, Path(self._metadata.json_file), out_dir / f"confusion_matrix_iter{iteration:05d}.png", score_threshold=DISPLAY_SCORE_THRESH)
            self._save_inference_samples(predictions_json, out_dir / f"inference_samples_iter{iteration:05d}.jpg")
        return results

    def _save_inference_samples(self, predictions_json: Path, out_path: Path) -> None:
        """Mosaic of IMAGE_LOG_SAMPLES of the dataset's frames, evenly spaced, each its sample_overlay of predictions_json's detections."""
        records = DatasetCatalog.get(self._metadata.name)
        samples = [records[index] for index in np.linspace(0, len(records) - 1, min(IMAGE_LOG_SAMPLES, len(records))).astype(int)]
        image_results = defaultdict(list)
        for result in json.loads(predictions_json.read_text()):
            image_results[result["image_id"]].append(result)
        tiles = []
        for sample in samples:
            rgb = detection_utils.read_image(sample["file_name"], format="RGB")
            predictions = _predicted_instances(image_results[sample["image_id"]], rgb.shape[:2], self._metadata.thing_dataset_id_to_contiguous_id)
            tiles.append((Path(sample["file_name"]).stem, np.array(sample_overlay(sample, rgb, predictions, self._metadata))))
        make_mosaic(tiles).save(out_path, quality=90)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type)
        if coco_eval is not None and class_names:
            results = {**results, **agnostic_ap50_recall50(coco_eval), **self._class_ap50s(coco_eval, iou_type, class_names)}
            self._logger.info(f"Class-agnostic {iou_type}: AP50 {results['AP50 agnostic']:.3f}, recall50 {results['recall50 agnostic']:.3f}")
        return {name: value for name, value in results.items() if not np.isnan(value)}

    def _class_ap50s(self, coco_eval, iou_type: str, class_names: list) -> dict:
        """{AP50-<class>: its AP50} of every class with some box, logged as a table."""
        precisions = coco_eval.eval["precision"][0, :, :, 0, -1]
        class_ap50s = {name: mean_precision(precision) for name, precision in zip(class_names, precisions.T)}
        rows = zip_longest(*[iter(chain(*class_ap50s.items()))] * 6)
        table = tabulate(rows, headers=["category", "AP50"] * 3, tablefmt="pipe", floatfmt=".3f", numalign="left")
        self._logger.info(f"Per-category {iou_type} AP50: \n{table}")
        return {f"AP50-{name}": ap50 for name, ap50 in class_ap50s.items()}


def build_evaluator(cfg, dataset_name):
    """output_dir is namespaced per dataset_name so scoring several datasets in one process
    doesn't have every COCOEvaluator overwrite the same coco_instances_results.json. bbox-only."""
    return ClassAP50Evaluator(dataset_name, tasks=("bbox",), output_dir=str(Path(cfg.OUTPUT_DIR) / "eval" / dataset_name))


class PeriodicImageWriter(HookBase):
    """Before training and every cfg.TEST.EVAL_PERIOD iterations (same cadence as the EvalHook), logs each split's
    fixed samples' ground truth and the model's predictions on one frame to TensorBoard as images/<split>/<frame> via
    EventStorage.put_image, so detection quality can be watched visually alongside the scalar
    loss/AP curves."""

    def __init__(self, cfg, model, split_samples: dict):
        self.model = model
        self.period = cfg.TEST.EVAL_PERIOD
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.split_samples = split_samples
        self.input_format = cfg.INPUT.FORMAT

    def before_train(self):
        self._log_images()

    def after_step(self):
        if self.period == 0 or (self.trainer.iter + 1) % self.period != 0:
            return
        self._log_images()

    def _log_images(self):
        """Puts each split's samples' sample_overlay to the event storage as images/<split>/<frame>, the model in eval mode
        meanwhile."""
        storage = get_event_storage()
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for split, sample in [(split, sample) for split, samples in self.split_samples.items() for sample in samples]:
                rgb, predictions = self._predict(sample)
                overlay = sample_overlay(sample, rgb, predictions, self.metadata)
                storage.put_image(f"images/{split}/{Path(sample['file_name']).stem}", np.array(overlay).transpose(2, 0, 1))
        self.model.train(was_training)

    def _predict(self, sample: dict) -> tuple:
        """(sample's rgb, the model's predictions on it)."""
        rgb = detection_utils.read_image(sample["file_name"], format="RGB")
        image = rgb if self.input_format == "RGB" else rgb[:, :, ::-1]
        inputs = {"image": torch.as_tensor(image.transpose(2, 0, 1).copy()), "height": rgb.shape[0], "width": rgb.shape[1]}
        return rgb, self.model([inputs])[0]["instances"].to("cpu")


class InitialEvalHook(HookBase):
    """Scores the loaded weights on cfg.DATASETS.TEST before the first step, logged at iteration 0 under the same
    keys as the EvalHook's later scores."""

    def before_train(self):
        results = self.trainer.test(self.trainer.cfg, self.trainer.model)
        self.trainer.storage.put_scalars(**flatten_results_dict(results), smoothing_hint=False)


class ClassCountsWriter(HookBase):
    """Logs to TensorBoard and saves as cfg.OUTPUT_DIR/class_counts.png, once before training, a bar chart of each
    split's annotation count per class and scene, from its annotations' (scene, class id) pairs."""

    def __init__(self, split_annotations: dict, classes: list):
        self.split_annotations = split_annotations
        self.classes = classes

    def before_train(self):
        counts = {split: self._scene_counts(annotations) for split, annotations in self.split_annotations.items()}
        chart = plot_class_counts(counts, self.classes)
        Image.fromarray(chart).save(Path(self.trainer.cfg.OUTPUT_DIR) / "class_counts.png")
        get_event_storage().put_image("class_counts", chart.transpose(2, 0, 1))

    def _scene_counts(self, annotations: list) -> dict:
        """{scene: count per class} of (scene, class id) pairs, scenes in name order."""
        return {
            scene: np.bincount([class_id for annotation_scene, class_id in annotations if annotation_scene == scene], minlength=len(self.classes))
            for scene in sorted({scene for scene, _ in annotations})
        }


class Trainer(DefaultTrainer):
    """DefaultTrainer with build_evaluator wired in, for pretrain.py/habitat_embodied_sl/main.py's own training-time
    EvalHook. build_hooks additionally appends PeriodicImageWriter, to log predictions on image_log_samples to
    TensorBoard alongside the default scalar writers."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return build_evaluator(cfg, dataset_name)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, PeriodicImageWriter(self.cfg, self.model, self.image_log_samples()))
        return hooks

    def image_log_samples(self) -> dict:
        """{split: samples PeriodicImageWriter logs predictions on}: the first IMAGE_LOG_SAMPLES of cfg.DATASETS.TEST[0]."""
        return {"test": list(islice(DatasetCatalog.get(self.cfg.DATASETS.TEST[0]), IMAGE_LOG_SAMPLES))}


def register_coco(split: str) -> str:
    """Registers coco_testbench's local COCO copy under a name distinct
    from any real coco_2017_* names detectron2's own builtins might
    separately register. register_coco_instances only wires up a lazy
    DatasetCatalog loader -- thing_classes isn't populated onto
    MetadataCatalog until that loader actually runs once, so we force it
    here to then fill in thing_colors (never set by register_coco_instances,
    only needed if something visualizes predictions against this name)."""
    name = f"coco_testbench_{split}"
    register_coco_instances(
        name,
        {},
        str(COCO_ROOT / "annotations" / f"instances_{split}.json"),
        str(COCO_ROOT / split),
    )
    DatasetCatalog.get(name)
    meta = MetadataCatalog.get(name)
    meta.thing_colors = make_colors(len(meta.thing_classes), seed=0, ctype=0)
    return name


def register_single_dataset(split_dir: Path) -> str:
    """register_dataset of split_dir, with a color per class for drawing its predictions."""
    name = register_dataset(split_dir)
    meta = MetadataCatalog.get(name)
    if not meta.get("thing_colors"):
        meta.thing_colors = make_colors(len(meta.thing_classes), seed=0, ctype=0)
    return name


def report_results(cfg, dataset_names: list, results: dict, checkpoint: str, best_evaluation: Optional[dict] = None) -> None:
    """Writes cfg.OUTPUT_DIR/results.json summarizing every dataset_names' COCOEvaluator
    results (Trainer.test returns a flat dict for one dataset, {dataset_name: metrics} for
    several), and best_evaluation when given. Shared by pretrain.py/habitat_embodied_sl/main.py's post-train report."""
    per_dataset = results if len(dataset_names) > 1 else {dataset_names[0]: results}
    print(f"Results: {per_dataset}")

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"run_name": out_dir.name, "checkpoint": checkpoint, "results": per_dataset}
    if best_evaluation is not None:
        summary["best_evaluation"] = best_evaluation
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_dir / 'results.json'}")
