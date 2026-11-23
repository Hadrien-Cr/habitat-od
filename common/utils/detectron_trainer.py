"""Training/eval infra shared by pretrain.py (COCO training, e.g. BoxSup-C2_L_CLIP_SwinB_896b32_4x
from scratch) and habitat_embodied_sl/main.py (CustomFT_Detic fine-tuning on ObjectViewsDataset): dataset
registration, the evaluator-wired Trainer, periodic TensorBoard image logging, and
final-checkpoint reporting."""
import contextlib
import io
import json
from collections import defaultdict
from itertools import chain, zip_longest
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data import detection_utils  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore
from detectron2.engine import DefaultTrainer, HookBase  # type: ignore
from detectron2.evaluation import COCOEvaluator  # type: ignore
from detectron2.modeling.roi_heads import CascadeROIHeads  # type: ignore
from detectron2.evaluation.testing import flatten_results_dict  # type: ignore
from detectron2.structures import Boxes, BoxMode, Instances  # type: ignore
from detectron2.utils import comm  # type: ignore
from detectron2.utils.events import EventStorage, get_event_storage  # type: ignore
from PIL import Image
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops import batched_nms, nms

from common.env_utils.vocab_constants import make_colors
from common.utils.eval_utils import compute_confusion_matrix, mean_precision, save_classification_confusion
from common.utils.plot_utils import MOSAIC_JPEG_QUALITY, make_mosaic, plot_boxes_overlay

COCO_ROOT = Path(__file__).resolve().parents[2] / "coco_testbench" / "datasets" / "coco"
IMAGE_LOG_SAMPLES = 4
IMAGE_LOG_HEIGHT = 512
DISPLAY_SCORE_THRESH = 0.3  # predictions kept in logged images and confusion matrices; AP uses SCORE_THRESH_TEST
DISPLAY_NMS_IOU = 0.5
AGNOSTIC_NMS_IOU = 0.5  # class-agnostic scoring keeps one detection, the top-scored, of boxes overlapping more
CLASSIFICATION_TOP_K = 5


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


@torch.no_grad()
def ground_truth_box_probs(roi_heads, features: Dict[str, torch.Tensor], image_size: tuple, boxes: torch.Tensor) -> torch.Tensor:
    """(N, C) class probabilities roi_heads give the (N, 4) xyxy boxes of one image_size image of backbone features,
    averaged over the cascade stages as Detic's _forward_box does."""
    box_features = [features[name] for name in roi_heads.box_in_features]
    proposals = [Instances(image_size, proposal_boxes=Boxes(boxes.to(box_features[0].device)))]
    if not isinstance(roi_heads, CascadeROIHeads):
        pooled = roi_heads.box_pooler(box_features, [proposals[0].proposal_boxes])
        return roi_heads.box_predictor.predict_probs(roi_heads.box_predictor(roi_heads.box_head(pooled)), proposals)[0][:, :-1]
    stage_probs = [
        roi_heads.box_predictor[stage].predict_probs(roi_heads._run_stage(box_features, proposals, stage), proposals)[0]
        for stage in range(roi_heads.num_cascade_stages)
    ]
    return torch.stack(stage_probs).mean(dim=0)[:, :-1]


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


def _event_storage() -> Optional[EventStorage]:
    """The training's event storage, None outside training."""
    try:
        return get_event_storage()
    except AssertionError:
        return None


class ClassAP50Evaluator(COCOEvaluator):
    """COCOEvaluator reporting and logging each class's AP50 (IoU 0.5) as AP50-<class>, instead of its AP averaged over
    IoU 0.5 to 0.95, whose stricter thresholds mostly measure box tightness (frozen when fine-tuning CustomFT_Detic), and
    the class-agnostic "AP50 agnostic" and "recall50 agnostic" (agnostic_ap50_recall50, named as eval_utils' 3D metrics),
    without the NaNs of classes or sizes the dataset has no box of. It also reports classification alone: the share of
    ground-truth boxes, given to model's ROI heads as proposals on the backbone features its inference just computed
    (ground_truth_box_probs, so batch-statistics normalization sees the same batch), whose class is the top-scored
    ("cls_top1acc") or among the CLASSIFICATION_TOP_K top-scored ("cls_top5acc"). Each evaluation also saves, over
    DISPLAY_SCORE_THRESH, its confusion matrix <output_dir>/confusion_matrix_iter<iteration>.png and
    inference_samples_iter<iteration>.jpg, whose tiles also go to TensorBoard during training (_save_inference_samples),
    and each ground-truth box's classification to coco_gt_box_classifications.json, with their confusion matrix, free of
    localization and score thresholds, cls_confusion_matrix_iter<iteration>.png."""

    def __init__(self, dataset_name: str, model, **kwargs):
        self._model = model.module if isinstance(model, DistributedDataParallel) else model
        super().__init__(dataset_name, **kwargs)

    def reset(self):
        super().reset()
        self._classifications, self._features = [], {}
        self._hook = self._model.backbone.register_forward_hook(lambda module, inputs, features: self._features.update(features))

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        for index, sample in enumerate(inputs):
            self._classifications += self._ground_truth_classifications(sample, index)

    def _ground_truth_classifications(self, sample: dict, index: int) -> List[dict]:
        """One record per ground-truth box of sample: its image and annotation ids, its category id, the rank of its class
        among the model's class probabilities of the box (0 for the top-scored) and the top CLASSIFICATION_TOP_K
        (category id, probability)."""
        annotations = [annotation for annotation in self._coco_api.imgToAnns[sample["image_id"]] if not annotation.get("iscrowd", 0)]
        if not annotations:
            return []
        scale = torch.tensor([sample["image"].shape[2] / sample["width"], sample["image"].shape[1] / sample["height"]] * 2)
        boxes = BoxMode.convert(torch.tensor([annotation["bbox"] for annotation in annotations], dtype=torch.float), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) * scale
        to_contiguous = self._metadata.thing_dataset_id_to_contiguous_id
        to_dataset = {contiguous: dataset_id for dataset_id, contiguous in to_contiguous.items()}
        features = {name: feature[index:index + 1] for name, feature in self._features.items()}
        probs = ground_truth_box_probs(self._model.roi_heads, features, tuple(sample["image"].shape[1:]), boxes).cpu()
        ranks = (probs > probs[range(len(annotations)), [to_contiguous[annotation["category_id"]] for annotation in annotations]][:, None]).sum(dim=1)
        top_probs, top_classes = probs.topk(min(CLASSIFICATION_TOP_K, probs.shape[1]), dim=1)
        return [
            {"image_id": sample["image_id"], "annotation_id": annotation["id"], "category_id": annotation["category_id"], "rank": int(rank),
             f"top{CLASSIFICATION_TOP_K}": [[to_dataset[int(class_id)], round(float(prob), 4)] for class_id, prob in zip(classes, class_probs)]}
            for annotation, rank, classes, class_probs in zip(annotations, ranks, top_classes, top_probs)
        ]

    def evaluate(self, img_ids=None):
        self._hook.remove()
        self._features = {}
        classifications = list(chain(*comm.gather(self._classifications)))
        results = super().evaluate(img_ids)
        if not results:
            return results
        ranks = torch.tensor([record["rank"] for record in classifications])
        results["bbox"].update({"cls_top1acc": float((ranks == 0).float().mean() * 100), "cls_top5acc": float((ranks < CLASSIFICATION_TOP_K).float().mean() * 100)})
        self._logger.info(f"Ground-truth box classification: top-1 {results['bbox']['cls_top1acc']:.3f}, top-{CLASSIFICATION_TOP_K} {results['bbox']['cls_top5acc']:.3f}")
        if self._output_dir:
            storage = _event_storage()
            out_dir, iteration = Path(self._output_dir), storage.iter if storage else 0
            predictions_json = out_dir / "coco_instances_results.json"
            compute_confusion_matrix(predictions_json, Path(self._metadata.json_file), out_dir / f"confusion_matrix_iter{iteration:05d}.png", score_threshold=DISPLAY_SCORE_THRESH)
            self._save_inference_samples(predictions_json, out_dir / f"inference_samples_iter{iteration:05d}.jpg", storage)
            self._save_classifications(classifications, out_dir / "coco_gt_box_classifications.json", out_dir / f"cls_confusion_matrix_iter{iteration:05d}.png")
        return results

    def _save_classifications(self, classifications: List[dict], json_path: Path, confusion_path: Path) -> None:
        """classifications to json_path, and the confusion matrix of their top-scored classes against their ground truth's
        (save_classification_confusion) to confusion_path."""
        json_path.write_text(json.dumps(classifications))
        to_contiguous = self._metadata.thing_dataset_id_to_contiguous_id
        true_classes = np.array([to_contiguous[record["category_id"]] for record in classifications], dtype=int)
        predicted_classes = np.array([to_contiguous[record[f"top{CLASSIFICATION_TOP_K}"][0][0]] for record in classifications], dtype=int)
        save_classification_confusion(true_classes, predicted_classes, self._metadata.thing_classes, confusion_path)

    def _save_inference_samples(self, predictions_json: Path, out_path: Path, storage: Optional[EventStorage]) -> None:
        """Mosaic of IMAGE_LOG_SAMPLES of the dataset's frames, evenly spaced, each its sample_overlay of predictions_json's
        detections; each tile also put to storage, when training, as images/<dataset>/<frame>."""
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
        make_mosaic(tiles).save(out_path, quality=MOSAIC_JPEG_QUALITY)
        for frame, tile in tiles if storage else []:
            storage.put_image(f"images/{self._metadata.name}/{frame}", tile.transpose(2, 0, 1))

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


def build_evaluator(cfg, dataset_name, model):
    """output_dir is namespaced per dataset_name so scoring several datasets in one process
    doesn't have every COCOEvaluator overwrite the same coco_instances_results.json. bbox-only; model is the one tested,
    whose ROI heads also classify the ground-truth boxes."""
    return ClassAP50Evaluator(dataset_name, model, tasks=("bbox",), output_dir=str(Path(cfg.OUTPUT_DIR) / "eval" / dataset_name))


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


class Trainer(DefaultTrainer):
    """DefaultTrainer testing with build_evaluator, for pretrain.py/habitat_embodied_sl/main.py's own training-time
    EvalHook, whose evaluator also logs each test set's sample predictions to TensorBoard."""

    @classmethod
    def test(cls, cfg, model):
        """DefaultTrainer.test with each of cfg.DATASETS.TEST's build_evaluator given the tested model."""
        return super().test(cfg, model, [build_evaluator(cfg, dataset_name, model) for dataset_name in cfg.DATASETS.TEST])


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
