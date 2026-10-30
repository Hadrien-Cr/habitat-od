"""Training/eval infra shared by pretrain.py (COCO training, e.g. BoxSup-C2_L_CLIP_SwinB_896b32_4x
from scratch) and train.py (CustomFT_Detic fine-tuning on ObjectViewsDataset): dataset
registration, the evaluator-wired Trainer, periodic TensorBoard image logging, and
final-checkpoint reporting."""
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data import detection_utils  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore
from detectron2.engine import DefaultTrainer, HookBase  # type: ignore
from detectron2.evaluation import COCOEvaluator  # type: ignore
from detectron2.utils.events import get_event_storage  # type: ignore

from common.env_utils.vocab_constants import make_colors
from common.env_utils.dataset_registry import register_dataset
from common.utils.eval_utils import compute_confusion_matrix
from common.utils.plot_utils import plot_segmentation_pred

COCO_ROOT = Path(__file__).resolve().parents[2] / "coco_testbench" / "datasets" / "coco"
IMAGE_LOG_SAMPLES = 4


def build_evaluator(cfg, dataset_name):
    """output_dir is namespaced per dataset_name so scoring several datasets in one process
    doesn't have every COCOEvaluator overwrite the same coco_instances_results.json. bbox-only:
    mask AP isn't scored, for simplicity -- PeriodicImageWriter still visualizes predicted masks."""
    return COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=str(Path(cfg.OUTPUT_DIR) / "eval" / dataset_name))


class PeriodicImageWriter(HookBase):
    """Every cfg.TEST.EVAL_PERIOD iterations (same cadence as the EvalHook), logs the model's
    predictions on a fixed sample of cfg.DATASETS.TEST[0] to TensorBoard via
    EventStorage.put_image, so detection quality can be watched visually alongside the scalar
    loss/AP curves."""

    def __init__(self, cfg, model):
        self.model = model
        self.period = cfg.TEST.EVAL_PERIOD
        dataset_name = cfg.DATASETS.TEST[0]
        self.classes = MetadataCatalog.get(dataset_name).thing_classes
        self.colors = MetadataCatalog.get(dataset_name).thing_colors
        self.samples = list(islice(DatasetCatalog.get(dataset_name), IMAGE_LOG_SAMPLES))
        self.input_format = cfg.INPUT.FORMAT

    def after_step(self):
        if self.period == 0 or (self.trainer.iter + 1) % self.period != 0:
            return
        storage = get_event_storage()
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for sample in self.samples:
                rgb = detection_utils.read_image(sample["file_name"], format="RGB")
                image = rgb if self.input_format == "RGB" else rgb[:, :, ::-1]
                inputs = {"image": torch.as_tensor(image.transpose(2, 0, 1).copy()), "height": rgb.shape[0], "width": rgb.shape[1]}
                predictions = self.model([inputs])[0]["instances"].to("cpu")

                confident_preds = predictions[predictions.scores > 0.3]
                vis = plot_segmentation_pred(rgb, confident_preds, self.classes, self.colors)
                storage.put_image(f"images/{Path(sample['file_name']).stem}", np.array(vis).transpose(2, 0, 1))
        self.model.train(was_training)


class Trainer(DefaultTrainer):
    """DefaultTrainer with build_evaluator wired in, for pretrain.py/train.py's own training-time
    EvalHook. build_hooks additionally appends PeriodicImageWriter, to log sample predictions to
    TensorBoard alongside the default scalar writers."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return build_evaluator(cfg, dataset_name)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, PeriodicImageWriter(self.cfg, self.model))
        return hooks


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


def register_single_dataset(dataset_dir: str, split_name: str = "val") -> str:
    """Registers one already-built COCO dataset directory -- dataset_dir/<split_name>.json +
    dataset_dir/<split_name>/ images -- under its own directory name."""
    dataset_dir_path = Path(dataset_dir)
    if not (dataset_dir_path / f"{split_name}.json").exists():
        raise RuntimeError(f"No {split_name}.json found under {dataset_dir}")
    name = register_dataset(dataset_dir_path, split_name, name=dataset_dir_path.name)
    meta = MetadataCatalog.get(name)
    if not meta.get("thing_colors"):
        meta.thing_colors = make_colors(len(meta.thing_classes), seed=0, ctype=0)
    return name


def report_results(cfg, dataset_names: list, results: dict, checkpoint: str) -> None:
    """Writes cfg.OUTPUT_DIR/results.json summarizing every dataset_names' COCOEvaluator
    results (Trainer.test returns a flat dict for one dataset, {dataset_name: metrics} for
    several) plus, for each dataset with dumped predictions, a confusion matrix PNG under
    cfg.OUTPUT_DIR/eval/<dataset_name>/ (see build_evaluator). Shared by pretrain.py/train.py's
    post-train report."""
    per_dataset = results if len(dataset_names) > 1 else {dataset_names[0]: results}
    print(f"Results: {per_dataset}")

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"run_name": out_dir.name, "checkpoint": checkpoint, "results": per_dataset}
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_dir / 'results.json'}")

    for dataset_name in dataset_names:
        predictions_json = out_dir / "eval" / dataset_name / "coco_instances_results.json"
        if predictions_json.exists():
            gt_json = Path(MetadataCatalog.get(dataset_name).json_file)
            confusion_matrix_png = out_dir / "eval" / dataset_name / "confusion_matrix.png"
            compute_confusion_matrix(
                predictions_json, gt_json, confusion_matrix_png, score_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
            )
            print(f"{dataset_name}: wrote confusion matrix to {confusion_matrix_png}")
