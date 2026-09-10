"""Standalone detectron2 evaluation entry point, split out of pretrain.py so a checkpoint can
be scored without also training (register a dataset, build the model, run COCOEvaluator, write
results.json + a confusion matrix). pretrain.py imports Trainer/register_coco/
register_ds_config/report_results from here for its own training-time EvalHook and post-train
reporting, rather than duplicating them (register_ds_config itself is only used by pretrain.py --
this script's own --dataset covers the same datasets/<run_name>/ shape directly).

--dataset points at any single already-built COCO dataset directory -- datasets/<run_name>/
(pooled, from collect_dataset.py) or datasets/<run_name>_<scene>/ (one scene, from
collect_validation_dataset.py -- see dataset.py::build_dataset) -- and scores just that one.
Omit it to fall back to registering coco_testbench's local COCO copy instead, to reproduce
detectron2's own model-zoo baselines. Always single-process (no --num-gpus launch()) -- scoring
a validation set doesn't need it.

Usage (score one already-built dataset directory -- config overrides, e.g. MODEL.WEIGHTS, are
positional (detectron2's own opts REMAINDER arg), not a --opts flag):
  PYTHONPATH=. python eval.py \
      --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml \
      --dataset datasets/procthor_reproduce_valdistinct_ProcTHOR-Val-0 \
      MODEL.WEIGHTS habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/model_final.pth
"""
import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf
from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup  # type: ignore
from detectron2.evaluation import COCOEvaluator  # type: ignore

from common.env_utils.vocab_constants import make_colors
from common.utils.eval_utils import compute_confusion_matrix
from habitat_embodied_al import constants
from habitat_embodied_al.dataset import register_dataset

COCO_ROOT = Path(__file__).resolve().parent / "coco_testbench" / "datasets" / "coco"
DEFAULT_CONFIG = "coco_testbench/config/faster_rcnn_R_50_FPN_1x.yaml"


def build_evaluator(cfg, dataset_name):
    """output_dir is namespaced per dataset_name so scoring several datasets in one process
    (e.g. pretrain.py's training-time EvalHook alongside a later run of this script) doesn't
    have every COCOEvaluator overwrite the same coco_instances_results.json."""
    return COCOEvaluator(dataset_name, output_dir=str(Path(cfg.OUTPUT_DIR) / "eval" / dataset_name))


class Trainer(DefaultTrainer):
    """DefaultTrainer with build_evaluator wired in -- for pretrain.py's own EvalHook during
    training, and reused here (Trainer.build_model/Trainer.test) for standalone eval-only
    runs."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return build_evaluator(cfg, dataset_name)


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


def register_ds_config(ds_config: str) -> Optional[str]:
    """Registers habitat_embodied_al's already-collected HSSD dataset
    under the fixed "train"/"val" names habitat_embodied_al.dataset
    .register_dataset always uses. ds_config's run_name locates
    datasets/<run_name>/{train,val}.json (built beforehand by
    collect_dataset.py against the same run_name/object_params -- the
    json's own "categories"/"vocab" fields already reflect that filtering,
    so registration here doesn't need object_params again).

    Returns None if train.json doesn't exist (eval-only ds_config, empty train_scenes)."""
    ds_cfg = OmegaConf.load(ds_config)
    dataset_dir = constants.DATASET_ROOT / ds_cfg.run_name
    train_dataset = register_dataset(dataset_dir, "train") if (dataset_dir / "train.json").exists() else None
    register_dataset(dataset_dir, "val")
    return train_dataset


def register_single_dataset(dataset_dir: str, split_name: str = "val") -> str:
    """Registers one already-built COCO dataset directory directly -- datasets/<run_name>/
    (pooled, from collect_dataset.py) or datasets/<run_name>_<scene>/ (one scene, from
    collect_validation_dataset.py -- see dataset.py::build_dataset) -- under its own directory
    name."""
    dataset_dir_path = Path(dataset_dir)
    if not (dataset_dir_path / f"{split_name}.json").exists():
        raise RuntimeError(
            f"No {split_name}.json found under {dataset_dir} -- build it first "
            "(collect_dataset.py or collect_validation_dataset.py)"
        )
    return register_dataset(dataset_dir_path, split_name, name=dataset_dir_path.name)


def report_results(cfg, dataset_names: list, results: dict, checkpoint: str) -> None:
    """Writes cfg.OUTPUT_DIR/results.json summarizing every dataset_names' COCOEvaluator
    results (Trainer.test returns a flat dict for one dataset, {dataset_name: metrics} for
    several) plus, for each dataset with dumped predictions, a confusion matrix PNG under
    cfg.OUTPUT_DIR/eval/<dataset_name>/ (see build_evaluator). Shared by pretrain.py's
    post-train report and this script's standalone runs."""
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
            compute_confusion_matrix(predictions_json, gt_json, confusion_matrix_png)
            print(f"{dataset_name}: wrote confusion matrix to {confusion_matrix_png}")


def main(args):
    if args.dataset:
        dataset_names = [register_single_dataset(args.dataset, args.split)]
        num_classes_dataset = dataset_names[0]
    else:
        register_coco("train2017")
        dataset_names = [register_coco("val2017")]
        num_classes_dataset = dataset_names[0]

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = tuple(dataset_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(num_classes_dataset).thing_classes)
    cfg.OUTPUT_DIR = str(Path(args.config_file).resolve().parent.parent / "logs" / Path(args.config_file).stem)
    cfg.freeze()
    default_setup(cfg, args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    results = Trainer.test(cfg, model)

    report_results(cfg, dataset_names, results, cfg.MODEL.WEIGHTS)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--dataset",
        default="",
        help="Path to a single already-built COCO dataset directory -- datasets/<run_name>/ "
        "(collect_dataset.py) or datasets/<run_name>_<scene>/ (collect_validation_dataset.py) -- "
        "to score. Omit to fall back to coco_testbench's local COCO copy instead.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Which split inside --dataset to score (its <split>.json/<split>/ pair). Only used "
        "with --dataset.",
    )
    args = parser.parse_args()
    args.config_file = args.config_file or DEFAULT_CONFIG
    main(args)
