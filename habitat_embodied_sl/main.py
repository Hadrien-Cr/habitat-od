"""Fine-tuning entry point for habitat_embodied_sl/meta_arch.py's CustomFT_Detic on
ObjectViewsDataset -- one of the collection config's three train variants (pseudo,
gt, gt++; habitat_embodied_sl/collect.py), scored against its written test set
(datasets/<run_name>/test). For a plain COCO architecture trained from scratch/a COCO checkpoint
instead (e.g. BoxSup-C2_L_CLIP_SwinB_896b32_4x), use pretrain.py.

Key methods:
  ObjectViewsTrainer.build_model        -- CustomFT_Detic loaded from cfg.MODEL.WEIGHTS, its
                                            zero-shot classifier reset to the test set's classes,
                                            its teacher loaded from the checkpoint's
                                            backbone and proposal generator.
  ObjectViewsTrainer.build_train_loader -- balanced_loader over DATASETS.TRAIN's variant's
                                            ObjectViewsDataset, its frames resized and flipped as
                                            cfg.INPUT's MIN_SIZE_TRAIN and RANDOM_FLIP say (test
                                            frames: MIN_SIZE_TEST), cycled past its one-epoch length
                                            since SimpleTrainer.run_step expects an infinite
                                            iterable.
  ObjectViewsTrainer.build_optimizer    -- Detic's build_custom_optimizer, so SOLVER.OPTIMIZER and
                                            SOLVER.BACKBONE_MULTIPLIER apply (detectron2's own is SGD only).
ObjectViewsTrainer.test scores every test set in its own vocabulary: the collection's, then each --static-test-dirs
set (e.g. common/env_utils/nyu_depth_v2.py's), the classifier reset to its classes' CLIP embeddings meanwhile.
Dataset registration, the evaluator-wired Trainer base (each evaluation's sample predictions logged to TensorBoard), and
final-checkpoint reporting are shared with pretrain.py via common/utils/detectron_trainer.py. After training,
write_run_analysis adds per-class and box-size plots and tables to the run's logs.

Single-GPU only (no launch()/--num-gpus, unlike pretrain.py).

Usage (defaults: procthor8, config/r50_procthor8_embodied_sl_gt++.yaml; the config's DATASETS.TRAIN names the variant):
  PYTHONPATH=. python habitat_embodied_sl/main.py --config habitat_embodied_sl/config/procthor8_collection.yaml \
      --config-file habitat_embodied_sl/config/r50_procthor8_embodied_sl_gt++.yaml
"""
import csv
import json
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Callable

import numpy as np
from detectron2.data import DatasetCatalog, DatasetMapper, detection_utils  # type: ignore
from detectron2.engine import default_argument_parser, default_setup  # type: ignore
from detectron2.engine import defaults as detectron2_defaults  # type: ignore
from omegaconf import OmegaConf

from common import constants
from common.utils.detectron_trainer import (
    IMAGE_LOG_SAMPLES, ClassCountsWriter, InitialEvalHook, PeriodicImageWriter, Trainer, register_single_dataset, report_results,
)
from common.utils.dataset_writers import IMAGES_DIR, OBJECT_DS_DIR, VARIANTS, frame_scene, split_json_path
from common.utils.object_views_dataset import ObjectViewsDataset, balanced_loader
from common.utils.plot_utils import (
    class_ap_history, load_metrics, plot_box_areas, plot_class_ap, plot_embedding_tsne, plot_metrics, plot_similarity_triangle,
)
from common.vision.detic import build_vocabulary_model, dataset_vocabulary, detic_cfg, reset_vocabulary
from detic.custom_solver import build_custom_optimizer  # type: ignore

import habitat_embodied_sl.meta_arch  # noqa: F401  registers CustomFT_Detic/PositiveOnlyDeticCascadeROIHeads

DEFAULT_CONFIG = "habitat_embodied_sl/config/r50_procthor8_embodied_sl_gt++.yaml"

# default_setup logs the config through _highlight's terminal colour codes, which land in log.txt too
detectron2_defaults._highlight = lambda code, filename: code


class ObjectViewsTrainer(Trainer):
    """vocabulary/train_dataset must be set before __init__ (i.e. super().__init__) runs, since
    DefaultTrainer.__init__ calls build_model/build_train_loader before cfg is stored on self."""

    def __init__(self, cfg, vocabulary, train_dataset: ObjectViewsDataset):
        self.vocabulary = vocabulary
        self.train_dataset = train_dataset
        super().__init__(cfg)

    def build_model(self, cfg):
        return build_vocabulary_model(cfg, self.vocabulary)

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_custom_optimizer(cfg, model)

    @classmethod
    def test(cls, cfg, model):
        """Trainer.test of each cfg.DATASETS.TEST dataset in its own vocabulary (score_each_vocabulary)."""
        return score_each_vocabulary(cfg, model, lambda dataset_name: Trainer.test(with_test_dataset(cfg, dataset_name), model))

    def build_train_loader(self, cfg):
        mapper = DatasetMapper(is_train=True, augmentations=detection_utils.build_augmentation(cfg, is_train=True), image_format="RGB")
        return cycle(balanced_loader(self.train_dataset, cfg.SOLVER.IMS_PER_BATCH, mapper))

    def build_hooks(self):
        """Trainer's hooks plus, before training, the loaded weights' test scores and each split's class counts, and the
        predictions on IMAGE_LOG_SAMPLES training views (fixed seed) logged to TensorBoard at every evaluation."""
        hooks = super().build_hooks()
        split_annotations = {
            "train": [(record["object_key"].split("/")[0], record["annotations"][0]["category_id"]) for record in self.train_dataset.records],
            "test": [
                (frame_scene(record["file_name"]), annotation["category_id"])
                for record in DatasetCatalog.get(self.cfg.DATASETS.TEST[0]) for annotation in record["annotations"]
            ],
        }
        hooks.insert(-1, ClassCountsWriter(split_annotations, list(self.vocabulary.classes)))
        hooks.insert(-1, InitialEvalHook())
        records = self.train_dataset.records
        views = [records[index] for index in np.random.default_rng(0).choice(len(records), size=IMAGE_LOG_SAMPLES, replace=False)]
        hooks.insert(-1, PeriodicImageWriter(self.cfg, self.model, {"train_views": views}))
        return hooks


def with_test_dataset(cfg, dataset_name: str):
    """A copy of cfg scoring dataset_name only."""
    dataset_cfg = cfg.clone()
    dataset_cfg.defrost()
    dataset_cfg.DATASETS.TEST = (dataset_name,)
    return dataset_cfg


def score_each_vocabulary(cfg, model, score: Callable[[str], dict]) -> dict:
    """score(dataset name) of each cfg.DATASETS.TEST dataset, model's classifier reset to the dataset's classes meanwhile
    and back to the first's (the training classes) after: the first's results, with every other's under its name."""
    results = {}
    for dataset_name in cfg.DATASETS.TEST:
        reset_vocabulary(model, dataset_vocabulary(dataset_name))
        results[dataset_name] = score(dataset_name)
    first, *others = cfg.DATASETS.TEST
    reset_vocabulary(model, dataset_vocabulary(first))
    return {**results[first], **{dataset_name: results[dataset_name] for dataset_name in others}}


def write_class_table(class_aps: dict, classes: list, train_records: list, test_records: list, out_path: Path) -> Path:
    """CSV of each class's training objects and views, test instances, and AP50 at the first, best and last evaluation."""
    train_objects = Counter(record["annotations"][0]["category_id"] for record in {record["object_key"]: record for record in train_records}.values())
    train_views = Counter(record["annotations"][0]["category_id"] for record in train_records)
    test_instances = Counter(annotation["category_id"] for record in test_records for annotation in record["annotations"])
    columns = ["class", "train_objects", "train_views", "test_instances", "ap50_initial", "ap50_best", "ap50_final", "ap50_change"]
    with open(out_path, "w", newline="") as table:
        writer = csv.DictWriter(table, columns)
        writer.writeheader()
        for class_id in sorted(range(len(classes)), key=lambda class_id: -test_instances[class_id]):
            aps = class_aps.get(classes[class_id])
            ap_columns = {} if aps is None else {"ap50_initial": aps[0], "ap50_best": aps.max(), "ap50_final": aps[-1], "ap50_change": aps[-1] - aps[0]}
            writer.writerow({
                "class": classes[class_id], "train_objects": train_objects[class_id], "train_views": train_views[class_id],
                "test_instances": test_instances[class_id], **{column: round(value, 2) for column, value in ap_columns.items()},
            })
    return out_path


def write_run_analysis(cfg, trainer: ObjectViewsTrainer) -> None:
    """class_ap.png, class_table.csv and box_areas.png of a finished run, in cfg.OUTPUT_DIR."""
    out_dir = Path(cfg.OUTPUT_DIR)
    train_records, test_records = trainer.train_dataset.records, DatasetCatalog.get(cfg.DATASETS.TEST[0])
    iterations, class_aps = class_ap_history(out_dir / "metrics.json")
    print(f"Wrote per-class AP to {plot_class_ap(iterations, class_aps, out_dir / 'class_ap.png')}")
    class_table = write_class_table(class_aps, list(trainer.vocabulary.classes), train_records, test_records, out_dir / "class_table.csv")
    print(f"Wrote per-class table to {class_table}")
    split_areas = {
        split: np.array([annotation["bbox"][2] * annotation["bbox"][3] for record in records for annotation in record["annotations"]])
        for split, records in {"train": train_records, "test": test_records}.items()
    }
    print(f"Wrote box sizes to {plot_box_areas(split_areas, out_dir / 'box_areas.png')}")


def collection_root(args) -> Path:
    return constants.DATASET_ROOT / OmegaConf.load(args.config).run_name


def setup(args, run_name: str):
    """(cfg logged to logs/<run_name>, scored on the collection's test set, then its train set (the train frames' full
    ground truth) and the static test sets; the test set's Vocabulary)."""
    run_root = collection_root(args)
    test_dataset = register_single_dataset(run_root / "test")
    vocabulary = dataset_vocabulary(test_dataset)
    cfg = detic_cfg(args.config_file, args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(vocabulary.classes)
    static_datasets = [register_single_dataset(Path(path)) for path in args.static_test_dirs]
    cfg.DATASETS.TEST = (test_dataset, register_single_dataset(run_root / "train"), *static_datasets)
    cfg.OUTPUT_DIR = str(Path(args.config_file).resolve().parent.parent / "logs" / run_name)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg, vocabulary


def object_views(cfg, vocabulary, run_root: Path) -> ObjectViewsDataset:
    """The ObjectViewsDataset of the train variant cfg.DATASETS.TRAIN names, its classes those of vocabulary."""
    if len(cfg.DATASETS.TRAIN) != 1 or cfg.DATASETS.TRAIN[0] not in VARIANTS:
        raise ValueError(f"DATASETS.TRAIN must name one train variant of {VARIANTS}, got {cfg.DATASETS.TRAIN}")
    object_ds_json = split_json_path(run_root / cfg.DATASETS.TRAIN[0] / OBJECT_DS_DIR)
    if tuple(json.loads(object_ds_json.read_text())["info"]["categories"]) != vocabulary.classes:
        raise ValueError(f"{object_ds_json}'s classes differ from the test set's, in content or order: train labels would index the wrong class")
    return ObjectViewsDataset.from_json(object_ds_json, run_root / IMAGES_DIR)


def main(args, trainer_class=ObjectViewsTrainer):
    cfg, vocabulary = setup(args, args.run_name or Path(args.config_file).stem)

    trainer = trainer_class(cfg, vocabulary, object_views(cfg, vocabulary, collection_root(args)))
    embeddings, classes = vocabulary.clip_embeddings().T.numpy(), list(vocabulary.classes)
    print(f"Wrote vocabulary t-SNE to {plot_embedding_tsne(embeddings, classes, Path(cfg.OUTPUT_DIR) / 'vocabulary_tsne.png')}")
    print(f"Wrote vocabulary similarity to {plot_similarity_triangle(embeddings, classes, Path(cfg.OUTPUT_DIR) / 'vocabulary_similarity.png')}")
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    metrics_json = Path(cfg.OUTPUT_DIR) / "metrics.json"
    best = max((record for record in load_metrics(metrics_json) if "bbox/AP" in record), key=lambda record: record["bbox/AP"])
    best_evaluation = {key: best[key] for key in ("iteration", "bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl")}
    checkpoint = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    report_results(cfg, list(cfg.DATASETS.TEST[:1]), trainer._last_eval_results, checkpoint, best_evaluation)
    print(f"Wrote training/eval curves to {plot_metrics(metrics_json, Path(cfg.OUTPUT_DIR) / 'metrics.png')}")
    write_run_analysis(cfg, trainer)

    print(f"Run '{Path(cfg.OUTPUT_DIR).name}' logged to {cfg.OUTPUT_DIR}")


def train_argument_parser():
    """detectron2's default_argument_parser plus the collection config, static test sets and log folder name of a run."""
    parser = default_argument_parser()
    parser.add_argument("--config", default="habitat_embodied_sl/config/procthor8_collection.yaml", help="Collection config: run_name, budgets, object_params.")
    parser.add_argument("--static-test-dirs", nargs="*", default=[], help="Static COCO test sets (<dir>/test.json, images in <dir>/../images/, e.g. common/env_utils/nyu_depth_v2.py's) also scored, each in its own vocabulary.")
    parser.add_argument("--run-name", help="Log folder name under logs/, defaulting to --config-file's stem.")
    return parser


if __name__ == "__main__":
    args = train_argument_parser().parse_args()
    args.config_file = args.config_file or DEFAULT_CONFIG
    main(args)
