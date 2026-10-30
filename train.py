"""Fine-tuning entry point for habitat_embodied_sl/meta_arch.py's CustomFT_Detic on
ObjectViewsDataset -- one of the collection config's three train variants (pseudo,
pseudo_gt_classes, recovered; habitat_embodied_sl/main.py), scored against its written test set
(datasets/<run_name>/test). For a plain COCO architecture trained from scratch/a COCO checkpoint
instead (e.g. BoxSup-C2_L_CLIP_SwinB_896b32_4x), use pretrain.py.

Key methods:
  ObjectViewsTrainer.build_model        -- CustomFT_Detic loaded from cfg.MODEL.WEIGHTS, its
                                            zero-shot classifier reset to the collection config's
                                            kept classes, its teacher synced to the loaded student.
  ObjectViewsTrainer.build_train_loader -- balanced_loader over the chosen variant's
                                            ObjectViewsDataset, cycled past its one-epoch length
                                            since SimpleTrainer.run_step expects an infinite
                                            iterable.
Dataset registration, the evaluator-wired Trainer base, PeriodicImageWriter, and
final-checkpoint reporting are shared with pretrain.py via common/utils/detectron_trainer.py.

Single-GPU only (no launch()/--num-gpus, unlike pretrain.py).

Usage:
  PYTHONPATH=. python train.py --config smoke_sl.yaml \
      --config-file habitat_embodied_sl/config/finetune_detic_swinb.yaml --variant pseudo
"""
from itertools import cycle
from pathlib import Path

from detectron2.data import DatasetMapper, MetadataCatalog  # type: ignore
from detectron2.engine import default_argument_parser, default_setup  # type: ignore
from omegaconf import OmegaConf

from common import constants
from common.utils.detectron_trainer import Trainer, register_single_dataset, report_results
from common.utils.object_views_dataset import ObjectViewsDataset, balanced_loader
from common.utils.plot_utils import plot_metrics
from common.vision.detic import Vocabulary, build_vocabulary_model, detic_cfg

import habitat_embodied_sl.meta_arch  # noqa: F401  registers CustomFT_Detic/PositiveOnlyDeticCascadeROIHeads

DEFAULT_CONFIG = "habitat_embodied_sl/config/finetune_detic_swinb.yaml"


class ObjectViewsTrainer(Trainer):
    """vocabulary/train_json must be set before __init__ (i.e. super().__init__) runs, since
    DefaultTrainer.__init__ calls build_model/build_train_loader before cfg is stored on self."""

    def __init__(self, cfg, vocabulary, train_json):
        self.vocabulary = vocabulary
        self.train_json = train_json
        super().__init__(cfg)

    def build_model(self, cfg):
        model = build_vocabulary_model(cfg, self.vocabulary)
        model.set_teacher()
        return model

    def build_train_loader(self, cfg):
        dataset = ObjectViewsDataset.from_json(self.train_json)
        mapper = DatasetMapper(is_train=True, augmentations=[], image_format="RGB", use_instance_mask=True, instance_mask_format="bitmask")
        return cycle(balanced_loader(dataset, cfg.SOLVER.IMS_PER_BATCH, mapper))


def setup(args):
    collection_cfg = OmegaConf.load(args.config)
    run_root = constants.DATASET_ROOT / collection_cfg.run_name
    test_dataset = register_single_dataset(str(run_root / "test"), "test")
    vocabulary = Vocabulary(test_dataset, tuple(MetadataCatalog.get(test_dataset).thing_classes))

    cfg = detic_cfg(args.config_file, args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(vocabulary.classes)
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.OUTPUT_DIR = str(Path(args.config_file).resolve().parent.parent / "logs" / Path(args.config_file).stem)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg, vocabulary, run_root / args.variant / "train" / "train.json"


def main(args):
    cfg, vocabulary, train_json = setup(args)

    trainer = ObjectViewsTrainer(cfg, vocabulary, train_json)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    checkpoint = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
    report_results(cfg, list(cfg.DATASETS.TEST), trainer._last_eval_results, checkpoint)

    metrics_json = Path(cfg.OUTPUT_DIR) / "metrics.json"
    if metrics_json.exists():
        metrics_png = plot_metrics(metrics_json, Path(cfg.OUTPUT_DIR) / "metrics.png")
        print(f"Wrote training/eval curves to {metrics_png}")

    print(f"Run '{Path(cfg.OUTPUT_DIR).name}' logged to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--config", default="smoke_sl.yaml", help="Collection config: scenes, budgets, object_params, run_name.")
    parser.add_argument("--variant", default="pseudo", choices=["pseudo", "pseudo_gt_classes", "recovered"], help="Which collection train variant to fine-tune on.")
    args = parser.parse_args()
    args.config_file = args.config_file or DEFAULT_CONFIG
    main(args)
