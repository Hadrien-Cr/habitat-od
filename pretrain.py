"""Generic detectron2 training entry point, structured to mirror
third_party/detectron2/tools/train_net.py directly (setup/Trainer/main/
invoke_main, default_argument_parser, default_setup, launch) rather than
habitat_embodied_al.maskrcnn_detector.MaskRCNNDetector: that class is
shaped around fine-tuning one fixed Mask R-CNN FPN architecture from a
COCO-pretrained checkpoint onto our own HSSD vocab (and isn't multi-GPU
aware), whereas this script needs to run whichever architecture a given
--config-file calls for, completely unmodified -- train_net.py already is
that generic runner.

The only additions over train_net.py: dataset registration (see below) and
OUTPUT_DIR routing into <config-file's-parent-dir>/logs/<run_name>/
(train_net.py leaves OUTPUT_DIR to the config/opts) -- e.g.
coco_testbench/config/foo.yaml logs to coco_testbench/logs/foo/,
habitat_embodied_al/pretrain/config/foo.yaml logs to
habitat_embodied_al/pretrain/logs/foo/.

Two dataset registration modes, dispatched on whether --ds-config is
given:

- (default) coco_testbench mode: registers coco_testbench's local COCO
  copy (train2017/val2017, hardlink-copied from an existing ~/Detic
  checkout) under fixed coco_testbench_{train2017,val2017} names, which
  coco_testbench/config/Base-RCNN-FPN.yaml's DATASETS.TRAIN/TEST already
  point at. Used to reproduce detectron2's own COCO-Detection model-zoo
  baselines (https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
  verbatim -- a way to confirm this repo's environment/detectron2 checkout
  reproduces published numbers, before trusting it to assess
  habitat_embodied_al's own (HSSD-tuned) Mask R-CNN training configs.

- --ds-config habitat_embodied_al/pretrain/config/ds_hssd.yaml: registers
  habitat_embodied_al's already-collected HSSD dataset (see
  collect_dataset.py, run separately beforehand) via
  habitat_embodied_al.dataset.register_dataset, under the fixed "train"/
  "val" names that function always uses -- matched by
  habitat_embodied_al/pretrain/config/Base-RCNN-FPN.yaml's own
  DATASETS.TRAIN/TEST. MODEL.ROI_HEADS.NUM_CLASSES is set at runtime from
  whichever vocab ds_hssd.yaml's object_params resolves to (currently 17
  classes -- see ds_hssd.yaml's header), since it can't be known until that
  dataset is registered; the config's own COCO-pretrained checkpoint has a
  mismatched 80-class box_predictor, which DetectionCheckpointer skips
  automatically rather than erroring.

Either way, --config-file is a small overlay (_BASE_: "Base-RCNN-FPN.yaml")
mirroring one of detectron2's own COCO-Detection/*.yaml model-zoo configs,
pointed at a local Base-RCNN-FPN.yaml copy instead of
third_party/detectron2/configs/Base-RCNN-FPN.yaml (see that file's header
for why). Both registration modes register their dataset with
evaluator_type "coco" (register_coco_instances/register_dataset
respectively), so build_evaluator always scores with COCOEvaluator.

No manual post-training evaluation call: DefaultTrainer.build_hooks()
already registers an EvalHook(cfg.TEST.EVAL_PERIOD, ...) with
eval_after_train=True, so a final evaluator pass already runs
automatically after the last iteration -- re-running it here would just
duplicate a full pass over the val set. trainer._last_eval_results
(populated by that hook) is used directly for results.json instead.

--num-gpus > 1 runs data-parallel via launch() (one process per GPU,
torch.distributed under the hood): cfg.SOLVER.IMS_PER_BATCH stays the
*global* batch size (detectron2 divides it across processes
automatically), so more GPUs only buys wall-clock speed, not a different
training recipe.

This host's 4 GPUs each sit on their own separate PCIe root complex (see
`lspci`: 8 root-complex segments total, one GPU per complex, none shared --
an AMD EPYC "Genoa" NPS4 layout, plausibly deliberate for per-tenant GPU
isolation on this shared machine), so no GPU pair here has a working direct
P2P path -- confirmed against an unmodified copy of train_net.py itself
(this is a host/NCCL issue, not something introduced by this script) and
narrowed down to exactly one necessary variable: NCCL_P2P_DISABLE=1 (route
GPU-GPU transfers through host memory instead) fixes every pair tried,
alone; NCCL_SOCKET_IFNAME=lo (forcing the bootstrap handshake onto
loopback) makes no measurable difference either way once P2P is disabled.
Like train_net.py, this script does not set any NCCL_* environment
variables itself -- on this host, --num-gpus > 1 needs
`NCCL_P2P_DISABLE=1` set in the environment before invoking it.

Usage (COCO testbench, reproducing a detectron2 model-zoo baseline):
  PYTHONPATH=. python pretrain.py \
      --config-file coco_testbench/config/faster_rcnn_R_50_FPN_1x.yaml \
      --num-gpus 4

Usage (HSSD/embodied-AL fine-tuning on already-collected data):
  PYTHONPATH=. python pretrain.py \
      --config-file habitat_embodied_al/pretrain/config/mask_rcnn_R_50_FPN.yaml \
      --ds-config habitat_embodied_al/pretrain/config/ds_hssd.yaml \
      --num-gpus 2
"""
import json
from pathlib import Path

import detectron2.utils.comm as comm  # type: ignore
from omegaconf import OmegaConf
from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch  # type: ignore
from detectron2.evaluation import COCOEvaluator  # type: ignore

from common.env_utils.vocab_constants import make_colors
from common.utils.eval_utils import compute_confusion_matrix
from common.utils.plot_utils import plot_metrics

COCO_ROOT = Path(__file__).resolve().parent / "coco_testbench" / "datasets" / "coco"
DEFAULT_CONFIG = "coco_testbench/config/faster_rcnn_R_50_FPN_1x.yaml"


def build_evaluator(cfg, dataset_name):
    return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


class Trainer(DefaultTrainer):
    """DefaultTrainer with build_evaluator wired in for
    cfg.TEST.EVAL_PERIOD's EvalHook."""

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


def register_hssd(ds_config: str) -> str:
    """Registers habitat_embodied_al's already-collected HSSD dataset
    under the fixed "train"/"val" names habitat_embodied_al.dataset
    .register_dataset always uses. ds_config's run_name locates
    datasets/<run_name>/{train,val}.json (built beforehand by
    collect_dataset.py against the same run_name/object_params -- the
    json's own "categories"/"vocab" fields already reflect that filtering,
    so registration here doesn't need object_params again)."""
    from habitat_embodied_al import constants
    from habitat_embodied_al.dataset import register_dataset

    ds_cfg = OmegaConf.load(ds_config)
    dataset_dir = constants.DATASET_ROOT / ds_cfg.run_name
    train_dataset = register_dataset(dataset_dir, "train")
    register_dataset(dataset_dir, "val")
    return train_dataset


def setup(args):
    if args.ds_config:
        train_dataset = register_hssd(args.ds_config)
    else:
        register_coco("train2017")
        register_coco("val2017")
        train_dataset = None

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if train_dataset is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(train_dataset).thing_classes)
    cfg.OUTPUT_DIR = str(Path(args.config_file).resolve().parent.parent / "logs" / Path(args.config_file).stem)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    if comm.is_main_process():
        results = trainer._last_eval_results
        print(f"Validation results: {results}")

        run_name = Path(cfg.OUTPUT_DIR).name
        summary = {"run_name": run_name, "checkpoint": str(Path(cfg.OUTPUT_DIR) / "model_final.pth"), "val": results}
        with open(Path(cfg.OUTPUT_DIR) / "results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        metrics_json = Path(cfg.OUTPUT_DIR) / "metrics.json"
        if metrics_json.exists():
            metrics_png = plot_metrics(metrics_json, Path(cfg.OUTPUT_DIR) / "metrics.png")
            print(f"Wrote training/eval curves to {metrics_png}")

        predictions_json = Path(cfg.OUTPUT_DIR) / "coco_instances_results.json"
        if predictions_json.exists():
            gt_json = Path(MetadataCatalog.get(cfg.DATASETS.TEST[0]).json_file)
            confusion_matrix_png = Path(cfg.OUTPUT_DIR) / "confusion_matrix.png"
            compute_confusion_matrix(predictions_json, gt_json, confusion_matrix_png)
            print(f"Wrote confusion matrix to {confusion_matrix_png}")

        print(f"Run '{run_name}' logged to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--ds-config",
        default="",
        help="Path to a habitat_embodied_al/pretrain/config/ds_hssd.yaml -- switches dataset registration to "
        "habitat_embodied_al's HSSD-collected data instead of coco_testbench's local COCO copy.",
    )
    args = parser.parse_args()
    args.config_file = args.config_file or DEFAULT_CONFIG

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
