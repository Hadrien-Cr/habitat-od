"""Generic detectron2 training entry point, structured to mirror
third_party/detectron2/tools/train_net.py directly (setup/Trainer/main/
invoke_main, default_argument_parser, default_setup, launch). Trains a plain COCO architecture
from scratch/a COCO checkpoint (e.g. BoxSup-C2_L_CLIP_SwinB_896b32_4x) -- for CustomFT_Detic
fine-tuning on ObjectViewsDataset, use train.py instead.

Dataset registration (register_coco), the evaluator-wired Trainer, and final-checkpoint
reporting (report_results) live in common/utils/detectron_trainer.py -- shared with train.py
rather than duplicated. The only additions over train_net.py: OUTPUT_DIR routing into
<config-file's-parent-dir>/logs/<run_name>/ (train_net.py leaves OUTPUT_DIR to the
config/opts) -- e.g. coco_testbench/config/foo.yaml logs to coco_testbench/logs/foo/ -- and the
training loop itself.

Registers coco_testbench's local COCO copy (train2017/val2017, hardlink-copied from an existing
~/Detic checkout) under fixed coco_testbench_{train2017,val2017} names, which
coco_testbench/config/Base-RCNN-FPN.yaml's DATASETS.TRAIN/TEST already point at. Used to
reproduce detectron2's own COCO-Detection model-zoo baselines
(https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) verbatim -- a way to
confirm this repo's environment/detectron2 checkout reproduces published numbers.

--config-file is a small overlay (_BASE_: "Base-RCNN-FPN.yaml") mirroring one of detectron2's
own COCO-Detection/*.yaml model-zoo configs, pointed at a local Base-RCNN-FPN.yaml copy instead
of third_party/detectron2/configs/Base-RCNN-FPN.yaml (see that file's header for why). The
dataset registers with evaluator_type "coco" (register_coco_instances), so build_evaluator
always scores with COCOEvaluator.

No manual post-training evaluation call: DefaultTrainer.build_hooks()
already registers an EvalHook(cfg.TEST.EVAL_PERIOD, ...) with
eval_after_train=True, so a final evaluator pass already runs
automatically after the last iteration -- re-running it here would just
duplicate a full pass over the val set. trainer._last_eval_results
(populated by that hook) is used directly for report_results instead.

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
"""
from pathlib import Path

import detectron2.utils.comm as comm  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.engine import default_argument_parser, default_setup, launch  # type: ignore

from common.utils.detectron_trainer import Trainer, register_coco, report_results
from common.utils.plot_utils import plot_metrics

DEFAULT_CONFIG = "coco_testbench/config/faster_rcnn_R_50_FPN_1x.yaml"


def setup(args):
    register_coco("train2017")
    register_coco("val2017")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(Path(args.config_file).resolve().parent.parent / "logs" / Path(args.config_file).stem)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    if comm.is_main_process():
        checkpoint = str(Path(cfg.OUTPUT_DIR) / "model_final.pth")
        report_results(cfg, list(cfg.DATASETS.TEST), trainer._last_eval_results, checkpoint)

        metrics_json = Path(cfg.OUTPUT_DIR) / "metrics.json"
        if metrics_json.exists():
            metrics_png = plot_metrics(metrics_json, Path(cfg.OUTPUT_DIR) / "metrics.png")
            print(f"Wrote training/eval curves to {metrics_png}")

        print(f"Run '{Path(cfg.OUTPUT_DIR).name}' logged to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = default_argument_parser()
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
