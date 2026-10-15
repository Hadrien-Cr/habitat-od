"""Trains a Detic config from the repo root with Detic's own train_net.py loop (do_train/do_test), its relative
paths resolved against third_party/Detic and its LVIS v1 splits read from third_party/Detic/datasets (laid out by
common/vision/setup_detic_data.sh). OUTPUT_DIR is <config-file's-parent-dir>/logs/<stem>/, as in the root pretrain.py.
IMS_PER_BATCH is the global batch size, so --num-gpus only changes wall-clock time; on this host --num-gpus > 1
needs NCCL_P2P_DISABLE=1 (see the root pretrain.py).

Usage (reproduces Detic's BoxSup-C2_L_CLIP_SwinB_896b32_4x LVIS baseline):
  NCCL_P2P_DISABLE=1 PYTHONPATH=. python habitat_embodied_sl/pretrain.py \
      --config-file common/vision/config/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml --num-gpus 4
"""
from pathlib import Path

import detectron2.utils.comm as comm  # type: ignore
from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
from detectron2.engine import default_argument_parser, default_setup, launch  # type: ignore
from detectron2.modeling import build_model  # type: ignore
from detectron2.utils.logger import setup_logger  # type: ignore
from torch.nn.parallel import DistributedDataParallel

from common.vision.detic import detic_cfg, register_lvis_v1
from train_net import do_test, do_train  # type: ignore  # Detic's, importable once common.vision.detic is


def setup(args):
    cfg = detic_cfg(args.config_file, args.opts)
    cfg.OUTPUT_DIR = str(Path(args.config_file).resolve().parent.parent / "logs" / Path(args.config_file).stem)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detic")
    register_lvis_v1(cfg.DATASETS.TRAIN + cfg.DATASETS.TEST)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )
    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
