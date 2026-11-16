"""Evaluates a Detic config and checkpoint (MODEL.WEIGHTS, in the trailing opts) on a COCO test split written by
habitat_embodied_sl/collect.py (datasets/<run_name>/test/, see README.md), with Detic's own train_net.py do_test and its
COCOEvaluator. Detic's zero-shot classifier is reset to the CLIP text embeddings of the split's categories, prompted as
`a <class>`. Results go to <test-dir>/inference/<config-file's stem>/.

Usage:
  PYTHONPATH=. python evaluate.py --test-dir datasets/procthor_medium/test \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
"""
from pathlib import Path

from detectron2.engine import default_argument_parser, default_setup, launch  # type: ignore

from common.env_utils.dataset_registry import register_dataset
from common.vision.detic import Vocabulary, build_vocabulary_model, dataset_vocabulary, detic_cfg
from train_net import do_test  # type: ignore  # Detic's, importable once common.vision.detic is


def setup(args, vocabulary: Vocabulary):
    cfg = detic_cfg(args.config_file, args.opts)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(vocabulary.classes)
    cfg.MODEL.RESET_CLS_TESTS = False
    cfg.DATASETS.TEST = (vocabulary.name,)
    cfg.OUTPUT_DIR = str(Path(args.test_dir) / "inference" / Path(args.config_file).stem)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    vocabulary = dataset_vocabulary(register_dataset(Path(args.test_dir)))
    cfg = setup(args, vocabulary)
    return do_test(cfg, build_vocabulary_model(cfg, vocabulary))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--test-dir", default="datasets/procthor_medium/test", help="A COCO split folder: <split>.json, its images in ../images/.")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
