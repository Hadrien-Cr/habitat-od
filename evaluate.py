"""Evaluates a Detic config and checkpoint (MODEL.WEIGHTS, in the trailing opts) on a COCO test split written by
habitat_embodied_sl/main.py (datasets/<run_name>/test/, see README.md), with Detic's own train_net.py do_test and its
COCOEvaluator. Detic's zero-shot classifier is reset to the CLIP text embeddings of the split's categories, prompted as
`a <class>`. Results go to <test-dir>/inference/<config-file's stem>/.

Usage:
  PYTHONPATH=. python evaluate.py --test-dir datasets/smoke_sl/test \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
"""
import json
from pathlib import Path

from detectron2.data.datasets import register_coco_instances  # type: ignore
from detectron2.engine import default_argument_parser, default_setup, launch  # type: ignore

from common.utils.dataset_writers import split_json_path
from common.utils.object_views_dataset import IMAGES_DIR
from common.vision.detic import Vocabulary, build_vocabulary_model, detic_cfg
from train_net import do_test  # type: ignore  # Detic's, importable once common.vision.detic is


def register_test_split(test_dir: Path) -> Vocabulary:
    """Registers test_dir's COCO json as <run_name>_<split>; returns its categories as a Vocabulary of the same name."""
    name = f"{test_dir.parent.name}_{test_dir.name}"
    register_coco_instances(name, {}, str(split_json_path(test_dir)), str(test_dir / IMAGES_DIR))
    categories = json.loads(split_json_path(test_dir).read_text())["categories"]
    return Vocabulary(name, tuple(category["name"] for category in categories))


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
    vocabulary = register_test_split(Path(args.test_dir))
    cfg = setup(args, vocabulary)
    return do_test(cfg, build_vocabulary_model(cfg, vocabulary))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--test-dir", default="datasets/smoke_sl/test", help="A COCO split folder: <split>.json and images/.")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
