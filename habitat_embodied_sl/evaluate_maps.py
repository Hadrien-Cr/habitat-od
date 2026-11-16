"""Scores the resolved maps of every train scene habitat_embodied_sl/collect.py wrote under datasets/<run_name>/, each
<scene>/predictions.json against its <scene>/ground_truth.json, every scene pooled (eval_utils.py's evaluate_scene_jsons:
3D box AP, class-agnostic matching, confusion matrix), into datasets/<run_name>/map_eval/.

Usage:
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_embodied_sl/evaluate_maps.py habitat_embodied_sl/config/procthor8_collection.yaml
"""
import argparse

from omegaconf import OmegaConf

from common import constants
from common.utils.eval_utils import PREDICTIONS_JSON, evaluate_scene_jsons


def main(args: argparse.Namespace) -> None:
    root = constants.DATASET_ROOT / OmegaConf.load(args.config).run_name
    evaluate_scene_jsons(sorted(path.parent for path in root.glob(f"*/{PREDICTIONS_JSON}")), root / "map_eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="The collection config whose run_name names the dataset to score.")
    main(parser.parse_args())
