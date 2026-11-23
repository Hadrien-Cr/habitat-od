"""Scores the resolved maps of every scene habitat_embodied_sl/build_dataset.py wrote into a dataset, datasets/<run_name>/, each
<scene>/predictions.json against its <scene>/ground_truth.json, every scene pooled (eval_utils.py's evaluate_scene_jsons:
3D box AP, TIDE error breakdown, class-agnostic matching, confusion matrix), into datasets/<run_name>/map_eval/.

Usage:
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  PYTHONPATH=. python habitat_embodied_sl/evaluate_maps.py datasets/procthor8_V2_drop_add
"""
import argparse
from pathlib import Path

from common.utils.eval_utils import PREDICTIONS_JSON, evaluate_scene_jsons


def main(dataset_dir: Path) -> None:
    evaluate_scene_jsons(sorted(path.parent for path in dataset_dir.glob(f"*/{PREDICTIONS_JSON}")), dataset_dir / "map_eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="A dataset folder build_dataset.py wrote, datasets/<run_name>/.")
    main(parser.parse_args().dataset_dir)
