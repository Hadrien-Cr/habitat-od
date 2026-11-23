"""Pools a dataset's per-scene shards, which habitat_embodied_sl/build_dataset.py's or collect_test.py's calls write one
scene each, into its split's json (object_ds/object_ds.json, train/train.json or test/test.json), mosaic and class counts,
its classes and info those of the dataset's config.yaml. Run once every scene's call is done, and again after rewriting a
scene.

Usage:
  PYTHONPATH=. python habitat_embodied_sl/merge.py datasets/procthor8_V2_drop_add
"""
import argparse
from pathlib import Path

import numpy as np

from common.utils.dataset_writers import SCENES_DIR, split_writer
from common.utils.raw_scene import load_run_config

_MOSAIC_SEED = 0  # only picks which records each split's mosaic previews


def main(dataset_dir: Path) -> None:
    cfg = load_run_config(dataset_dir)
    for split_dir in sorted(shards.parent for shards in dataset_dir.glob(f"*/{SCENES_DIR}")):
        split_writer(split_dir, cfg).merge(np.random.default_rng(_MOSAIC_SEED))
        print(f"Merged {split_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="A dataset folder, datasets/<run_name>/, holding its config.yaml.")
    main(parser.parse_args().dataset_dir)
