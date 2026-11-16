"""Pools a collection's per-scene shards, which habitat_embodied_sl/collect.py's calls write one scene each, into
datasets/<run_name>/train/train.json, test/test.json and every variant's <variant>/object_ds/object_ds.json, each with its
mosaic. Run once every scene's call is done, and again after recollecting a scene; a split without any shard is skipped.

Usage:
  PYTHONPATH=. python habitat_embodied_sl/merge.py habitat_embodied_sl/config/procthor8_collection.yaml
"""
import argparse

import numpy as np
from omegaconf import OmegaConf

from common.env_utils.object_annotations import kept_classes_from_args
from common.utils.dataset_writers import dataset_writers


def main(config: str) -> None:
    cfg = OmegaConf.load(config)
    for split, writer in dataset_writers(cfg, list(kept_classes_from_args(cfg.object_params))).items():
        writer.merge(np.random.default_rng(cfg.seed))
        print(f"Merged {split} ({writer.split_dir}), if it has any shard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Collection config: run_name, budgets, object_params.")
    main(parser.parse_args().config)
