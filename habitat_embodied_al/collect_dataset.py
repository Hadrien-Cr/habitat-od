"""Collects pretraining data over many HSSD-HAB scenes and builds a
COCO-format dataset from it -- the data half of a pretrain run (see
repo-root pretrain.py --ds-config, run afterwards against the same
--config, for the train half).

--config points at a habitat_embodied_al/pretrain/config/ds_hssd.yaml: run_name
(keys datasets/<run_name>/) + object_params (vocab, env_name -- see
common/env_utils/env_registry.py::resolve_env for which env_names are
supported) + train_scenes/val_scenes/steps_per_episode/filter_empty
(collection params). train_scenes may be omitted/empty for an eval-only run
(e.g. a val-only dataset built just to score a pretrained model against, not
to also fine-tune on) -- that split is then skipped entirely rather than
collected with zero scenes.

Writes to datasets/<run_name>/: train.json/val.json + images, plus
GT-overlay mosaics. The raw collected rgb+bbsgt sense files are deleted by
default once the dataset is built from them (see --keep-raw) -- they're
large (tens of GB) and, since object_params is baked into the sensor at
collection time, not reusable across a different ds_hssd.yaml anyway.

Usage:
  PYTHONPATH=. python habitat_embodied_al/collect_dataset.py \
      --config habitat_embodied_al/pretrain/config/ds_hssd.yaml
"""
import argparse
import shutil

import habitat  # type: ignore
from omegaconf import OmegaConf

from habitat_embodied_al import constants
from habitat_embodied_al.collection import collect_random
from habitat_embodied_al.dataset import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="habitat_embodied_al/pretrain/config/ds_hssd.yaml")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the collected raw sense files after building the dataset from them (deleted by default).",
    )
    args = parser.parse_args()
    ds_cfg = OmegaConf.load(args.config)

    habitat_config = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")

    for split_name, scenes in [("train", ds_cfg.get("train_scenes", [])), ("val", ds_cfg.get("val_scenes", []))]:
        for scene_name in scenes:
            collect_random(habitat_config, ds_cfg, split_name, scene_name)
        build_dataset(ds_cfg, split_name, mode="combine")

    if not args.keep_raw:
        shutil.rmtree(constants.DATA_ROOT / ds_cfg.run_name, ignore_errors=True)

    print(f"Dataset for run '{ds_cfg.run_name}' built at {constants.DATASET_ROOT / ds_cfg.run_name}")
