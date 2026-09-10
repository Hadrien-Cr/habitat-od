"""Collects a *distinct* validation set -- built to score an already
pretrained checkpoint's generalization to unseen scenes, not to also train
on -- the counterpart of embodied-active-learning-od's
validation_data_create.py. Unlike collect_dataset.py's broad exploration
policy, walks straight to each target object in turn (see
collection.py::collect_validation).

Builds one independent COCO dataset per ds_cfg.val_scenes entry
(datasets/<run_name>_<scene>/) instead of pooling them into a single
val.json, so eval.py can score a checkpoint against each scene separately.

Usage:
  PYTHONPATH=. python habitat_embodied_al/collect_validation_dataset.py \
      --config habitat_embodied_al/reproduce/config/ds_procthor_reproduce_valdistinct.yaml
"""
import argparse
import shutil

import habitat  # type: ignore
from omegaconf import OmegaConf

from habitat_embodied_al import constants
from habitat_embodied_al.collection import collect_validation
from habitat_embodied_al.dataset import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the collected raw sense files after building the dataset from them (deleted by default).",
    )
    args = parser.parse_args()
    ds_cfg = OmegaConf.load(args.config)

    habitat_config = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
    for scene_name in ds_cfg.val_scenes:
        collect_validation(habitat_config, ds_cfg, "val", scene_name)
    run_names = build_dataset(ds_cfg, "val", mode="separate")

    if not args.keep_raw:
        shutil.rmtree(constants.DATA_ROOT / ds_cfg.run_name, ignore_errors=True)

    for run_name in run_names:
        print(f"Validation dataset '{run_name}' built at {constants.DATASET_ROOT / run_name}")
