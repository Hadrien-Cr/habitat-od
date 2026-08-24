"""Collects pretraining data over many HSSD-HAB scenes and builds a
COCO-format dataset from it -- the data half of a pretrain run (see
repo-root pretrain.py --ds-config, run afterwards against the same
--config, for the train half).

--config points at a habitat_embodied_al/pretrain/config/ds_hssd.yaml: run_name
(keys datasets/<run_name>/) + object_params (vocab, env_name -- see
common/env_utils/env_registry.py::resolve_env for which env_names are
supported) + train_scenes/val_scenes/steps_per_episode/trainer_name/
filter_empty (collection params). train_scenes may be omitted/empty for an
eval-only run (e.g. a val-only dataset built just to score a pretrained
model against, not to also fine-tune on) -- that split is then skipped
entirely rather than collected with zero scenes.

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
from habitat_embodied_al.collection import collect_raw
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

    raw_root = constants.DATA_ROOT / ds_cfg.run_name
    dataset_dir = constants.DATASET_ROOT / ds_cfg.run_name

    for split_name, scenes in [("train", ds_cfg.get("train_scenes", [])), ("val", ds_cfg.val_scenes)]:
        if not scenes:
            continue

        habitat_config = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
        raw_dir = raw_root / "raw" / split_name

        collect_raw(
            habitat_config,
            scenes=list(scenes),
            steps_per_episode=ds_cfg.steps_per_episode,
            trainer_name=ds_cfg.trainer_name,
            object_params=ds_cfg.object_params,
            out_dir=raw_dir,
            create_mosaic=True,
        )
        build_dataset(raw_dir, dataset_dir, split_name, ds_cfg.object_params, filter_empty=ds_cfg.filter_empty)

        if not args.keep_raw:
            shutil.rmtree(raw_dir)

    print(f"Dataset for run '{ds_cfg.run_name}' built at {dataset_dir}")
