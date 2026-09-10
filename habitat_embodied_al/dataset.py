"""Thin wrapper around this package's own COCO dataset conversion and
registration.
"""
from pathlib import Path
from typing import Any, Optional

from detectron2.data import DatasetCatalog  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore

from habitat_embodied_al import constants
from habitat_embodied_al.coco_writer import build_coco_dataset


def build_dataset(ds_cfg: Any, split_name: str, mode: str = "combine") -> list[str]:
    """Converts DATA_ROOT/ds_cfg.run_name/{split_name}/<scene>/raw/ (one directory per scene,
    produced by one collect_random/collect_validation call per scene -- see collection.py) into
    COCO dataset(s) under DATASET_ROOT, for whichever scenes under this split actually have raw
    data collected.

    mode="combine" pools every scene into one DATASET_ROOT/ds_cfg.run_name/{split_name}.json +
    images -- the shape register_ds_config expects. mode="separate" instead builds one
    independent DATASET_ROOT/<run_name>_<scene>/{split_name}.json per scene, so a checkpoint can
    be scored against each scene on its own (see eval.py).

    Returns the built run_name(s): [ds_cfg.run_name] for "combine", [f"{run_name}_{scene}", ...]
    for "separate" -- or [] if no scene under this split has raw data collected."""
    split_root = constants.DATA_ROOT / ds_cfg.run_name / split_name
    raw_dirs = sorted((p for p in split_root.glob("*/raw") if p.is_dir()), key=lambda p: p.parent.name)
    if not raw_dirs:
        return []

    object_params = ds_cfg.object_params
    if mode == "combine":
        build_coco_dataset(
            [str(d) for d in raw_dirs],
            str(constants.DATASET_ROOT / ds_cfg.run_name),
            split_name,
            object_params["env_name"],
            object_params["vocab_name"],
            object_params.get("filter_classes"),
            filter_empty=ds_cfg.filter_empty,
        )
        return [ds_cfg.run_name]

    if mode == "separate":
        run_names = []
        for raw_dir in raw_dirs:
            run_name = f"{ds_cfg.run_name}_{raw_dir.parent.name}"
            build_coco_dataset(
                [str(raw_dir)],
                str(constants.DATASET_ROOT / run_name),
                split_name,
                object_params["env_name"],
                object_params["vocab_name"],
                object_params.get("filter_classes"),
                filter_empty=ds_cfg.filter_empty,
            )
            run_names.append(run_name)
        return run_names

    raise ValueError(f"Unknown build_dataset mode {mode!r}, expected 'combine' or 'separate'")


def register_dataset(dataset_root: Path, split_name: str, name: Optional[str] = None) -> str:
    """Registers `dataset_root/<split_name>.json` with detectron2, under catalog key `name`
    (defaults to `split_name`). Idempotent: a `name` already in `DatasetCatalog` is left as-is,
    so callers must pick a fresh name whenever the underlying JSON changes. `name` lets several
    datasets that all use the fixed on-disk split_name "val" (e.g. eval.py scoring
    build_dataset(..., mode="separate")'s datasets in one process) register under distinct
    catalog entries instead of colliding under one shared "val"."""
    name = name or split_name
    if name not in DatasetCatalog:
        json_file = str(dataset_root / f"{split_name}.json")
        image_root = str(dataset_root / split_name)
        register_coco_instances(name, {}, json_file, image_root)
        # Forces the otherwise-lazy loader so thing_classes is populated
        # before pretrain.py's NUM_CLASSES lookup runs.
        DatasetCatalog.get(name)
    return name
