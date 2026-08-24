"""Thin wrapper around this package's own COCO dataset conversion and
registration.
"""
from pathlib import Path

from detectron2.data import DatasetCatalog  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore

from habitat_embodied_al.coco_writer import build_coco_dataset


def build_dataset(raw_dataset_path: Path, dataset_root: Path, split_name: str, object_params: dict, filter_empty: bool = False) -> Path:
    """Converts a raw collected sense dir into a COCO split under `dataset_root`."""
    json_path = build_coco_dataset(
        str(raw_dataset_path),
        str(dataset_root),
        split_name,
        object_params["vocab_name"],
        object_params["filter_out_classes"],
        filter_empty=filter_empty,
    )
    return Path(json_path)


def register_dataset(dataset_root: Path, split_name: str) -> str:
    """Registers `dataset_root/<split_name>.json` with detectron2, keyed by
    `split_name`. Idempotent: a `split_name` already in `DatasetCatalog` is
    left as-is, so callers must pick a fresh name whenever the underlying
    JSON changes."""
    if split_name not in DatasetCatalog:
        json_file = str(dataset_root / f"{split_name}.json")
        image_root = str(dataset_root / split_name)
        register_coco_instances(split_name, {}, json_file, image_root)
        # Forces the otherwise-lazy loader so thing_classes is populated
        # before pretrain.py's NUM_CLASSES lookup runs.
        DatasetCatalog.get(split_name)
    return split_name
