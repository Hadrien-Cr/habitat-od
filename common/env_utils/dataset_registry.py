"""Registers an already-built COCO dataset directory (common/utils/dataset_writers.py shape:
dataset_root/<split_name>.json + dataset_root/images/) with detectron2."""
from pathlib import Path
from typing import Optional

from detectron2.data import DatasetCatalog  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore

from common.utils.object_views_dataset import IMAGES_DIR


def register_dataset(dataset_root: Path, split_name: str, name: Optional[str] = None) -> str:
    """Registers `dataset_root/<split_name>.json` with detectron2, under catalog key `name`
    (defaults to `split_name`). Idempotent: a `name` already in `DatasetCatalog` is left as-is,
    so callers must pick a fresh name whenever the underlying JSON changes. `name` lets several
    datasets that all use the fixed on-disk split_name "val" register under distinct catalog
    entries instead of colliding under one shared "val"."""
    name = name or split_name
    if name not in DatasetCatalog:
        json_file = str(dataset_root / f"{split_name}.json")
        image_root = str(dataset_root / IMAGES_DIR)
        register_coco_instances(name, {}, json_file, image_root)
        # Forces the otherwise-lazy loader so thing_classes is populated
        # before pretrain.py's NUM_CLASSES lookup runs.
        DatasetCatalog.get(name)
    return name
