"""Registers an already-built COCO split directory (common/utils/dataset_writers.py shape:
<split dir>/<split_name>.json, its images in the dataset's images folder next to the split directory) with detectron2."""
from pathlib import Path

from detectron2.data import DatasetCatalog  # type: ignore
from detectron2.data.datasets import register_coco_instances  # type: ignore

from common.utils.dataset_writers import IMAGES_DIR, split_json_path


def register_dataset(split_dir: Path) -> str:
    """Registers split_dir's <split>.json with detectron2 as <its parent directory's name>_<its name>, e.g.
    procthor_medium_test, so the splits of different datasets don't collide. Idempotent: a name already in DatasetCatalog
    is left as-is."""
    name = f"{split_dir.parent.name}_{split_dir.name}"
    if name not in DatasetCatalog:
        register_coco_instances(name, {}, str(split_json_path(split_dir)), str(split_dir.parent / IMAGES_DIR))
        # Forces the otherwise-lazy loader so thing_classes is populated before it is read.
        DatasetCatalog.get(name)
    return name
