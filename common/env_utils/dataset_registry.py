"""Registers built dataset splits (common/utils/dataset_writers.py shape: <dataset>/<split>/<split>.json, its images in
<dataset>/images/) with detectron2, lazily, each as <dataset folder>_<split>, e.g. procthor8_test_test or
procthor8_V2_drop_add_object_ds, so configs name them in DATASETS.TRAIN and DATASETS.TEST: fully annotated frames as
COCO instances, each annotation keeping its object_id, and object views (an OBJECT_DS_DIR split) as ObjectViewsDataset's
records, their metadata's object_views set. A split's thing_classes are set once it is loaded (load_dataset)."""
import json
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data.datasets import load_coco_json  # type: ignore

from common.utils.dataset_writers import IMAGES_DIR, OBJECT_DS_DIR, split_json_path
from common.utils.object_views_dataset import ObjectViewsDataset
from common.utils.retrieval import SegmentationStyle

DATASETS_ROOT = Path("datasets")


def register_dataset(split_dir: Path) -> str:
    """Registers split_dir's <split>.json as <its parent directory's name>_<its name>, so the splits of different datasets
    don't collide; its name. Idempotent: a name already in DatasetCatalog is left as-is."""
    name, json_path, images_dir = f"{split_dir.parent.name}_{split_dir.name}", split_json_path(split_dir), split_dir.parent / IMAGES_DIR
    if name in DatasetCatalog:
        return name
    if split_dir.name == OBJECT_DS_DIR:
        DatasetCatalog.register(name, lambda: _object_view_records(name, json_path, images_dir))
        MetadataCatalog.get(name).set(object_views=True)
    else:
        DatasetCatalog.register(name, lambda: load_coco_json(str(json_path), str(images_dir), name, extra_annotation_keys=["object_id"]))
        MetadataCatalog.get(name).set(json_file=str(json_path), image_root=str(images_dir), evaluator_type="coco")
    return name


def _object_view_records(name: str, json_path: Path, images_dir: Path) -> list:
    """ObjectViewsDataset's records of json_path, its categories set as name's thing_classes."""
    MetadataCatalog.get(name).set(thing_classes=json.loads(json_path.read_text())["info"]["categories"])
    return ObjectViewsDataset.from_json(json_path, images_dir).records


def register_datasets(root: Path = DATASETS_ROOT) -> None:
    """register_dataset of every split of every dataset under root."""
    for json_path in sorted(root.glob("*/*/*.json")):
        if json_path.stem == json_path.parent.name:
            register_dataset(json_path.parent)


def load_dataset(name: str) -> str:
    """Loads the registered split name, which sets its thing_classes, and gives each class a color for drawing; its name."""
    DatasetCatalog.get(name)
    meta = MetadataCatalog.get(name)
    if not meta.get("thing_colors"):
        meta.thing_colors = SegmentationStyle.of(meta.thing_classes).colors
    return name
