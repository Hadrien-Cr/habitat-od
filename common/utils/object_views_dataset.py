"""Train set of object views, read from an ObjectViewsWriter train.json (common/utils/dataset_writers.py): one detectron2
record per view, its frame's image left on disk and its mask as RLE, the record's single annotation labelled with its
object's class. Views are drawn with replacement at N_c ** -temperature, N_c the number of objects of their class, and mapped
to model inputs by a detectron2 DatasetMapper."""
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from detectron2.data.build import trivial_batch_collator
from detectron2.data.common import MapDataset
from detectron2.structures import BoxMode
from PIL import Image
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common.utils.plot_utils import rows_mosaic
from common.utils.retrieval import SegmentationStyle, segmentation_tile

IMAGES_DIR = "images"


class ObjectViewsDataset(Dataset):
    records: List[dict]

    def __init__(self, records: List[dict]):
        self.records = records

    @classmethod
    def from_json(cls, json_path: Path) -> "ObjectViewsDataset":
        """One record per view of every object of json_path, its image under the images folder next to it."""
        data = json.loads(json_path.read_text())
        return cls([
            _view_record(json_path.parent / IMAGES_DIR, data["frames"][view["frame"]], object_key, entry["gt_class"], view)
            for object_key, entry in data["retrieved_objects"].items() for view in entry["views"]
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def sample_weights(self, temperature: float) -> torch.Tensor:
        """Each view's N_c ** -temperature, N_c the number of objects of its class."""
        object_classes = {record["object_key"]: record["annotations"][0]["category_id"] for record in self.records}
        class_objects = Counter(object_classes.values())
        return torch.tensor([class_objects[object_classes[record["object_key"]]] ** -temperature for record in self.records], dtype=torch.double)


def _view_record(image_root: Path, frame: dict, object_key: str, class_id: int, view: dict) -> dict:
    annotation = {
        "bbox": mask_util.toBbox(view["mask"]).tolist(), "bbox_mode": BoxMode.XYWH_ABS, "category_id": class_id,
        "segmentation": view["mask"], "iscrowd": 0,
    }
    return {
        "file_name": str(image_root / f"{view['frame']}.jpg"), "height": frame["height"], "width": frame["width"],
        "image_id": view["frame"], "object_key": object_key, "annotations": [annotation],
    }


def balanced_loader(dataset: ObjectViewsDataset, batch_size: int, mapper: Callable[[dict], dict], temperature: float = 0.5) -> DataLoader:
    """Full batches of dataset's views mapped by mapper, drawn with replacement by their sample_weights, len(dataset) views an epoch."""
    sampler = WeightedRandomSampler(dataset.sample_weights(temperature), num_samples=len(dataset), replacement=True)
    return DataLoader(MapDataset(dataset, mapper), batch_size=batch_size, sampler=sampler, collate_fn=trivial_batch_collator, drop_last=True)


def batches_mosaic(batches: Iterable[List[dict]], style: SegmentationStyle) -> Optional[Image.Image]:
    """One mosaic row per batch of mapper-mapped records, each its rgb with its annotations' boxes and classes."""
    return rows_mosaic([[_record_tile(record, style) for record in batch] for batch in batches])


def _record_tile(record: dict, style: SegmentationStyle) -> tuple:
    instances = record["instances"]
    instances.remove("gt_masks")
    frame = Path(record["file_name"]).stem
    label = f"{record['object_key']} @ {frame}" if "object_key" in record else frame
    return segmentation_tile(np.ascontiguousarray(record["image"].permute(1, 2, 0).numpy()), instances, style, label)
