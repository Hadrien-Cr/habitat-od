"""Splits pooling several scenes, each in its own folder <split>/: every frame as <split>/images/<scene>_<frame>.jpg, plus
<split>/<split>.json and <split>/<split>_mosaic.png, _MOSAIC_RECORDS of the split's records read back from that json and
drawn in random order.

ObjectViewsWriter writes a train split of object views: {"info", "frames": {"<scene>_<frame>": {"height", "width"}},
"retrieved_objects": {"<scene>/<object>": {"gt_class", "views": [{"frame", "mask" (RLE)}]}}}, read by ObjectViewsDataset.
CocoFramesWriter writes a COCO instance-segmentation split of fully annotated frames, each annotation also carrying its
simulator object_id; its image, annotation and category ids start at 1 (category id = class index + 1), as COCOeval
treats annotation id 0 as unmatched and detectron2 expects categories in [1, #categories]."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from detectron2.data import DatasetMapper
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Instances
from PIL import Image
from pycocotools import mask as mask_util

from common.env_utils.vocab_constants import make_colors
from common.utils.ground_truth_recovery import encoded_masks
from common.utils.object_views_dataset import IMAGES_DIR, ObjectViewsDataset, batches_mosaic
from common.utils.retrieval import RetrievedView, SegmentationStyle

_MOSAIC_RECORDS = 32
_MOSAIC_COLUMNS = 8
_JPEG_QUALITY = 95


def frame_key(scene: str, frame_id: int) -> str:
    return f"{scene}_{frame_id}"


def _json_rle(rle: dict) -> dict:
    return dict(rle, counts=rle["counts"].decode("ascii"))


def _save_frame(split_dir: Path, key: str, rgb: np.ndarray) -> None:
    path = split_dir / IMAGES_DIR / f"{key}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=_JPEG_QUALITY)


def split_json_path(split_dir: Path) -> Path:
    return split_dir / f"{split_dir.name}.json"


def _save_mosaic(records: List[dict], categories: list, split_dir: Path, rng: np.random.Generator) -> None:
    """split_dir/<split>_mosaic.png of _MOSAIC_RECORDS of records in random order, mapper-mapped, _MOSAIC_COLUMNS a row."""
    mapper = DatasetMapper(is_train=True, augmentations=[], image_format="RGB", use_instance_mask=True, instance_mask_format="bitmask")
    style = SegmentationStyle(categories, make_colors(len(categories), seed=0, ctype=0))
    mapped = [mapper(records[index]) for index in rng.permutation(len(records))[:_MOSAIC_RECORDS]]
    mosaic = batches_mosaic([mapped[start:start + _MOSAIC_COLUMNS] for start in range(0, len(mapped), _MOSAIC_COLUMNS)], style)
    if mosaic is not None:
        mosaic.save(split_dir / f"{split_dir.name}_mosaic.png")


@dataclass
class ObjectViewsWriter:
    split_dir: Path
    info: dict
    frames: dict = field(default_factory=dict)
    objects: dict = field(default_factory=dict)

    def add_scene(self, scene: str, views: List[RetrievedView]) -> None:
        """Writes the frames of the scene's views and records each view's mask under its object."""
        for view in views:
            key = frame_key(scene, view.frame_id)
            if key not in self.frames:
                _save_frame(self.split_dir, key, view.rgb)
                self.frames[key] = {"height": view.rgb.shape[0], "width": view.rgb.shape[1]}
            entry = self.objects.setdefault(f"{scene}/{view.object_id}", {"gt_class": view.class_id, "views": []})
            entry["views"].append({"frame": key, "mask": _json_rle(encoded_masks(view.mask[None])[0])})

    def write(self, rng: np.random.Generator) -> None:
        with open(split_json_path(self.split_dir), "w") as file:
            json.dump({"info": self.info, "frames": self.frames, "retrieved_objects": self.objects}, file)
        _save_mosaic(ObjectViewsDataset.from_json(split_json_path(self.split_dir)).records, self.info["categories"], self.split_dir, rng)


@dataclass
class CocoFramesWriter:
    split_dir: Path
    info: dict
    categories: list
    images: List[dict] = field(default_factory=list)
    annotations: List[dict] = field(default_factory=list)

    def add_scene(self, scene: str, frames: List[Tuple[int, np.ndarray, Instances]]) -> None:
        """Writes each (frame id, rgb, ground-truth instances) of the scene as an image with one annotation per instance."""
        for frame_id, rgb, ground_truth in frames:
            key, image_id = frame_key(scene, frame_id), len(self.images) + 1
            _save_frame(self.split_dir, key, rgb)
            self.images.append({"id": image_id, "file_name": f"{key}.jpg", "height": rgb.shape[0], "width": rgb.shape[1]})
            rles = encoded_masks(ground_truth.gt_masks)
            self.annotations += [
                {
                    "id": len(self.annotations) + index + 1, "image_id": image_id, "category_id": class_id + 1,
                    "segmentation": _json_rle(rle), "bbox": mask_util.toBbox(rle).tolist(), "area": float(mask_util.area(rle)),
                    "iscrowd": 0, "object_id": info["object_id"],
                }
                for index, (rle, class_id, info) in enumerate(zip(rles, ground_truth.gt_classes.tolist(), ground_truth.infos))
            ]

    def write(self, rng: np.random.Generator) -> None:
        """Categories are written alphabetically (COCOEvaluator's per-category AP table follows
        category id order), with annotations' category_id remapped from self.categories' original
        class-index order to match."""
        json_path = split_json_path(self.split_dir)
        old_ids_by_name = sorted(enumerate(self.categories, start=1), key=lambda pair: pair[1])
        old_to_new_id = {old_id: new_id for new_id, (old_id, _) in enumerate(old_ids_by_name, start=1)}
        sorted_classes = [name for _, name in old_ids_by_name]
        categories = [{"id": new_id, "name": name} for new_id, name in enumerate(sorted_classes, start=1)]
        annotations = [{**annotation, "category_id": old_to_new_id[annotation["category_id"]]} for annotation in self.annotations]
        with open(json_path, "w") as file:
            json.dump({"info": self.info, "categories": categories, "images": self.images, "annotations": annotations}, file)
        records = load_coco_json(str(json_path), str(self.split_dir / IMAGES_DIR), dataset_name=str(json_path))
        _save_mosaic(records, sorted_classes, self.split_dir, rng)
