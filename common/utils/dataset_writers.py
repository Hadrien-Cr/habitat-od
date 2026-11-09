"""Splits pooling several scenes, each in its own folder <split>/: every frame as <split>/images/<scene>_<frame>.jpg, plus
<split>/<split>.json and <split>/<split>_mosaic.png, _MOSAIC_RECORDS of the split's records read back from that json and
drawn in random order. Each scene is written on its own (write_scene), its records to its shard
<split>/scenes/<scene>.json; merge then pools every shard written so far into <split>.json, so scenes can be written by
parallel processes, each merging under locked() once its scene is done.

ObjectViewsWriter writes a train split of object views: {"info", "frames": {"<scene>_<frame>": {"height", "width"}},
"retrieved_objects": {"<scene>/<object>": {"gt_class", "views": [{"frame", "mask" (RLE)}]}}}, read by ObjectViewsDataset.
CocoFramesWriter writes a COCO instance-segmentation split of fully annotated frames, each annotation also carrying its
simulator object_id; its image, annotation and category ids start at 1 (category id = class index + 1), as COCOeval
treats annotation id 0 as unmatched and detectron2 expects categories in [1, #categories]."""
import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

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
SCENES_DIR = "scenes"


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


def scene_shard_path(split_dir: Path, scene: str) -> Path:
    return split_dir / SCENES_DIR / f"{scene}.json"


def _write_json(path: Path, content: dict) -> None:
    """content as json at path, written to a temporary file first so a concurrent merge never reads it half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as file:
        json.dump(content, file)
    os.replace(temporary, path)


def _scene_shards(split_dir: Path) -> List[dict]:
    """Every scene's shard written to split_dir so far, in scene name order."""
    return [json.loads(path.read_text()) for path in sorted((split_dir / SCENES_DIR).glob("*.json"))]


@contextmanager
def locked(lock_path: Path) -> Iterator[None]:
    """Holds an exclusive lock on lock_path for the block, so parallel processes writing one dataset merge one at a time."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _annotations(ground_truth: Instances) -> List[dict]:
    """One COCO annotation, without ids, per ground-truth instance, its category id its class index + 1."""
    rles = encoded_masks(ground_truth.gt_masks)
    return [
        {
            "category_id": class_id + 1, "segmentation": _json_rle(rle), "bbox": mask_util.toBbox(rle).tolist(), "area": float(mask_util.area(rle)),
            "iscrowd": 0, "object_id": info["object_id"],
        }
        for rle, class_id, info in zip(rles, ground_truth.gt_classes.tolist(), ground_truth.infos)
    ]


def _save_mosaic(records: List[dict], categories: list, split_dir: Path, rng: np.random.Generator) -> None:
    """split_dir/<split>_mosaic.png of _MOSAIC_RECORDS of records in random order, mapper-mapped, _MOSAIC_COLUMNS a row."""
    mapper = DatasetMapper(is_train=True, augmentations=[], image_format="RGB", use_instance_mask=True, instance_mask_format="bitmask")
    style = SegmentationStyle(categories, make_colors(len(categories), seed=0, ctype=0))
    mapped = [mapper(records[index]) for index in rng.permutation(len(records))[:_MOSAIC_RECORDS]]
    mosaic = batches_mosaic([mapped[start:start + _MOSAIC_COLUMNS] for start in range(0, len(mapped), _MOSAIC_COLUMNS)], style)
    if mosaic is not None:
        mosaic.save(split_dir / f"{split_dir.name}_mosaic.png")


@dataclass(frozen=True)
class ObjectViewsWriter:
    split_dir: Path
    info: dict

    def write_scene(self, scene: str, views: List[RetrievedView]) -> None:
        """Writes the frames of the scene's views, and each view's mask under its object to the scene's shard."""
        frames, objects = {}, {}
        for view in views:
            key = frame_key(scene, view.frame_id)
            if key not in frames:
                _save_frame(self.split_dir, key, view.rgb)
                frames[key] = {"height": view.rgb.shape[0], "width": view.rgb.shape[1]}
            entry = objects.setdefault(f"{scene}/{view.object_id}", {"gt_class": view.class_id, "views": []})
            entry["views"].append({"frame": key, "mask": _json_rle(encoded_masks(view.mask[None])[0])})
        _write_json(scene_shard_path(self.split_dir, scene), {"frames": frames, "retrieved_objects": objects})

    def merge(self, rng: np.random.Generator) -> None:
        """<split>.json pooling every scene's shard written so far, then its mosaic."""
        shards = _scene_shards(self.split_dir)
        frames = {key: frame for shard in shards for key, frame in shard["frames"].items()}
        objects = {key: entry for shard in shards for key, entry in shard["retrieved_objects"].items()}
        _write_json(split_json_path(self.split_dir), {"info": self.info, "frames": frames, "retrieved_objects": objects})
        _save_mosaic(ObjectViewsDataset.from_json(split_json_path(self.split_dir)).records, self.info["categories"], self.split_dir, rng)


@dataclass(frozen=True)
class CocoFramesWriter:
    split_dir: Path
    info: dict
    categories: list

    def write_scene(self, scene: str, frames: List[Tuple[int, np.ndarray, Instances]]) -> None:
        """Writes each (frame id, rgb, ground-truth instances) of the scene as an image, and the scene's shard of them, one
        annotation per instance, each pointing at its image by its index in the shard."""
        images, annotations = [], []
        for frame_id, rgb, ground_truth in frames:
            key = frame_key(scene, frame_id)
            _save_frame(self.split_dir, key, rgb)
            annotations += [{"image": len(images), **annotation} for annotation in _annotations(ground_truth)]
            images.append({"file_name": f"{key}.jpg", "height": rgb.shape[0], "width": rgb.shape[1]})
        _write_json(scene_shard_path(self.split_dir, scene), {"images": images, "annotations": annotations})

    def merge(self, rng: np.random.Generator) -> None:
        """<split>.json pooling every scene's shard written so far, then its mosaic. Categories are written alphabetically
        (COCOEvaluator's per-category AP table follows category id order), with annotations' category_id remapped from
        self.categories' original class-index order to match."""
        json_path = split_json_path(self.split_dir)
        images, annotations = _pooled_frames(_scene_shards(self.split_dir))
        old_ids_by_name = sorted(enumerate(self.categories, start=1), key=lambda pair: pair[1])
        old_to_new_id = {old_id: new_id for new_id, (old_id, _) in enumerate(old_ids_by_name, start=1)}
        sorted_classes = [name for _, name in old_ids_by_name]
        categories = [{"id": new_id, "name": name} for new_id, name in enumerate(sorted_classes, start=1)]
        annotations = [{**annotation, "category_id": old_to_new_id[annotation["category_id"]]} for annotation in annotations]
        _write_json(json_path, {"info": self.info, "categories": categories, "images": images, "annotations": annotations})
        records = load_coco_json(str(json_path), str(self.split_dir / IMAGES_DIR), dataset_name=str(json_path))
        _save_mosaic(records, sorted_classes, self.split_dir, rng)


def _pooled_frames(shards: List[dict]) -> Tuple[List[dict], List[dict]]:
    """(images, annotations) of CocoFramesWriter's shards, their image and annotation ids numbered from 1 across shards."""
    images, annotations = [], []
    for shard in shards:
        image_ids = range(len(images) + 1, len(images) + 1 + len(shard["images"]))
        images += [{"id": image_id, **image} for image_id, image in zip(image_ids, shard["images"])]
        annotations += [
            {"id": len(annotations) + index + 1, "image_id": image_ids[annotation["image"]], **{key: value for key, value in annotation.items() if key != "image"}}
            for index, annotation in enumerate(shard["annotations"])
        ]
    return images, annotations
