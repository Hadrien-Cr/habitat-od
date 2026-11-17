"""Splits pooling several scenes, each in its own folder <split>/: every frame as <images_dir>/<scene>_<frame>.jpg, a
folder the dataset's splits share (collect.py's datasets/<run_name>/images/), plus <split>/<split>.json and
<split>/<split>_mosaic.jpg, _MOSAIC_RECORDS of the split's records read back from that json and drawn in random order.
Each scene is written on its own (write_scene), its records to its shard <split>/scenes/<scene>.json, so scenes can be
written by parallel processes; merge, run once they are all done (habitat_embodied_sl/merge.py), pools every shard into
<split>.json.

ObjectViewsWriter writes a train split of object views: {"info", "frames": {"<scene>_<frame>": {"height", "width"}},
"retrieved_objects": {"<scene>/<object>": {"gt_class", "views": [{"frame", "mask" (RLE)}]}}}, read by ObjectViewsDataset.
CocoFramesWriter writes a COCO instance-segmentation split of fully annotated frames, each annotation also carrying its
simulator object_id; its image, annotation and category ids start at 1 (category id = class index + 1), as COCOeval
treats annotation id 0 as unmatched and detectron2 expects categories in [1, #categories]."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from detectron2.data import DatasetMapper
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Instances
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pycocotools import mask as mask_util

from common import constants
from common.env_utils.vocab_constants import make_colors
from common.utils.coco_utils import encoded_masks, json_rle, write_json
from common.utils.object_views_dataset import ObjectViewsDataset, batches_mosaic
from common.utils.plot_utils import MOSAIC_JPEG_QUALITY
from common.utils.retrieval import RetrievedView, SegmentationStyle, ViewSelection

_MOSAIC_RECORDS = 32
_MOSAIC_COLUMNS = 8
_JPEG_QUALITY = 95
SCENES_DIR = "scenes"
IMAGES_DIR = "images"
OBJECT_DS_DIR = "object_ds"
VARIANTS = ("pseudo", "gt", "gt++")


def frame_key(scene: str, frame_id: int) -> str:
    return f"{scene}_{frame_id}"


def frame_scene(file_name: str) -> str:
    """The scene of a frame image named after its frame_key."""
    return Path(file_name).stem.rsplit("_", 1)[0]


def _save_frame(images_dir: Path, key: str, rgb: np.ndarray) -> None:
    """Saves rgb as images_dir/<key>.jpg unless another split of the dataset already did (collect.py flushes a scene's
    frames before recollecting it)."""
    path = images_dir / f"{key}.jpg"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=_JPEG_QUALITY)


def split_json_path(split_dir: Path) -> Path:
    return split_dir / f"{split_dir.name}.json"


def scene_shard_path(split_dir: Path, scene: str) -> Path:
    return split_dir / SCENES_DIR / f"{scene}.json"


def _scene_shards(split_dir: Path) -> List[dict]:
    """Every scene's shard written to split_dir, in scene name order."""
    return [json.loads(path.read_text()) for path in sorted((split_dir / SCENES_DIR).glob("*.json"))]


def _annotations(ground_truth: Instances) -> List[dict]:
    """One COCO annotation, without ids, per ground-truth instance, its category id its class index + 1."""
    rles = encoded_masks(ground_truth.gt_masks)
    return [
        {
            "category_id": class_id + 1, "segmentation": json_rle(rle), "bbox": mask_util.toBbox(rle).tolist(), "area": float(mask_util.area(rle)),
            "iscrowd": 0, "object_id": info["object_id"],
        }
        for rle, class_id, info in zip(rles, ground_truth.gt_classes.tolist(), ground_truth.infos)
    ]


def _save_mosaic(records: List[dict], categories: list, split_dir: Path, rng: np.random.Generator) -> None:
    """split_dir/<split>_mosaic.jpg of _MOSAIC_RECORDS of records in random order, mapper-mapped, _MOSAIC_COLUMNS a row."""
    mapper = DatasetMapper(is_train=True, augmentations=[], image_format="RGB", use_instance_mask=True, instance_mask_format="bitmask")
    style = SegmentationStyle(categories, make_colors(len(categories), seed=0, ctype=0))
    mapped = [mapper(records[index]) for index in rng.permutation(len(records))[:_MOSAIC_RECORDS]]
    mosaic = batches_mosaic([mapped[start:start + _MOSAIC_COLUMNS] for start in range(0, len(mapped), _MOSAIC_COLUMNS)], style)
    if mosaic is not None:
        mosaic.save(split_dir / f"{split_dir.name}_mosaic.jpg", quality=MOSAIC_JPEG_QUALITY)


@dataclass(frozen=True)
class ObjectViewsWriter:
    split_dir: Path
    info: dict
    images_dir: Path

    def write_scene(self, scene: str, views: List[RetrievedView]) -> None:
        """Writes the frames of the scene's views, and each view's mask under its object to the scene's shard."""
        frames, objects = {}, {}
        for view in views:
            key = frame_key(scene, view.frame_id)
            if key not in frames:
                _save_frame(self.images_dir, key, view.rgb)
                frames[key] = {"height": view.rgb.shape[0], "width": view.rgb.shape[1]}
            entry = objects.setdefault(f"{scene}/{view.object_id}", {"gt_class": view.class_id, "views": []})
            entry["views"].append({"frame": key, "mask": json_rle(encoded_masks(view.mask[None])[0])})
        write_json(scene_shard_path(self.split_dir, scene), {"frames": frames, "retrieved_objects": objects})

    def merge(self, rng: np.random.Generator) -> None:
        """<split>.json pooling every scene's shard, then its mosaic; nothing without any shard."""
        shards = _scene_shards(self.split_dir)
        if not shards:
            return
        frames = {key: frame for shard in shards for key, frame in shard["frames"].items()}
        objects = {key: entry for shard in shards for key, entry in shard["retrieved_objects"].items()}
        write_json(split_json_path(self.split_dir), {"info": self.info, "frames": frames, "retrieved_objects": objects})
        _save_mosaic(ObjectViewsDataset.from_json(split_json_path(self.split_dir), self.images_dir).records, self.info["categories"], self.split_dir, rng)


@dataclass(frozen=True)
class CocoFramesWriter:
    split_dir: Path
    info: dict
    categories: list
    images_dir: Path

    def write_scene(self, scene: str, frames: List[Tuple[int, np.ndarray, Instances]]) -> None:
        """Writes each (frame id, rgb, ground-truth instances) of the scene as an image, and the scene's shard of them, one
        annotation per instance, each pointing at its image by its index in the shard."""
        images, annotations = [], []
        for frame_id, rgb, ground_truth in frames:
            key = frame_key(scene, frame_id)
            _save_frame(self.images_dir, key, rgb)
            annotations += [{"image": len(images), **annotation} for annotation in _annotations(ground_truth)]
            images.append({"file_name": f"{key}.jpg", "height": rgb.shape[0], "width": rgb.shape[1]})
        write_json(scene_shard_path(self.split_dir, scene), {"images": images, "annotations": annotations})

    def merge(self, rng: np.random.Generator) -> None:
        """<split>.json pooling every scene's shard, then its mosaic; nothing without any shard. Categories keep
        self.categories' order, the train sets' (ObjectViewsWriter) class order, so a class index means the same class in both."""
        shards = _scene_shards(self.split_dir)
        if not shards:
            return
        json_path = split_json_path(self.split_dir)
        images, annotations = _pooled_frames(shards)
        categories = [{"id": category_id, "name": name} for category_id, name in enumerate(self.categories, start=1)]
        write_json(json_path, {"info": self.info, "categories": categories, "images": images, "annotations": annotations})
        records = load_coco_json(str(json_path), str(self.images_dir), dataset_name=str(json_path))
        _save_mosaic(records, self.categories, self.split_dir, rng)


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


def dataset_writers(cfg: DictConfig, classes: list) -> Dict[str, Union[CocoFramesWriter, ObjectViewsWriter]]:
    """{split: its writer} of collect.py's datasets/<run_name>/: train/ (the train frames' full ground truth), test/, and
    every variant's <variant>/object_ds/, all over classes, their frames in the shared images/."""
    root = constants.DATASET_ROOT / cfg.run_name
    retrieval = {**asdict(ViewSelection(k=cfg.collection.k_retrieval)), "min_mesh_visibility_fraction": cfg.object_params.min_mesh_visibility_fraction}
    test_info = {"evaluation": OmegaConf.to_container(cfg.evaluation), "frontier_agent": OmegaConf.to_container(cfg.frontier_agent)}
    return {
        "train": CocoFramesWriter(root / "train", {"collection": OmegaConf.to_container(cfg.collection)}, classes, root / IMAGES_DIR),
        "test": CocoFramesWriter(root / "test", test_info, classes, root / IMAGES_DIR),
        **{variant: ObjectViewsWriter(root / variant / OBJECT_DS_DIR, {"categories": classes, "retrieval": retrieval}, root / IMAGES_DIR) for variant in VARIANTS},
    }
