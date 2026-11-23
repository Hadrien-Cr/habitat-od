"""Splits pooling several scenes, each in its own folder <split>/: every frame as <images_dir>/<scene>_<frame>.jpg, a
folder in the dataset (datasets/<run_name>/images/), plus <split>/<split>.json and
<split>/<split>_mosaic.jpg, _MOSAIC_RECORDS of the split's records read back from that json and drawn in random order,
and <split>/<split>_class_counts.png, its annotation (or view) count per class stacked by scene.
Each scene is written on its own (write_scene), its records to its shard <split>/scenes/<scene>.json, so scenes can be
written by parallel processes; merge, run once they are all done (habitat_embodied_sl/merge.py), pools every shard into
<split>.json, its "info" the dataset's config. A dataset (build_dataset.py's one variant, collect_test.py's test set)
holds one split.

ObjectViewsWriter writes a train split of object views: {"info", "frames": {"<scene>_<frame>": {"height", "width"}},
"retrieved_objects": {"<scene>/<object>": {"gt_class", "views": [{"frame", "mask" (RLE)}]}}}, read by ObjectViewsDataset.
CocoFramesWriter writes a COCO instance-segmentation split of fully annotated frames, each annotation also carrying its
simulator object_id; its image, annotation and category ids start at 1 (category id = class index + 1), as COCOeval
treats annotation id 0 as unmatched and detectron2 expects categories in [1, #categories]."""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from detectron2.data import DatasetMapper
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Instances
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pycocotools import mask as mask_util

from common.utils.coco_utils import encoded_masks, json_rle, write_json
from common.utils.object_views_dataset import ObjectViewsDataset, batches_mosaic
from common.utils.plot_utils import MOSAIC_JPEG_QUALITY, plot_class_counts
from common.utils.retrieval import RetrievedView, SegmentationStyle

_MOSAIC_RECORDS = 32
_MOSAIC_COLUMNS = 8
_JPEG_QUALITY = 95
SCENES_DIR = "scenes"
IMAGES_DIR = "images"
OBJECT_DS_DIR = "object_ds"
VARIANTS = ("V1_pseudo", "V2_drop_add", "V3_cls_box_drop_add")
FULL_FRAMES = "V4_full"  # the train frames with every ground-truth instance, not object views


def frame_key(scene: str, frame_id: int) -> str:
    return f"{scene}_{frame_id}"


def frame_scene(file_name: str) -> str:
    """The scene of a frame image named after its frame_key."""
    return Path(file_name).stem.rsplit("_", 1)[0]


def _save_frame(images_dir: Path, key: str, rgb: np.ndarray) -> None:
    """Saves rgb as images_dir/<key>.jpg unless already saved (flush_scene deletes a scene's frames before it is rewritten)."""
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
    style = SegmentationStyle.of(categories)
    mapped = [mapper(records[index]) for index in rng.permutation(len(records))[:_MOSAIC_RECORDS]]
    mosaic = batches_mosaic([mapped[start:start + _MOSAIC_COLUMNS] for start in range(0, len(mapped), _MOSAIC_COLUMNS)], style)
    if mosaic is not None:
        mosaic.save(split_dir / f"{split_dir.name}_mosaic.jpg", quality=MOSAIC_JPEG_QUALITY)


def _save_class_counts(records: List[dict], categories: list, split_dir: Path) -> Path:
    """split_dir/<split>_class_counts.png of records' annotation count per class (one per view for an object-view set),
    stacked by scene in name order."""
    pairs = [(frame_scene(record["file_name"]), annotation["category_id"]) for record in records for annotation in record["annotations"]]
    scene_counts = {
        scene: np.bincount([class_id for pair_scene, class_id in pairs if pair_scene == scene], minlength=len(categories))
        for scene in sorted({scene for scene, _ in pairs})
    }
    title = f"{split_dir.parent.name}/{split_dir.name}"
    return plot_class_counts(scene_counts, categories, title, split_dir / f"{split_dir.name}_class_counts.png")


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
        """<split>.json pooling every scene's shard, then its mosaic and class counts; nothing without any shard."""
        shards = _scene_shards(self.split_dir)
        if not shards:
            return
        frames = {key: frame for shard in shards for key, frame in shard["frames"].items()}
        objects = {key: entry for shard in shards for key, entry in shard["retrieved_objects"].items()}
        write_json(split_json_path(self.split_dir), {"info": self.info, "frames": frames, "retrieved_objects": objects})
        records = ObjectViewsDataset.from_json(split_json_path(self.split_dir), self.images_dir).records
        _save_mosaic(records, self.info["categories"], self.split_dir, rng)
        _save_class_counts(records, self.info["categories"], self.split_dir)


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
        """<split>.json pooling every scene's shard, then its mosaic and class counts; nothing without any shard. Categories keep
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
        _save_class_counts(records, self.categories, self.split_dir)


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


def collection_root(cfg: DictConfig) -> Path:
    """A dataset config's folder, <dataset_root>/<run_name>/."""
    return Path(cfg.dataset_root) / cfg.run_name


SplitWriter = Union[CocoFramesWriter, ObjectViewsWriter]


def split_writer(split_dir: Path, cfg: DictConfig) -> SplitWriter:
    """The writer of a dataset's split_dir, over the classes its config cfg records, cfg its info, its frames in the
    dataset's images/: object views for an OBJECT_DS_DIR split, fully annotated frames (COCO) for any other."""
    info, classes, images_dir = {"config": OmegaConf.to_container(cfg, resolve=True)}, list(cfg.classes), split_dir.parent / IMAGES_DIR
    if split_dir.name == OBJECT_DS_DIR:
        return ObjectViewsWriter(split_dir, {"categories": classes, **info}, images_dir)
    return CocoFramesWriter(split_dir, info, classes, images_dir)


def flush_scene(root: Path, scene: str) -> None:
    """Deletes what an earlier write of the scene left in the dataset under root: its folder, its shard, its frames, its
    retrieval mosaics and their parts and its semantic top-down map."""
    shutil.rmtree(root / scene, ignore_errors=True)
    patterns = (
        f"*/{SCENES_DIR}/{scene}.json", f"{IMAGES_DIR}/{scene}_*.jpg", f"retrieval_mosaics/{scene}.jpg", f"retrieval_mosaics/{scene}_*.jpg",
        f"semantic_topdown_maps/{scene}.png",
    )
    for path in [path for pattern in patterns for path in root.glob(pattern)]:
        path.unlink()
