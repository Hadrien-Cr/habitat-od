"""Converts a directory of raw collected habitat frames (rgb + bbsgt sense
files) into a plain COCO instance-detection dataset: one JPEG per frame plus
a COCO-format JSON annotation file, loadable as-is via detectron2's own
`load_coco_json` -- no custom loader needed.

`categories` only lists the kept (non-`filter_out_classes`) vocab, so it
stays consistent with what the model actually trains against; a top-level
"vocab" field traces the full vocab it was filtered from.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from PIL import Image

from common.utils.dataset_utils import SampleLoader

RARE_MAX_IMAGES = 10
COMMON_MAX_IMAGES = 100


def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().astype(float).tolist()
        if len(contour) >= 6 and len(contour) % 2 == 0:
            polygons.append(contour)
    return polygons


def build_coco_dataset(
    raw_dataset_path: str,
    dataset_root: str,
    split_name: str,
    vocab_name: str,
    filter_out_classes: list,
    filter_empty: bool = False,
) -> str:
    """Writes `dataset_root/{split_name}/*.jpg` + `dataset_root/{split_name}.json`
    from raw sense files under `raw_dataset_path`. Returns the json path.

    `filter_empty` drops frames with zero valid annotations (no GT object in
    view, or all masks failed to yield a polygon) instead of writing them in
    as empty-annotation images."""

    full_classes = MetadataCatalog.get(vocab_name).thing_classes
    kept_classes = [c for c in full_classes if c not in filter_out_classes]
    full_id_to_kept_id = {i: kept_classes.index(name) for i, name in enumerate(full_classes) if name in kept_classes}

    dataset_root_path = Path(dataset_root)
    image_root = dataset_root_path / split_name
    os.makedirs(image_root, exist_ok=True)

    sampler = SampleLoader(str(raw_dataset_path))
    episodes, steps = sampler.get_episode_and_steps_dense_list()

    images = []
    annotations = []
    image_ids_per_category = defaultdict(set)
    instance_count_per_category = defaultdict(int)

    image_id = 0
    annotation_id = 0

    for episode, step in zip(episodes.tolist(), steps.tolist()):
        gt_instances = sampler.get_sample(episode, 0, "bbsgt", step).get_bbs_as_gt()  # type: ignore

        gt_boxes = gt_instances.gt_boxes.tensor.numpy()
        gt_classes = gt_instances.gt_classes.numpy()

        frame_annotations = []
        for box, class_id, mask in zip(gt_boxes, gt_classes, gt_instances.gt_masks):
            kept_id = full_id_to_kept_id.get(int(class_id))
            if kept_id is None:
                continue

            polygons = _mask_to_polygons(mask.cpu().numpy())
            if not polygons:
                continue

            x1, y1, x2, y2 = box.tolist()
            frame_annotations.append({
                "category_id": kept_id + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": polygons,
                "area": float(mask.sum()),
                "iscrowd": 0,
            })

        if filter_empty and not frame_annotations:
            continue

        rgb = sampler.get_sample(episode, 0, "rgb", step).data[:, :, :3]  # type: ignore
        image_id += 1
        file_name = f"{episode:06d}_{step:05d}.jpg"
        Image.fromarray(rgb).save(image_root / file_name, quality=95)

        height, width = gt_instances.image_size
        images.append({
            "id": image_id,
            "file_name": file_name,
            "height": int(height),
            "width": int(width),
        })

        for ann in frame_annotations:
            annotation_id += 1
            category_id = ann["category_id"]
            annotations.append({"id": annotation_id, "image_id": image_id, **ann})
            image_ids_per_category[category_id].add(image_id)
            instance_count_per_category[category_id] += 1

    categories = []
    for idx, class_name in enumerate(kept_classes):
        category_id = idx + 1
        image_count = len(image_ids_per_category[category_id])

        if image_count <= RARE_MAX_IMAGES:
            frequency = "r"
        elif image_count <= COMMON_MAX_IMAGES:
            frequency = "c"
        else:
            frequency = "f"

        categories.append({
            "id": category_id,
            "name": class_name,
            "frequency": frequency,
            "image_count": image_count,
            "instance_count": instance_count_per_category[category_id],
        })

    json_path = dataset_root_path / f"{split_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "vocab": {"name": vocab_name, "full_classes": full_classes, "filter_out_classes": sorted(filter_out_classes)},
        }, f)

    return str(json_path)
