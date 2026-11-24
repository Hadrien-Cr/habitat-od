"""LVIS v1 val (common/vision/detic.py's register_lvis_v1, under Detic's datasets/) as a static COCO test set of
detection_sources.py: its classes CLIP-matched to --vocab (lvis_class_mosaic.py draws them on LVIS train), each object
written as its polygons' mask. LVIS is federated, so each category unverified on an image gets an ignored whole-image
iscrowd box there (detection_sources.py's federated).

Usage (every val image showing a matched class; --max-images for a subset):
  BASE_DIR=$(pwd) PYTHONPATH=. python habitat_embodied_sl/static_datasets/lvis_val.py datasets/lvis_val --vocab "trash can" sofa \
      --sim-word-thresh 0.9
"""
import numpy as np
import torch
from detectron2.data import DatasetCatalog  # type: ignore
from detectron2.structures import Instances, polygons_to_bitmask  # type: ignore

from common.vision.detic import LVIS, register_lvis_v1
from habitat_embodied_sl.static_datasets.detection_sources import Source, federated, source_parser, write_test_set


def polygon_instances(record: dict, image_size: tuple, class_ids: dict) -> Instances:
    """Ground truth of record's objects whose class is in class_ids, each its polygons' mask."""
    kept = [(index, annotation) for index, annotation in enumerate(record["annotations"]) if annotation["category_id"] in class_ids]
    masks = [polygons_to_bitmask(annotation["segmentation"], *image_size) for _, annotation in kept]
    ground_truth = Instances(image_size)
    ground_truth.gt_masks = torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, *image_size), dtype=torch.bool)
    ground_truth.gt_classes = torch.tensor([class_ids[annotation["category_id"]] for _, annotation in kept], dtype=torch.long)
    ground_truth.infos = [{"object_id": index} for index, _ in kept]
    return ground_truth


def lvis_val_records() -> list:
    """LVIS v1 val's records, their neg_category_ids and not_exhaustive_category_ids (LVIS ids, from 1) as class indices."""
    register_lvis_v1(["lvis_v1_val"])
    return [
        dict(record, neg_category_ids=[index - 1 for index in record["neg_category_ids"]],
             not_exhaustive_category_ids=[index - 1 for index in record["not_exhaustive_category_ids"]])
        for record in DatasetCatalog.get("lvis_v1_val")
    ]


if __name__ == "__main__":
    args = source_parser(__doc__).parse_args()
    print(f"Wrote {write_test_set(Source(LVIS, lvis_val_records(), federated(polygon_instances)), args)}")
