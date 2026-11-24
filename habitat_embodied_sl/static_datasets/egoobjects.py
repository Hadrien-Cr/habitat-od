"""EgoObjects' eval split (INSTALL.MD's EgoObjects section; an LVIS-format json: boxes, and per image neg_category_ids and not_exhaustive_category_ids) as a
static COCO test set of detection_sources.py: its classes CLIP-matched to --vocab, each box written as its filled rectangle
(box_frame_instances). EgoObjects is federated, so each category unverified on an image gets an ignored whole-image
iscrowd box there (detection_sources.py's federated).

Usage (every eval image showing a matched class; --max-images for a subset):
  BASE_DIR=$(pwd) PYTHONPATH=. python habitat_embodied_sl/static_datasets/egoobjects.py datasets/egoobjects_eval --vocab "trash can" sofa \
      --sim-word-thresh 0.9
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from detectron2.structures import BoxMode  # type: ignore

from common.vision.detic import Vocabulary
from habitat_embodied_sl.static_datasets.detection_sources import Source, box_frame_instances, federated, source_parser, write_test_set

EGOOBJECTS_ROOT = Path(os.environ.get("HABITAT_DATA", "")) / "egoobjects"


def egoobjects_source(json_path: Path, image_root: Path) -> Source:
    """json_path's records, every category id (its own, from 1, and the federated lists') as its index in id order."""
    content = json.loads(json_path.read_text())
    categories = sorted(content["categories"], key=lambda category: category["id"])
    class_index = {category["id"]: index for index, category in enumerate(categories)}
    annotations = defaultdict(list)
    for annotation in content["annotations"]:
        annotations[annotation["image_id"]].append({"bbox": annotation["bbox"], "bbox_mode": BoxMode.XYWH_ABS, "category_id": class_index[annotation["category_id"]]})
    records = [
        {
            "file_name": str(image_root / image["file_name"]), "height": image["height"], "width": image["width"], "image_id": image["id"],
            "annotations": annotations[image["id"]],
            "neg_category_ids": [class_index[category_id] for category_id in image["neg_category_ids"]],
            "not_exhaustive_category_ids": [class_index[category_id] for category_id in image["not_exhaustive_category_ids"]],
        }
        for image in content["images"]
    ]
    return Source(Vocabulary("egoobjects", tuple(category["name"] for category in categories)), records, federated(box_frame_instances))


if __name__ == "__main__":
    parser = source_parser(__doc__)
    parser.add_argument("--json", type=Path, default=EGOOBJECTS_ROOT / "data" / "EgoObjectsV1_unified_eval.json")
    parser.add_argument("--image-root", type=Path, default=EGOOBJECTS_ROOT / "images", help="The json's file_names are relative to it.")
    args = parser.parse_args()
    print(f"Wrote {write_test_set(egoobjects_source(args.json, args.image_root), args)}")
