"""Objects365 v2 val (INSTALL.MD's Objects365 section: its COCO json with fixed class names, read with detectron2's load_coco_json) as a static COCO test set of detection_sources.py:
its classes CLIP-matched to --vocab, each box written as its filled rectangle (box_frame_instances), crowd boxes as iscrowd.
Objects365 annotates every object of its 365 classes, so no class is unverified.

Usage (every val image showing a matched class; --max-images for a subset):
  BASE_DIR=$(pwd) PYTHONPATH=. python habitat_embodied_sl/static_datasets/objects365.py datasets/objects365_val --vocab "trash can" sofa \
      --sim-word-thresh 0.9 --max-images 5000
"""
from pathlib import Path

from detectron2.data import MetadataCatalog  # type: ignore
from detectron2.data.datasets import load_coco_json  # type: ignore

from common.vision.detic import DETIC_ROOT, Vocabulary
from habitat_embodied_sl.static_datasets.detection_sources import Source, box_frame_instances, source_parser, write_test_set

OBJECTS365_ROOT = DETIC_ROOT / "datasets" / "objects365"
_DATASET_NAME = "objects365_v2_val_static"  # its own name, so its metadata never collides with Detic's objects365 splits


def objects365_source(json_path: Path, image_root: Path) -> Source:
    """json_path's records, their category ids as class indices."""
    records = load_coco_json(str(json_path), str(image_root), _DATASET_NAME)
    return Source(Vocabulary("objects365", tuple(MetadataCatalog.get(_DATASET_NAME).thing_classes)), records, box_frame_instances)


if __name__ == "__main__":
    parser = source_parser(__doc__)
    parser.add_argument("--json", type=Path, default=OBJECTS365_ROOT / "annotations" / "zhiyuan_objv2_val_fixname.json")
    parser.add_argument("--image-root", type=Path, default=OBJECTS365_ROOT / "val", help="The json's file_names are relative to it.")
    args = parser.parse_args()
    print(f"Wrote {write_test_set(objects365_source(args.json, args.image_root), args)}")
