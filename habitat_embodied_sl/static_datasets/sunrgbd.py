"""SUN RGB-D's 2D boxes (SUNRGBDMeta2DBB_v2.mat's groundtruth2DBB, read with scipy; notebooks/sunrgbd_visualization.ipynb)
written as a static COCO test set, <out_dir>/test/test.json and its images in <out_dir>/images/, by
common/utils/dataset_writers.py's CocoFramesWriter; registered as <out_dir's name>_test (common/env_utils/dataset_registry.py),
a training config's DATASETS.TEST scores it. Its vocabulary is
vocabulary.py's frequent classes of the written frames' boxes but EXCLUDED_CLASSES, sorted; the other classes'
boxes are left out. Each box is written as its
filled rectangle (box_instances). The NYU Depth V2 frames SUN RGB-D includes are left out
by default: nyu_depth_v2.py writes them as their own test set.

Usage (every non-NYU frame; --frames for a subset):
  PYTHONPATH=. python habitat_embodied_sl/static_datasets/sunrgbd.py datasets/sunrgbd
"""
import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from detectron2.structures import Instances  # type: ignore
from PIL import Image

from common.utils.dataset_writers import IMAGES_DIR, CocoFramesWriter, box_instances
from habitat_embodied_sl.static_datasets.vocabulary import MIN_OCCURRENCES, frequent_classes

SUNRGBD_ROOT = Path(os.environ.get("HABITAT_DATA", "")) / "sunrgbd"
EXCLUDED_CLASSES = ("wall", "floor", "ceiling", "otherprop", "unknown")  # stuff, and the catch-alls


def frame_boxes(entry) -> list:
    """entry's groundtruth2DBB as a list: squeeze_me collapses a 1-box struct array to the struct, a 0-box one to an array."""
    boxes = entry.groundtruth2DBB
    return list(boxes) if isinstance(boxes, np.ndarray) else [boxes]


def frame_instances(boxes: list, image_size: tuple, class_ids: dict) -> Instances:
    """Ground truth of a frame's boxes whose raw class name is in class_ids and that overlap the image, as CocoFramesWriter
    writes it."""
    kept = [(index, box) for index, box in enumerate(boxes) if box.classname in class_ids]
    corners = np.array([box.gtBb2D for _, box in kept], dtype=float).reshape(-1, 4)
    return box_instances(corners, [class_ids[box.classname] for _, box in kept], [{"object_id": index} for index, _ in kept], image_size)


def write_test_set(meta: np.ndarray, root: Path, frames: list, out_dir: Path) -> Path:
    """Writes meta's frames (indices) as out_dir/test, one scene "sunrgbd", and returns that test directory."""
    boxes = {index: frame_boxes(meta[index]) for index in frames}
    classes = frequent_classes((box.classname for frame in boxes.values() for box in frame), EXCLUDED_CLASSES)
    categories = sorted(set(classes.values()))
    class_ids = {name: categories.index(category) for name, category in classes.items()}
    info = {"source": str(root), "frames": frames, "min_occurrences": MIN_OCCURRENCES}
    writer = CocoFramesWriter(out_dir / "test", info, categories, out_dir / IMAGES_DIR)
    rgbs = ((index, np.asarray(Image.open(root / meta[index].rgbpath[meta[index].rgbpath.index("SUNRGBD"):]).convert("RGB"))) for index in frames)
    writer.write_scene("sunrgbd", ((index, rgb, frame_instances(boxes[index], rgb.shape[:2], class_ids)) for index, rgb in rgbs))
    writer.merge(np.random.default_rng(0))
    return out_dir / "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="Dataset directory; its test set goes to out_dir/test.")
    parser.add_argument("--frames", type=int, nargs="+", help="Frame indices to write (default: every non-NYU frame).")
    parser.add_argument("--root", type=Path, default=SUNRGBD_ROOT, help="Holds SUNRGBD/ and SUNRGBDMeta2DBB_v2.mat.")
    args = parser.parse_args()
    meta = sio.loadmat(args.root / "SUNRGBDMeta2DBB_v2.mat", struct_as_record=False, squeeze_me=True)["SUNRGBDMeta2DBB"]
    frames = args.frames or [index for index, entry in enumerate(meta) if "/NYUdata/" not in entry.rgbpath]
    print(f"Wrote {write_test_set(meta, args.root, frames, args.out_dir)}")
