"""NYU Depth V2's labeled subset (nyu_depth_v2_labeled.mat: 1,449 RGB-D frames with per-pixel class and instance maps,
MATLAB v7.3 read with h5py; notebooks/nyu_depth_v2_visualization.ipynb) written as a static COCO test set,
<out_dir>/test/test.json and its images in <out_dir>/images/, by common/utils/dataset_writers.py's CocoFramesWriter; habitat_embodied_sl's
--static-test-dirs scores it. Its vocabulary is the dataset's own class names but EXCLUDED_CLASSES, sorted, whichever frames
are written. An instance is a (class, instance index) pair of a frame's maps: NYU counts instances within each class.

Usage (all 1,449 frames; --frames for a subset):
  PYTHONPATH=. python common/env_utils/nyu_depth_v2.py datasets/nyu_depth_v2
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from detectron2.structures import Instances  # type: ignore

from common.utils.dataset_writers import IMAGES_DIR, CocoFramesWriter

NYU_MAT = Path(os.environ.get("HABITAT_DATA", "")) / "nyu_depth_v2" / "nyu_depth_v2_labeled.mat"
EXCLUDED_CLASSES = ("wall", "floor", "ceiling", "unknown")  # stuff, and the catch-all
_INSTANCE_RANGE = 1 << 16  # instance indices are uint16, so label * _INSTANCE_RANGE + instance keys a (class, instance) pair


def class_names(mat: h5py.File) -> list:
    """mat's class names, label 1 first, each stored as an HDF5 reference to a column of uint16 char codes."""
    return ["".join(chr(code) for code in mat[reference][:, 0]) for reference in mat["names"][0]]


def frame_instances(labels: np.ndarray, instances: np.ndarray, class_ids: dict) -> Instances:
    """Ground truth of a frame's (label, instance) pairs whose label is in class_ids, as CocoFramesWriter writes it."""
    keys = labels.astype(np.int64) * _INSTANCE_RANGE + instances
    pairs = np.unique(keys[(instances > 0) & np.isin(labels, list(class_ids))])
    ground_truth = Instances(labels.shape)
    ground_truth.gt_masks = torch.from_numpy(keys[None] == pairs[:, None, None])
    ground_truth.gt_classes = torch.tensor([class_ids[int(pair // _INSTANCE_RANGE)] for pair in pairs], dtype=torch.long)
    ground_truth.infos = [{"object_id": int(pair)} for pair in pairs]
    return ground_truth


def write_test_set(mat_path: Path, frames: list, out_dir: Path) -> Path:
    """Writes mat_path's frames (indices) as out_dir/test, one scene "nyu", and returns that test directory."""
    with h5py.File(mat_path, "r") as mat:
        names = class_names(mat)
        categories = sorted({name for name in names if name not in EXCLUDED_CLASSES})
        category_index = {name: index for index, name in enumerate(categories)}
        class_ids = {label: category_index[name] for label, name in enumerate(names, start=1) if name in category_index}
        writer = CocoFramesWriter(out_dir / "test", {"source": str(mat_path), "frames": frames}, categories, out_dir / IMAGES_DIR)
        writer.write_scene("nyu", [
            (index, np.ascontiguousarray(mat["images"][index].transpose(2, 1, 0)), frame_instances(mat["labels"][index].T, mat["instances"][index].T, class_ids))
            for index in frames
        ])
    writer.merge(np.random.default_rng(0))
    return out_dir / "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="Dataset directory; its test set goes to out_dir/test.")
    parser.add_argument("--frames", type=int, nargs="+", help="Frame indices to write (default: all 1,449).")
    parser.add_argument("--mat", type=Path, default=NYU_MAT)
    args = parser.parse_args()
    with h5py.File(args.mat, "r") as mat:
        frames = args.frames or list(range(mat["images"].shape[0]))
    print(f"Wrote {write_test_set(args.mat, frames, args.out_dir)}")
