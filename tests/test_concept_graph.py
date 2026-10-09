"""
Standalone viewer for common/utils/concept_graph.py::ConceptGraphObjectSet.

Reuses semantic_point_cloud_viewer.py's write/read harness. Each frame's gt_instances (GT masks,
depth and one-hot class logits) get a CLIP feature of their padded box crop, are projected with
project_detections_to_objects, merged into the map via ConceptGraphObjectSet.__add__, and
.resolve() runs concept-graphs' denoise/filter/overlap-merge.

Not collected by pytest - run it directly.
"""

import argparse
import os
import time

import numpy as np
from detectron2.structures import Instances

from common.utils.concept_graph import ConceptGraphObjectSet, project_detections_to_objects
from common.utils.similarity_utils import ClipCropEncoder
from semantic_point_cloud_viewer import (
    HFOV_DEG, MAX_DEPTH, MIN_DEPTH, fusion_states, observe_frame, run_viewer, save_input_mosaic,
    save_point_cloud, save_segmentation, viewer_arg_parser,
)


def frame_objects(encoder: ClipCropEncoder, gt_instances: Instances, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> ConceptGraphObjectSet:
    gt_instances.clip_features = encoder.encode_crops(rgb, gt_instances.gt_boxes.tensor.numpy())
    return project_detections_to_objects(depth, pose, gt_instances, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)


def fuse_concept_graph_frames(encoder: ClipCropEncoder, viewer, args: argparse.Namespace) -> ConceptGraphObjectSet:
    fused, input_tiles = ConceptGraphObjectSet(), []
    for i, state in enumerate(fusion_states(viewer, args)):
        rgb, depth, pose, gt_instances = observe_frame(viewer, state, args.classes_keep)
        input_tiles.append(save_segmentation(viewer, rgb, gt_instances, f"frame{i:02d}"))
        if len(gt_instances):
            st = time.time()
            fused = fused + frame_objects(encoder, gt_instances, rgb, depth, pose)
            print(f"frame {i:02d}: encode+project+match {len(gt_instances)} detections in {time.time() - st:.3f}s, {len(fused)} map objects")
        if len(fused) == 0:
            continue
        st = time.time()
        fused.resolve()
        print(f"step {i + 1}: resolve in {time.time() - st:.3f}s, {len(fused.resolved_points)} resolved points")
        step_label = "step1" if i == 0 else f"step1:{i + 1}"
        if len(fused.resolved_points):
            save_point_cloud(fused, viewer.colors, os.path.join(viewer.out_dir, f"{viewer.out_stem}_{step_label}.ply"))
    save_input_mosaic(viewer, input_tiles)
    return fused


if __name__ == "__main__":
    parser = viewer_arg_parser(__doc__, "ConceptGraphObjectSet")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP checkpoint used for the per-detection features")
    args = parser.parse_args()
    encoder = ClipCropEncoder(args.clip_model)
    run_viewer(args, "concept_graph_viewer", lambda viewer, args: fuse_concept_graph_frames(encoder, viewer, args))
