"""
Standalone viewer for common/utils/concept_graph.py::ConceptGraphObjectSet.

Reuses test_semantic_point_cloud.py's FrontierAgent-driven write/read harness. Each fused
frame's gt_instances (GT masks, depth and one-hot class logits) get a CLIP feature of their padded
box crop, are projected with project_detections_to_objects, merged into the map via
ConceptGraphObjectSet.__add__, and .resolve() runs concept-graphs' denoise/filter/overlap-merge.

Has no pytest test functions - run it directly.
"""

from functools import partial

import numpy as np
from detectron2.structures import Instances

from common.utils.concept_graph import ConceptGraphObjectSet, project_detections_to_objects
from common.utils.similarity_utils import ClipCropEncoder
from test_semantic_point_cloud import HFOV_DEG, MAX_DEPTH, MIN_DEPTH, run_viewer, viewer_arg_parser


def frame_objects(encoder: ClipCropEncoder, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray, gt_instances: Instances) -> ConceptGraphObjectSet:
    gt_instances.clip_features = encoder.encode_crops(rgb, gt_instances.gt_boxes.tensor.numpy())
    return project_detections_to_objects(depth, pose, gt_instances, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)


if __name__ == "__main__":
    parser = viewer_arg_parser(__doc__, "ConceptGraphObjectSet")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP checkpoint used for the per-detection features")
    args = parser.parse_args()
    run_viewer(args, "test_concept_graph", ConceptGraphObjectSet(), partial(frame_objects, ClipCropEncoder(args.clip_model)))
