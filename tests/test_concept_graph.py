"""
Standalone viewer for common/utils/concept_graph.py::ConceptGraphObjectSet.

Reuses test_semantic_point_cloud.py's FrontierAgent-driven write/read harness. Each fused
frame's gt_instances (GT masks and depth) get the Detic region embedding of their box, are projected with project_detections_to_objects, merged into the map via
ConceptGraphObjectSet.__add__, and .resolve() runs concept-graphs' denoise/filter/overlap-merge, then
Detector.classify_objects classifies each object from its raytraced masks in every explored frame.
Then saves the same retrieval_mosaic.png as test_semantic_point_cloud.py.

Has no pytest test functions - run it directly.
"""

from functools import partial
from pathlib import Path
from typing import List

import numpy as np
from detectron2.structures import Instances

from common.env_utils.object_annotations import resolve_classes
from common.utils.concept_graph import ConceptGraphObject, ConceptGraphObjectSet, project_detections_to_objects
from common.utils.image_memory import ImageMemory
from common.utils.projection_utils import DepthCamera, ResolvedObjects
from detector import Detector
from test_semantic_point_cloud import _ENV_NAME, _VOCAB_NAME, HFOV_DEG, MAX_DEPTH, MIN_DEPTH, run_viewer, save_retrieval_mosaic, viewer_arg_parser

_MIN_VISIBILITY_FRACTION = 0.2


def frame_objects(detector: Detector, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray, gt_instances: Instances) -> List[ConceptGraphObject]:
    gt_instances.region_features = detector.encode_regions(rgb, gt_instances.gt_boxes.tensor.numpy())
    return project_detections_to_objects(depth, pose, gt_instances, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)


def resolve_and_classify(detector: Detector, mosaic_path: Path, fused: ConceptGraphObjectSet, memory: ImageMemory) -> ResolvedObjects:
    fused.resolve()
    return detector.classify_resolved(fused, memory, DepthCamera(HFOV_DEG, MIN_DEPTH, MAX_DEPTH), _MIN_VISIBILITY_FRACTION, mosaic_path)


if __name__ == "__main__":
    parser = viewer_arg_parser(__doc__, "ConceptGraphObjectSet")
    parser.add_argument("--config-file", required=True, help="Detic config whose region embeddings are fused")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam-checkpoint", default="data/sam/sam_vit_b_01ec64.pth")
    args = parser.parse_args()
    detector = Detector(args.config_file, args.checkpoint, resolve_classes(_ENV_NAME, _VOCAB_NAME), args.sam_checkpoint)
    mosaic_path = Path(__file__).parent / "testdump" / "test_concept_graph" / "resolved_trial_mosaic.png"
    viewer, fused, memory = run_viewer(
        args, "test_concept_graph", ConceptGraphObjectSet(), partial(frame_objects, detector), partial(resolve_and_classify, detector, mosaic_path),
    )
    save_retrieval_mosaic(viewer, fused, memory, args.retrieval_objects)
